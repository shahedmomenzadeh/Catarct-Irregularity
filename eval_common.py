"""
eval_common.py
Unified evaluation infrastructure and shared utilities across all VLM models for Cataract Irregularity Evaluation.
Handles GPU VRAM management, progressive frame-retry, resumable execution,
and strictly evaluates questions one by one.
"""

import os
import gc
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, List, Dict, Any, Set
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from tqdm import tqdm
from evaluator import score_irregularity_task, aggregate_irregularity_metrics

log = logging.getLogger("eval_common")


# =============================================================================
# 1. GPU & SYSTEM UTILITIES
# =============================================================================

def vram_stats(label: str = "") -> str:
    """Returns a string describing allocated and reserved VRAM across all CUDA devices."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return "CUDA unavailable"
    lines = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        res = torch.cuda.memory_reserved(i) / (1024 ** 3)
        lines.append(f"GPU{i}: alloc={alloc:.1f}GB res={res:.1f}GB")
    tag = f" [{label}]" if label else ""
    return "  ".join(lines) + tag


def flush_memory(*objs) -> None:
    """Deletes objects, triggers garbage collection, and clears CUDA memory cache."""
    for o in objs:
        del o
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def first_device(model) -> Any:
    """Returns the primary device of a model (supporting device_map='auto')."""
    if not HAS_TORCH:
        return "cpu"
    hf_map = getattr(model, "hf_device_map", {})
    if hf_map:
        for key in ("model.embed_tokens", "transformer.wte", "lm_head", "visual"):
            if key in hf_map:
                dev = hf_map[key]
                if dev == "cpu":
                    return torch.device("cpu")
                return torch.device(f"cuda:{dev}" if isinstance(dev, int) else dev)
        first_val = next(iter(hf_map.values()))
        if first_val == "cpu":
            return torch.device("cpu")
        return torch.device(f"cuda:{first_val}" if isinstance(first_val, int) else first_val)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_inputs_to_device(inputs: dict, device: Any) -> dict:
    """Moves all tensors in an inputs dictionary to the designated device and casts floats to float16."""
    if not HAS_TORCH:
        return inputs
    moved = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            moved[k] = v
            continue
        v = v.to(device)
        if v.is_floating_point():
            v = v.to(torch.float16)
        moved[k] = v
    return moved


def probe_total_frames(video_path: str) -> Optional[int]:
    """Returns total frame count of a video via fast metadata probing with decord or opencv."""
    try:
        import decord
        vr = decord.VideoReader(video_path, num_threads=1)
        return len(vr)
    except Exception:
        pass

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return cnt if cnt > 0 else None
    except Exception:
        return None


# =============================================================================
# 2. RESUME TRACKING & JSONL I/O
# =============================================================================

def get_processed_ids(file_path: str, id_key: str = "record_id") -> Set[str]:
    """Extracts a set of processed record_ids from a JSONL results file for resume support."""
    processed = set()
    if not os.path.exists(file_path):
        return processed
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rec_id = row.get(id_key)
                if rec_id:
                    processed.add(rec_id)
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def write_jsonl(file_handle, record: dict) -> None:
    """Appends a single JSON record to an open file and flushes buffer."""
    file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    file_handle.flush()


# =============================================================================
# 3. GENERIC EVALUATION EXECUTION LOOP
# =============================================================================

def run_evaluation_loop(
    generate_fn: Callable[[str, str, str], Optional[str]],
    records: List[Dict[str, Any]],
    output_dir: str,
    tag: str,
    args: Any,
    logger=None
) -> dict:
    """
    Generic execution loop for cataract irregularity evaluation:
      - Iterates strictly one-by-one over each record.
      - Dispatches deterministic scoring immediately (zero external LLM judge dependency).
      - Produces {tag}_responses.jsonl, {tag}_scores.jsonl, and {tag}_summary.json.

    Args:
        generate_fn: Callable(video_path: str, question_text: str, log_id: str) -> str | None
        records: List of records from dataset_loader.
        output_dir: Directory where results will be written.
        tag: Run tag identifier.
        args: Command-line arguments namespace.
        logger: Optional logger instance.
    """
    _log = logger or log
    os.makedirs(output_dir, exist_ok=True)
    responses_path = os.path.join(output_dir, f"{tag}_responses.jsonl")
    scores_path = os.path.join(output_dir, f"{tag}_scores.jsonl")
    summary_path = os.path.join(output_dir, f"{tag}_summary.json")

    # Check for resuming
    processed_ids = get_processed_ids(scores_path, "record_id")
    if processed_ids:
        _log.info(f"Resuming evaluation: {len(processed_ids)} records already scored.")

    n_ok = n_skip = n_error = 0
    concurrency = max(1, getattr(args, "concurrency", 1))

    resp_f = open(responses_path, "a", encoding="utf-8")
    score_f = open(scores_path, "a", encoding="utf-8")
    write_lock = threading.Lock()

    def process_single_record(record: dict, pbar: tqdm) -> None:
        nonlocal n_ok, n_skip, n_error
        record_id = record["record_id"]
        video_path = record.get("resolved_video_path") or record.get("video_path")
        qtype = record.get("question_type", "cataract_irregularity_classification")
        correct_answer = record["correct_answer"]
        question_text = record["question_text"]

        # Skip already evaluated records
        if record_id in processed_ids:
            with write_lock:
                n_skip += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_error)
            return

        # Query the model
        model_response = generate_fn(
            video_path=video_path,
            question_text=question_text,
            log_id=record_id
        )

        with write_lock:
            if model_response is None:
                n_error += 1
                resp_record = {
                    "record_id": record_id,
                    "video_path": video_path,
                    "video": record.get("video", Path(video_path).name),
                    "task_category": record.get("task_category", "mcq"),
                    "question_type": qtype,
                    "category": record.get("category", ""),
                    "correct_answer": correct_answer,
                    "question_text": question_text,
                    "reference_reasoning": record.get("reference_reasoning", ""),
                    "metadata": record.get("metadata", {}),
                    "model_response": "ERROR: Model generation failed or timed out after all retries."
                }
                write_jsonl(resp_f, resp_record)
                score_record = {
                    "record_id": record_id,
                    "question_type": qtype,
                    "category": record.get("category", ""),
                    "correct_answer": correct_answer,
                    "score": 0,
                    "max_score": 1,
                    "normalised_score": 0.0,
                    "extracted_answer": "ERROR",
                    "predicted_category": "Error",
                    "correct": False,
                    "method": "failed_generation",
                    "format_valid": False
                }
                write_jsonl(score_f, score_record)
            else:
                # Record raw response
                resp_record = {
                    "record_id": record_id,
                    "video_path": video_path,
                    "video": record.get("video", Path(video_path).name),
                    "task_category": record.get("task_category", "mcq"),
                    "question_type": qtype,
                    "category": record.get("category", ""),
                    "correct_answer": correct_answer,
                    "question_text": question_text,
                    "reference_reasoning": record.get("reference_reasoning", ""),
                    "metadata": record.get("metadata", {}),
                    "model_response": model_response
                }
                write_jsonl(resp_f, resp_record)

                # Deterministic scoring
                score_info = score_irregularity_task(
                    model_response=model_response,
                    correct_answer=correct_answer,
                    question_type=qtype
                )

                score_record = {
                    "record_id": record_id,
                    "question_type": qtype,
                    "category": record.get("category", ""),
                    "correct_answer": correct_answer,
                    **score_info
                }
                write_jsonl(score_f, score_record)
                n_ok += 1

            resp_f.flush()
            score_f.flush()
            pbar.update(1)
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_error)

    try:
        pbar = tqdm(records, desc=f"Eval Irregularity [{tag}]", leave=True, dynamic_ncols=True)
        if concurrency <= 1:
            for record in records:
                process_single_record(record, pbar)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(process_single_record, record, pbar) for record in records]
                for f in as_completed(futures):
                    f.result()
    finally:
        resp_f.close()
        score_f.close()

    # Aggregate summary metrics
    summary = aggregate_irregularity_metrics(
        scores_path=scores_path,
        output_summary_path=summary_path,
        model_id=getattr(args, "model_id", tag),
        tag=tag
    )

    return summary
