"""
dataset_loader.py
Module for loading cataract irregularity benchmark records.
Supports loading from:
  1. Flat standardized directory: Irregularity_dataset/
  2. Categorized subfolders: data/{Normal, Lens_irregularity, Pupil_Contraction}
  3. Remote Hugging Face Hub dataset repository via hf_loader
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

log = logging.getLogger("dataset_loader")


def resolve_dataset_dir(dataset_root: str) -> Path:
    """
    Resolves the dataset directory from a given dataset root.
    Handles:
      - root / "Irregularity_dataset"
      - root / "data"
      - root directly containing .jsonl / .mp4 files
    """
    root_path = Path(dataset_root).resolve()

    # 1. Check if Irregularity_dataset subfolder exists
    if (root_path / "Irregularity_dataset").is_dir():
        return root_path / "Irregularity_dataset"

    # 2. Check if data subfolder exists
    if (root_path / "data").is_dir():
        return root_path / "data"

    # 3. Direct folder
    return root_path


def ensure_dataset(
    dataset_root: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_token: Optional[str] = None,
    download_dir: Optional[str] = None,
) -> Path:
    """
    Ensures that the evaluation dataset is available locally.
    If hf_dataset is provided, downloads it from Hugging Face Hub.
    """
    if hf_dataset:
        from hf_loader import download_hf_dataset
        return download_hf_dataset(
            repo_id=hf_dataset,
            local_dir=download_dir,
            token=hf_token
        )

    from hf_loader import resolve_dataset_root
    return resolve_dataset_root(
        dataset_root=dataset_root,
        hf_dataset=None,
        hf_token=hf_token,
        download_dir=download_dir
    )


def _extract_user_text(record: dict) -> str:
    """Extracts the prompt/question text from messages[] or prompt[]."""
    for key in ("prompt", "messages"):
        for m in record.get(key, []):
            if m.get("role") == "user":
                content = m.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            return block.get("text", "").strip()
                elif isinstance(content, str):
                    return content.strip()
    return record.get("question_text", "").strip()


def _extract_reference(record: dict) -> str:
    """Extracts ground truth reasoning or assistant reference."""
    for m in record.get("messages", []):
        if m.get("role") == "assistant":
            return str(m.get("content", "")).strip()
    return str(record.get("reference_reasoning", "")).strip()


def load_irregularity_records(
    dataset_root: str,
    validate_videos: bool = True
) -> List[Dict[str, Any]]:
    """
    Loads all cataract irregularity evaluation tasks from either Irregularity_dataset/
    or data/{Normal, Lens_irregularity, Pupil_Contraction}.
    Each record represents an isolated single-case 3-class classification question.
    """
    resolved_dir = resolve_dataset_dir(dataset_root)
    log.info(f"Loading irregularity records from {resolved_dir}")

    # Find all JSONL files
    jsonl_files = []
    if resolved_dir.name == "data" or (resolved_dir / "Normal").is_dir():
        # Scrape category subfolders
        search_dirs = [resolved_dir / c for c in ("Normal", "Lens_irregularity", "Pupil_Contraction") if (resolved_dir / c).is_dir()]
        if not search_dirs:
            search_dirs = [resolved_dir]
        for d in search_dirs:
            # Avoid duplicate aliases by preferring base case_XXXX.jsonl
            files = sorted(list(d.glob("case_*.jsonl")))
            seen_cases = set()
            for f in files:
                case_stem = f.stem.replace("_irregularity_classification", "")
                if case_stem not in seen_cases:
                    seen_cases.add(case_stem)
                    jsonl_files.append(f)
    else:
        # Flat directory (Irregularity_dataset)
        files = sorted(list(resolved_dir.glob("case_*.jsonl")))
        seen_cases = set()
        for f in files:
            case_stem = f.stem.replace("_irregularity_classification", "")
            if case_stem not in seen_cases:
                seen_cases.add(case_stem)
                jsonl_files.append(f)

    records = []
    for jf in jsonl_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)

                    # Resolve video path
                    video_fn = rec.get("video")
                    if not video_fn:
                        log.warning(f"No video field in record {rec.get('record_id')} ({jf})")
                        continue

                    # Search video in same folder or resolved_dir or data/
                    cand_video = jf.parent / video_fn
                    if not cand_video.exists():
                        cand_video = resolved_dir / video_fn
                    if not cand_video.exists():
                        # Try searching data subfolders
                        base_d = resolved_dir.parent if resolved_dir.name in ("Irregularity_dataset", "data") else resolved_dir
                        for sub in ("Normal", "Lens_irregularity", "Pupil_Contraction"):
                            p = base_d / "data" / sub / video_fn
                            if p.exists():
                                cand_video = p
                                break

                    if validate_videos and not cand_video.exists():
                        log.warning(f"Video file {cand_video} not found for record {rec.get('record_id')}. Skipping.")
                        continue

                    rec["resolved_video_path"] = str(cand_video.resolve())
                    rec["question_text"] = _extract_user_text(rec)
                    rec["reference_reasoning"] = _extract_reference(rec)
                    records.append(rec)
        except Exception as e:
            log.warning(f"Failed to read {jf}: {e}")

    log.info(f"Loaded {len(records)} cataract irregularity evaluation records.")
    return records
