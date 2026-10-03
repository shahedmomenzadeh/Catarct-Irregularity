"""
qwen3VL_inference.py
Inference and deterministic evaluation execution for Qwen3-VL model series.
Evaluates irregularity tasks strictly one-by-one with progressive frame OOM retry.
"""

import os
import gc
import logging
from typing import Optional, List, Dict, Any
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from eval_common import (
    vram_stats,
    flush_memory,
    first_device,
    move_inputs_to_device,
    probe_total_frames,
    run_evaluation_loop
)

log = logging.getLogger("qwen3vl_inference")


def build_inputs(
    processor,
    video_path: str,
    question_text: str,
    max_frames: int,
    max_pixels: int,
    min_pixels: int,
    primary_device: torch.device
) -> dict:
    """Constructs tokenized multimodal inputs for Qwen3-VL using process_vision_info."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": min(min_pixels, max_pixels),
                    "max_pixels": max_pixels,
                    "nframes": max_frames
                },
                {
                    "type": "text",
                    "text": question_text
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = move_inputs_to_device(inputs, primary_device)
    return inputs


def run_qwen3vl_generation(
    model,
    processor,
    video_path: str,
    question_text: str,
    max_frames: int,
    max_pixels: int,
    min_pixels: int,
    max_new_tokens: int,
    temperature: float,
    primary_device: torch.device,
    log_id: str
) -> Optional[str]:
    """Runs generation with progressive frame-count retry on CUDA OOM or short-video errors."""
    effective_max = max_frames
    probed_total = probe_total_frames(video_path)
    if probed_total and probed_total > 0 and max_frames > probed_total:
        effective_max = probed_total

    retry_frames = []
    f = effective_max
    while f >= 2:
        retry_frames.append(f)
        f = f // 2
    if not retry_frames:
        retry_frames = [2]

    model_response = None
    for attempt_frames in retry_frames:
        try:
            inputs = build_inputs(
                processor=processor,
                video_path=video_path,
                question_text=question_text,
                max_frames=attempt_frames,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
                primary_device=primary_device
            )
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0.0 else None,
                    do_sample=temperature > 0.0
                )

            trimmed = [out_ids[input_len:] for out_ids in output_ids]
            out_text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            flush_memory(inputs, output_ids, trimmed)
            model_response = out_text
            break

        except torch.cuda.OutOfMemoryError:
            log.warning(f"[{log_id}] CUDA OOM at {attempt_frames} frames. Flushing cache and retrying with fewer frames...")
            flush_memory()
            continue
        except Exception as e:
            err_str = str(e)
            if "out of memory" in err_str.lower():
                log.warning(f"[{log_id}] Generic OOM error at {attempt_frames} frames: {err_str[:100]}. Retrying...")
                flush_memory()
                continue
            log.error(f"[{log_id}] Generation exception: {e}")
            flush_memory()
            return None

    return model_response


def run(args, records: List[Dict[str, Any]]) -> dict:
    """Entry point for Qwen3-VL model family."""
    model_id = args.model_id
    log.info(f"Loading Qwen3-VL processor from: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )

    bnb_config = None
    if getattr(args, "load_in_4bit", False):
        log.info("Enabling 4-bit NF4 quantization via bitsandbytes.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif getattr(args, "load_in_8bit", False):
        log.info("Enabling 8-bit quantization via bitsandbytes.")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    log.info(f"Loading model weights from: {model_id} (attn: {args.attn_implementation})")
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "attn_implementation": args.attn_implementation
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        **model_kwargs
    )
    model.eval()

    primary_dev = first_device(model)
    log.info(f"Model loaded. Primary device: {primary_dev} | {vram_stats('init')}")

    def generate_fn(video_path: str, question_text: str, log_id: str) -> Optional[str]:
        return run_qwen3vl_generation(
            model=model,
            processor=processor,
            video_path=video_path,
            question_text=question_text,
            max_frames=args.max_frames,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            primary_device=primary_dev,
            log_id=log_id
        )

    summary = run_evaluation_loop(
        generate_fn=generate_fn,
        records=records,
        output_dir=args.output_dir,
        tag=args.tag,
        args=args,
        logger=log
    )

    flush_memory(model, processor)
    return summary
