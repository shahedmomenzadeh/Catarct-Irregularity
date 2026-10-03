"""
mage_vl_inference.py
Inference and deterministic evaluation execution for microsoft/Mage-VL.
Evaluates irregularity tasks strictly one-by-one.
"""

import os
import gc
import logging
import traceback
from typing import Optional, List, Dict, Any
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from eval_common import (
    vram_stats,
    flush_memory,
    first_device,
    run_evaluation_loop
)

log = logging.getLogger("mage_vl_inference")


def sample_video_ffmpeg(video_path: str, num_frames: int) -> List[Image.Image]:
    """Fallback frame extractor using ffmpeg image pipe."""
    import subprocess
    import io
    import json

    dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path]
    try:
        res = subprocess.run(dur_cmd, capture_output=True, text=True, check=True)
        dur = float(json.loads(res.stdout)["format"]["duration"])
    except Exception:
        dur = 60.0

    frames = []
    for i in range(num_frames):
        ts = (i + 0.5) * (dur / max(1, num_frames))
        cmd = [
            "ffmpeg", "-v", "error", "-ss", f"{ts:.2f}",
            "-i", video_path, "-vframes", "1",
            "-f", "image2pipe", "-vcodec", "png", "-"
        ]
        fres = subprocess.run(cmd, capture_output=True)
        if fres.stdout:
            img = Image.open(io.BytesIO(fres.stdout)).convert("RGB")
            frames.append(img)
    return frames


def sample_video(video_path: str, num_frames: int) -> List[Image.Image]:
    """Uniformly samples up to `num_frames` RGB PIL frames from video with automatic ffmpeg fallback."""
    try:
        import cv2
        import numpy as np

        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            capture.release()
            log.warning(f"OpenCV could not determine frame count for {video_path}. Using ffmpeg fallback.")
            return sample_video_ffmpeg(video_path, num_frames)

        indices = np.linspace(0, frame_count - 1, min(num_frames, frame_count), dtype=int)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                capture.release()
                log.warning(f"OpenCV failed to decode frame {index} from {video_path}. Using ffmpeg fallback.")
                return sample_video_ffmpeg(video_path, num_frames)
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        capture.release()
        return frames
    except Exception as e:
        log.warning(f"OpenCV sampling failed: {e}. Falling back to ffmpeg frame extraction.")
        return sample_video_ffmpeg(video_path, num_frames)


def build_inputs(
    processor,
    model_device: torch.device,
    model_dtype: torch.dtype,
    video_path: str,
    question_text: str,
    num_frames: int
) -> dict:
    """Builds tokenized inputs for Mage-VL."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question_text}
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    frames = sample_video(video_path, num_frames)
    inputs = processor(text=prompt, videos=frames, return_tensors="pt")

    return {
        k: (v.to(model_device, dtype=model_dtype)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v.to(model_device) if isinstance(v, torch.Tensor)
            else v)
        for k, v in inputs.items()
    }


def run_mage_generation(
    model,
    processor,
    video_path: str,
    question_text: str,
    num_frames: int,
    max_new_tokens: int,
    temperature: float,
    primary_device: torch.device,
    log_id: str = ""
) -> Optional[str]:
    """Runs Mage-VL generation with fallback frames retry on OOM."""
    retry_frames = []
    f = num_frames
    while f >= 2:
        retry_frames.append(f)
        f = f // 2
    if not retry_frames:
        retry_frames = [2]

    model_dtype = next(model.parameters()).dtype
    model_response = None

    for attempt_frames in retry_frames:
        try:
            inputs = build_inputs(
                processor=processor,
                model_device=primary_device,
                model_dtype=model_dtype,
                video_path=video_path,
                question_text=question_text,
                num_frames=attempt_frames
            )

            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    pad_token_id=processor.tokenizer.eos_token_id
                )

            generated_ids = output_ids[:, input_len:]
            model_response = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            del inputs, output_ids, generated_ids
            break

        except torch.cuda.OutOfMemoryError as e:
            log.error(f"{log_id} — Mage-VL CUDA OOM (frames={attempt_frames}): {e} | {vram_stats()}")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            log.error(f"{log_id} — Mage-VL generation error: {e}\n{traceback.format_exc()}")
            break

    gc.collect()
    torch.cuda.empty_cache()
    return model_response


def run(args, records: List[Dict[str, Any]]) -> dict:
    """Main runner for Mage-VL inference called by main.py."""
    quant_config = None
    if getattr(args, "load_in_4bit", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif getattr(args, "load_in_8bit", False):
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    n_gpus = torch.cuda.device_count()
    max_memory = None
    if getattr(args, "gpu_memory_budget", None) and n_gpus > 0:
        max_memory = {i: args.gpu_memory_budget for i in range(n_gpus)}

    log.info(f"Loading Mage-VL processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    log.info(f"Loading Mage-VL model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        trust_remote_code=True
    )
    model.eval()

    primary_dev = first_device(model)

    def generate_fn(video_path: str, question_text: str, log_id: str) -> Optional[str]:
        return run_mage_generation(
            model=model,
            processor=processor,
            video_path=video_path,
            question_text=question_text,
            num_frames=args.max_frames,
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
