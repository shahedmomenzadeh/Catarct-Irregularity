"""
hulumed_inference.py
Inference and deterministic evaluation execution for HuluMed model series.
Evaluates irregularity tasks strictly one-by-one using native HuluMed video conversation structure.
"""

import os
import gc
import logging
import traceback
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

from eval_common import (
    vram_stats,
    flush_memory,
    first_device,
    probe_total_frames,
    run_evaluation_loop
)

log = logging.getLogger("hulumed_inference")


def run_hulumed_generation(
    model,
    processor,
    video_path: str,
    question_text: str,
    fps: float,
    max_frames: int,
    frame_size: int,
    max_new_tokens: int,
    temperature: float,
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
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": fps,
                            "max_frames": attempt_frames,
                            "size": frame_size
                        }
                    },
                    {
                        "type": "text",
                        "text": question_text
                    }
                ]
            }
        ]

        try:
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            inputs = {
                k: (v.cuda().to(torch.float16) if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v.cuda() if isinstance(v, torch.Tensor)
                    else v)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id
                )

            # Strip input prompt tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            model_response = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            del inputs, output_ids, generated_ids
            break

        except torch.cuda.OutOfMemoryError as e:
            log.error(f"{log_id} — HuluMed CUDA OOM (frames={attempt_frames}): {e} | {vram_stats()}")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            log.error(f"{log_id} — HuluMed generation error: {e}\n{traceback.format_exc()}")
            break

    gc.collect()
    torch.cuda.empty_cache()
    return model_response


def run(args, records: List[Dict[str, Any]]) -> dict:
    """Main runner for HuluMed inference called by main.py."""
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

    log.info(f"Loading HuluMed model '{args.model_id}'...")
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval()

    log.info(f"Model loaded: {vram_stats('init')}")

    def generate_fn(video_path: str, question_text: str, log_id: str) -> Optional[str]:
        return run_hulumed_generation(
            model=model,
            processor=processor,
            video_path=video_path,
            question_text=question_text,
            fps=getattr(args, "fps", 1.0),
            max_frames=args.max_frames,
            frame_size=args.frame_size if isinstance(args.frame_size, int) else 224,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
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
