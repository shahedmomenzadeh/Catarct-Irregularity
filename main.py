"""
main.py
Central command-line entry point to orchestrate Cataract Irregularity evaluations (CSI-Bench).
Dispatches model families (Qwen3-VL, HuluMed, Lingshu, Mage-VL, API endpoints).
All evaluations are strictly deterministic (3-class MCQ) and questions are queried one by one.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("irregularity_orchestrator")

import dataset_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Cataract Surgery Irregularity Understanding (CSI-Bench)")

    # Model architecture
    parser.add_argument("--model-family", type=str, required=False,
                        choices=["qwen3vl", "hulumed", "lingshu", "qwen2_5_vl", "mage_vl", "openai_api", "gemini", "api"],
                        help="Model family architecture type.")
    parser.add_argument("--model-id", type=str, required=False,
                        help="Hugging Face model identifier, local model path, or API model name (e.g. 'ag/gemini-3.8-flash').")

    # Dataset paths
    parser.add_argument("--dataset-root", type=str, default=str(Path(__file__).parent / "Irregularity_dataset"),
                        help="Root directory containing the evaluation benchmark (default: Irregularity_dataset).")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="Hugging Face dataset identifier. If provided, downloads and loads from HF Hub.")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face API token.")
    parser.add_argument("--download-dir", type=str, default=None,
                        help="Directory to store downloaded Hugging Face dataset.")

    # Output and tagging
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / "results"),
                        help="Directory to save output JSONL responses, scores, and summary reports.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag identifier for output filenames. If None, generated automatically.")

    # Sampling & Multimodal Parameters
    parser.add_argument("--max-frames", type=int, default=12,
                        help="Maximum frames to sample from each video.")
    parser.add_argument("--frame-size", type=str, default="224",
                        help="Frame resizing parameter for HuluMed (int or 'height,width').")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Video sampling frame rate.")
    parser.add_argument("--max-pixels", type=int, default=307200,
                        help="Qwen3-VL/Lingshu max pixels parameter (default: 640*480).")
    parser.add_argument("--min-pixels", type=int, default=100352,
                        help="Qwen3-VL/Lingshu min pixels parameter.")

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=400,
                        help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Model sampling temperature (0.0 for greedy).")

    # Attention and Memory
    parser.add_argument("--use-flash-attn", action="store_true", default=False,
                        help="Use FlashAttention-2 if supported.")
    parser.add_argument("--attn-implementation", type=str, default=None, choices=["sdpa", "flash_attention_2", "eager"],
                        help="Explicit attention implementation.")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Load model in 4-bit NF4 format using bitsandbytes (default: True).")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                        help="Disable 4-bit quantization.")
    parser.add_argument("--load-in-8bit", action="store_true", default=False,
                        help="Load model in 8-bit format using bitsandbytes.")
    parser.add_argument("--gpu-memory-budget", type=str, default=None,
                        help="Per-GPU memory budget constraint (e.g., '14GiB').")

    # API Settings (for OpenAI / Gemini / 9router endpoints)
    parser.add_argument("--api-base-url", type=str, default=None,
                        help="Base URL for OpenAI-compatible vision endpoint (e.g. http://localhost:20128/v1).")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for endpoint (default: 'none').")
    parser.add_argument("--api-timeout", type=float, default=600.0,
                        help="Request timeout in seconds for API calls (default: 600.0s / 10 minutes).")
    parser.add_argument("--api-retries", type=int, default=5,
                        help="Maximum retry attempts on network errors or timeouts (default: 5).")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent worker count for API requests (default: 1).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optionally limit evaluation to first N records.")

    # Dry run
    parser.add_argument("--dry-run", action="store_true",
                        help="Only loads and prints dataset records without loading models.")

    args = parser.parse_args()

    # Attention resolution
    if args.attn_implementation is None:
        args.attn_implementation = "flash_attention_2" if args.use_flash_attn else "sdpa"

    # Frame size parsing
    if args.frame_size:
        size_str = args.frame_size.strip()
        if "," in size_str:
            try:
                args.frame_size = [int(x.strip()) for x in size_str.split(",")]
            except ValueError:
                parser.error(f"Invalid format for --frame-size: {args.frame_size}")
        elif "x" in size_str:
            try:
                args.frame_size = [int(x.strip()) for x in size_str.split("x")]
            except ValueError:
                parser.error(f"Invalid format for --frame-size: {args.frame_size}")
        else:
            try:
                args.frame_size = int(size_str)
            except ValueError:
                parser.error(f"Invalid format for --frame-size: {args.frame_size}")

    if not args.dry_run:
        if not args.model_family:
            parser.error("Argument --model-family is required when not running --dry-run.")
        if not args.model_id:
            parser.error("Argument --model-id is required when not running --dry-run.")

    return args


def print_summary(summary: dict):
    """Formats and prints final summary tables based on evaluation run."""
    print("\n" + "=" * 78)
    print(f"  CATARACT IRREGULARITY EVALUATION SUMMARY (CSI-Bench)")
    print("=" * 78)

    print(f"  Model ID                 : {summary.get('model_id', 'Unknown')}")
    print(f"  Tag                      : {summary.get('tag', 'eval')}")
    print(f"  Total Cases Evaluated    : {summary.get('total_cases_evaluated', 0)}")
    print(f"  Overall Accuracy         : {summary.get('overall_accuracy', 0.0) * 100:.2f}%")
    print(f"  Macro-Balanced Accuracy  : {summary.get('balanced_accuracy', 0.0) * 100:.2f}%")
    print(f"  JSON Format Adherence    : {summary.get('json_format_adherence_rate', 0.0) * 100:.2f}%")

    print("\n  [PER-CLASS DIAGNOSTIC METRICS]")
    print(f"  {'Option':<7} {'Category':<20} {'Cases':<7} {'Sens/Rec':<10} {'Spec':<10} {'Prec':<10} {'F1':<8}")
    print("  " + "-" * 74)
    per_class = summary.get("per_class_metrics", {})
    for name, m in per_class.items():
        print(f"  Option {m['option']:<2} {name:<20} {m['n_samples']:<7} {m['sensitivity_recall']*100:>6.1f}%    {m['specificity']*100:>6.1f}%    {m['precision_ppv']*100:>6.1f}%    {m['f1_score']:>6.4f}")

    print("\n  [3x3 CONFUSION MATRIX]")
    print(f"  {'Ground Truth':<24} | {'Normal (A)':<12} | {'Lens Irreg (B)':<15} | {'Pupil Cont (C)':<15} | {'Invalid':<8}")
    print("  " + "-" * 82)
    matrix = summary.get("confusion_matrix", {})
    labels = [("A", "Normal (A)"), ("B", "Lens Irregularity (B)"), ("C", "Pupil Contraction (C)")]
    for opt, label in labels:
        row = matrix.get(opt, {})
        print(f"  {label:<24} | {row.get('A', 0):<12} | {row.get('B', 0):<15} | {row.get('C', 0):<15} | {row.get('INVALID', 0):<8}")

    print("=" * 78 + "\n")


def main():
    args = parse_args()

    # 1. Determine tag
    if args.tag is None and args.model_id is not None:
        model_name_clean = args.model_id.split("/")[-1].replace("-", "_").lower()
        args.tag = f"{args.model_family}_{model_name_clean}"
    elif args.tag is None:
        args.tag = "dry_run"

    # 2. Resolve Dataset Root (Local or Hugging Face)
    if args.hf_dataset:
        args.dataset_root = str(dataset_loader.ensure_dataset(
            hf_dataset=args.hf_dataset,
            hf_token=args.hf_token,
            download_dir=args.download_dir
        ))
    else:
        args.dataset_root = str(dataset_loader.ensure_dataset(
            dataset_root=args.dataset_root
        ))

    log.info(f"Model Family: {args.model_family}")
    log.info(f"Model ID: {args.model_id}")
    log.info(f"Tag Label: {args.tag}")
    log.info(f"Dataset Root: {args.dataset_root}")

    # 3. Load Dataset Records
    records = dataset_loader.load_irregularity_records(
        dataset_root=args.dataset_root,
        validate_videos=True
    )

    if args.limit is not None and args.limit > 0:
        records = records[:args.limit]
        log.info(f"Limiting evaluation to first {args.limit} records.")

    if args.dry_run:
        log.info("Dry-run mode active. Summary of loaded records:")
        log.info(f"Total irregularity records: {len(records)}")
        if records:
            log.info(f"Sample record ID: {records[0]['record_id']} | Category: {records[0]['category']} | Answer: {records[0]['correct_answer']}")
            log.info(f"Sample prompt:\n{records[0]['question_text']}")
        sys.exit(0)

    # 4. Dispatch to Model Inference Runner
    summary = {}
    if args.model_family == "qwen3vl":
        import qwen3VL_inference
        summary = qwen3VL_inference.run(args, records)
    elif args.model_family == "hulumed":
        import hulumed_inference
        summary = hulumed_inference.run(args, records)
    elif args.model_family in ("lingshu", "qwen2_5_vl"):
        import lingshu_inference
        summary = lingshu_inference.run(args, records)
    elif args.model_family == "mage_vl":
        import mage_vl_inference
        summary = mage_vl_inference.run(args, records)
    elif args.model_family in ("openai_api", "gemini", "api"):
        import api_inference
        summary = api_inference.run(args, records)
    else:
        log.error(f"Unsupported model family: {args.model_family}")
        sys.exit(1)

    # 5. Print Summary
    if summary:
        print_summary(summary)


if __name__ == "__main__":
    main()
