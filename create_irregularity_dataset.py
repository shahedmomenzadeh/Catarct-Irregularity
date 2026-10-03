"""
create_irregularity_dataset.py
Automated script to inspect, probe, and construct VLM-ready JSONL task envelopes
for all cataract surgery videos in E:\\Catarct-Irregularity\\data.
Sources: Cataract-1K dataset.

Outputs:
  1. Per-video single-line .jsonl files in E:\\Catarct-Irregularity\\data\\<Category>\\
  2. Standardized flat Irregularity_dataset\\ (using NTFS hardlinks for 0 extra disk usage)
  3. Metadata: metadata\\case_manifest.csv and metadata\\dataset_summary.json
  4. Hugging Face Dataset Card: Irregularity_dataset\\README.md
"""

import os
import sys
import json
import csv
import shutil
import subprocess
from pathlib import Path

from prompts import (
    IRREGULARITY_PROMPT_TEMPLATE,
    FOLDER_TO_CLASS,
    TAXONOMY_3CLASS,
    TAXONOMY_DESCRIPTIONS
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "Irregularity_dataset"
METADATA_DIR = BASE_DIR / "metadata"


def probe_video(video_path: Path) -> dict:
    """Probes a video file using ffprobe to obtain duration, resolution, fps, codec, and size."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,bit_rate:stream=codec_name,codec_type,width,height,r_frame_rate,nb_frames",
        "-of", "json",
        str(video_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        meta = json.loads(res.stdout or "{}")
    except Exception as e:
        print(f"Warning: ffprobe failed on {video_path.name}: {e}")
        return {
            "duration_sec": 0.0,
            "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "codec": "unknown",
            "has_audio": False
        }

    fmt = meta.get("format", {})
    streams = meta.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    dur = float(fmt.get("duration", 0))
    sz_mb = float(fmt.get("size", 0)) / (1024 * 1024)
    w = int(v_stream.get("width", 0))
    h = int(v_stream.get("height", 0))
    fps_raw = v_stream.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, fps_raw.split("/"))
        fps = num / den if den != 0 else 0.0
    except Exception:
        fps = 0.0

    return {
        "duration_sec": round(dur, 2),
        "size_mb": round(sz_mb, 2),
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "codec": v_stream.get("codec_name", "unknown"),
        "has_audio": a_stream is not None
    }


def link_or_copy(src_file: Path, dst_file: Path):
    """Hardlinks src_file to dst_file on NTFS (0 extra disk usage); falls back to copy."""
    if dst_file.exists():
        return
    try:
        os.link(src_file, dst_file)
    except Exception:
        shutil.copy2(src_file, dst_file)


def build_jsonl_record(video_file: Path, folder_name: str, video_meta: dict) -> dict:
    """Builds a standardized single-line JSONL evaluation envelope for a video."""
    class_info = FOLDER_TO_CLASS[folder_name]
    correct_letter = class_info["letter"]
    category_name = class_info["category"]
    reasoning = class_info["reference_reasoning"]
    case_id = video_file.stem
    record_id = f"{case_id}_irregularity_classification"

    prompt_text = IRREGULARITY_PROMPT_TEMPLATE.strip()

    record = {
        "record_id": record_id,
        "task_category": "mcq",
        "question_type": "cataract_irregularity_classification",
        "reward_type": "deterministic",
        "split": "Evaluation",
        "track": "cataract-1k",
        "video": video_file.name,
        "category": category_name,
        "prompt": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_file.name
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
        "correct_answer": correct_letter,
        "reference_reasoning": reasoning,
        "metadata": {
            "case_id": case_id,
            "category": category_name,
            "source_folder": folder_name,
            "source_dataset": "cataract-1k",
            "duration_sec": video_meta["duration_sec"],
            "width": video_meta["width"],
            "height": video_meta["height"],
            "fps": video_meta["fps"],
            "size_mb": video_meta["size_mb"],
            "codec": video_meta["codec"],
            "has_audio": video_meta["has_audio"],
            "split": "Evaluation"
        }
    }
    return record


def generate_hf_readme(manifest: list) -> str:
    """Generates a Hugging Face Dataset Card README.md for Irregularity_dataset."""
    cat_counts = {}
    for r in manifest:
        c = r["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    total_dur = sum(r["duration_sec"] for r in manifest) / 60
    total_sz = sum(r["size_mb"] for r in manifest)

    readme = f"""---
license: cc-by-nc-4.0
language:
  - en
tags:
  - medical
  - surgery
  - ophthalmology
  - cataract
  - cataract-1k
  - irregularity-classification
  - video-understanding
  - vision-language-models
task_categories:
  - visual-question-answering
  - video-text-to-text
size_categories:
  - n<1K
---

# Cataract Surgery Irregularity Benchmark (CSI-Bench)

CSI-Bench is a standardized, self-contained evaluation benchmark for Vision-Language Models (VLMs) on intraoperative ophthalmic irregularity detection and 3-class taxonomy classification.
Derived from the clinical **Cataract-1K** dataset.

## Dataset Overview

- **Total Videos:** {len(manifest)} procedures ({total_dur:.1f} minutes of surgical video, {total_sz:.1f} MB)
- **Modality:** Pure visual surgical video (all audio tracks completely stripped, silent H.264)
- **Task:** 3-Class Intraoperative Irregularity Classification with Clinical Explanation
- **Evaluation Paradigm:** 100% Deterministic closed-set MCQ scoring (no LLM judge needed)

### Class Distribution
| Option | Category | Cases | Percentage | Description |
|:------:|:---------|:-----:|:----------:|:------------|
| **A** | Normal | {cat_counts.get('Normal', 0)} | {cat_counts.get('Normal', 0)/len(manifest)*100:.1f}% | Standard routine cataract extraction without capsular or pupil irregularity |
| **B** | Lens Irregularity | {cat_counts.get('Lens Irregularity', 0)} | {cat_counts.get('Lens Irregularity', 0)/len(manifest)*100:.1f}% | Capsular tear, zonular laxity/dialysis, subluxation, abnormal lens morphology |
| **C** | Pupil Contraction | {cat_counts.get('Pupil Contraction', 0)} | {cat_counts.get('Pupil Contraction', 0)/len(manifest)*100:.1f}% | Intraoperative miosis, floppy iris syndrome (IFIS), pupil margin irregularities |

## JSONL Task Envelope Schema

Each `.jsonl` file corresponds to exactly one `.mp4` video and contains a single-line task envelope:

```json
{{
  "record_id": "case_XXXX_irregularity_classification",
  "task_category": "mcq",
  "question_type": "cataract_irregularity_classification",
  "reward_type": "deterministic",
  "split": "Evaluation",
  "track": "cataract-1k",
  "video": "case_XXXX.mp4",
  "category": "Normal | Lens Irregularity | Pupil Contraction",
  "prompt": [
    {{
      "role": "user",
      "content": [
        {{"type": "video", "video": "case_XXXX.mp4"}},
        {{"type": "text", "text": "You are given a cataract procedure video. Analyze the surgical video carefully, explain the visible intraoperative events, anatomical structures, and any observed irregularities, and then classify the case into one of the following 3 options: A) Normal B) Lens Irregularity C) Pupil Contraction. Respond ONLY with JSON with explanation (3-10 sentences) and answer (A, B, or C)."}}
      ]
    }}
  ],
  "correct_answer": "A | B | C",
  "reference_reasoning": "...",
  "metadata": {{
    "case_id": "case_XXXX",
    "category": "...",
    "source_dataset": "cataract-1k",
    "duration_sec": 233.4,
    "width": 512,
    "height": 384,
    "fps": 30.0
  }}
}}
```
"""
    return readme


def main():
    print("=" * 70)
    print("BUILDING CATARACT IRREGULARITY EVALUATION DATASET")
    print(f"Base Directory: {BASE_DIR}")
    print("=" * 70)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_processed = 0

    for folder_name in ("Normal", "Lens_irregularity", "Pupil_Contraction"):
        folder_path = DATA_DIR / folder_name
        if not folder_path.is_dir():
            print(f"Warning: Expected folder {folder_path} not found.")
            continue

        video_files = sorted(list(folder_path.glob("*.mp4")))
        print(f"\nProcessing category: {folder_name} ({len(video_files)} videos)...")

        for vf in video_files:
            meta = probe_video(vf)
            record = build_jsonl_record(vf, folder_name, meta)

            # 1. Save .jsonl beside .mp4 in data/<Category>/
            jsonl_path = folder_path / f"{vf.stem}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Also create alias with _irregularity_classification for flexible loaders
            alias_path = folder_path / f"{vf.stem}_irregularity_classification.jsonl"
            with open(alias_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 2. Hardlink to standardized flat Irregularity_dataset/
            dst_mp4 = DATASET_DIR / vf.name
            dst_jsonl = DATASET_DIR / f"{vf.stem}.jsonl"
            dst_alias = DATASET_DIR / f"{vf.stem}_irregularity_classification.jsonl"
            link_or_copy(vf, dst_mp4)
            link_or_copy(jsonl_path, dst_jsonl)
            link_or_copy(alias_path, dst_alias)

            manifest.append({
                "case_id": vf.stem,
                "category": record["category"],
                "source_folder": folder_name,
                "correct_answer": record["correct_answer"],
                "filename": vf.name,
                "duration_sec": meta["duration_sec"],
                "size_mb": meta["size_mb"],
                "width": meta["width"],
                "height": meta["height"],
                "fps": meta["fps"],
                "codec": meta["codec"]
            })
            total_processed += 1
            print(f"  [✓] {vf.name} -> Answer: {record['correct_answer']} ({record['category']}) | {meta['duration_sec']}s, {meta['size_mb']}MB")

    # 3. Write metadata CSV & JSON
    csv_path = METADATA_DIR / "case_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "category", "source_folder", "correct_answer",
            "filename", "duration_sec", "size_mb", "width", "height", "fps", "codec"
        ])
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "total_cases": len(manifest),
        "total_duration_minutes": round(sum(r["duration_sec"] for r in manifest) / 60, 2),
        "total_size_mb": round(sum(r["size_mb"] for r in manifest), 2),
        "categories": {
            "Normal": len([r for r in manifest if r["category"] == "Normal"]),
            "Lens Irregularity": len([r for r in manifest if r["category"] == "Lens Irregularity"]),
            "Pupil Contraction": len([r for r in manifest if r["category"] == "Pupil Contraction"])
        },
        "taxonomy": TAXONOMY_3CLASS,
        "taxonomy_descriptions": TAXONOMY_DESCRIPTIONS
    }

    summary_path = METADATA_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 4. Write Hugging Face Dataset Card
    readme_path = DATASET_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(generate_hf_readme(manifest))

    print("\n" + "=" * 70)
    print(f"DATASET GENERATION COMPLETE: {total_processed} cases processed.")
    print(f"Manifest written to: {csv_path}")
    print(f"Summary written to:  {summary_path}")
    print(f"Standardized flat benchmark in: {DATASET_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
