#!/usr/bin/env python3
"""
upload_to_hf.py - Upload Cataract Irregularity Evaluation Dataset (CSI-Bench) to Hugging Face Hub.

Dataset Repo: https://huggingface.co/datasets/shahedm2001/Catarct-Irregularity
Folder: Irregularity_dataset (41 mp4 videos, 82 jsonl files, README.md)
"""

import os
import sys
import pathlib
import argparse
from pathlib import Path

# Load .env if present
env_path = pathlib.Path(".env")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k not in os.environ or not os.environ[k]:
            os.environ[k] = v


def verify_remote(api, repo_id: str, expected_mp4_count: int, expected_jsonl_count: int):
    print(f"\n--- Verifying Remote Repository ({repo_id}) ---")
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"Remote total files: {len(files)}")

        remote_mp4 = [f for f in files if f.endswith(".mp4")]
        remote_jsonl = [f for f in files if f.endswith(".jsonl")]
        has_readme = "README.md" in files

        print(f"Remote Videos: {len(remote_mp4)} (expected: {expected_mp4_count})")
        print(f"Remote JSONL:  {len(remote_jsonl)} (expected: {expected_jsonl_count})")
        print(f"Remote README: {'Present' if has_readme else 'MISSING'}")

        if len(remote_mp4) == expected_mp4_count and len(remote_jsonl) == expected_jsonl_count and has_readme:
            print("\n[SUCCESS] All files uploaded and verified successfully!")
            print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
        else:
            print("\n[WARNING] Discrepancy detected between expected and remote file counts.")
    except Exception as e:
        print(f"Remote verification failed: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Upload Cataract Irregularity Evaluation Dataset (CSI-Bench) to Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Irregularity_dataset",
        help="Local directory containing dataset files (default: Irregularity_dataset)",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="Catarct-Irregularity",
        help="Repository name on Hugging Face Hub (default: Catarct-Irregularity)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Full repository ID (e.g. shahedm2001/Catarct-Irregularity). Overrides --repo-name.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (overrides HF_TOKEN from .env / env vars)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Mark repository as private (default is public)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify remote repository files without uploading",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found. Export it or place it in .env file.", file=sys.stderr)
        sys.exit(1)
    print(f"HF_TOKEN found: len={len(token)}, prefix={token[:8]}...")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Install via: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    # 1. Authenticate & Resolve Repo ID
    try:
        who = api.whoami()
        username = who.get("name") or who.get("fullname") or who["name"]
        print(f"Authenticated as: {username}")
    except Exception as e:
        print(f"ERROR: whoami failed: {e}", file=sys.stderr)
        sys.exit(1)

    repo_id = args.repo_id or f"{username}/{args.repo_name}"
    print(f"Target repository: {repo_id}")
    print(f"Repository URL:   https://huggingface.co/datasets/{repo_id}")

    # 2. Verify Local Dataset Directory
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    mp4_files = sorted(list(dataset_dir.glob("*.mp4")))
    jsonl_files = sorted(list(dataset_dir.glob("case_*.jsonl")))
    readme_file = dataset_dir / "README.md"

    total_mp4_bytes = sum(f.stat().st_size for f in mp4_files)
    total_mp4_mb = total_mp4_bytes / (1024 * 1024)

    print("\n--- Local Dataset Verification ---")
    print(f"Dataset path:      {dataset_dir}")
    print(f"Videos found:      {len(mp4_files)} mp4 files ({total_mp4_mb:.1f} MB)")
    print(f"JSONL tasks found: {len(jsonl_files)} files")
    print(f"Dataset card:      {'Found' if readme_file.exists() else 'MISSING'}")

    if not mp4_files or not jsonl_files or not readme_file.exists():
        print("ERROR: Local dataset verification failed. Missing files.", file=sys.stderr)
        sys.exit(1)

    # If --verify-only is requested, skip uploading
    if args.verify_only:
        verify_remote(api, repo_id, len(mp4_files), len(jsonl_files))
        return

    # 3. Create / Ensure Repo Exists
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"Repository ready: {repo_id} (private={args.private})")
    except Exception as e:
        print(f"create_repo note: {e}")

    # 4. Upload Stage 1: Dataset Card (README.md) & Annotations (.jsonl)
    print(f"\n[Stage 1/2] Uploading Dataset Card and JSONL tasks ({len(jsonl_files)} jsonl + README.md)...")
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["README.md", "*.jsonl"],
        commit_message="Add dataset card (README.md) and JSONL evaluation tasks",
        ignore_patterns=[".git/*", "*.mp4"],
    )
    print("Stage 1 complete: Metadata & annotations uploaded.")

    # 5. Upload Stage 2: Videos (*.mp4 via Git LFS)
    print(f"\n[Stage 2/2] Uploading video procedures ({len(mp4_files)} mp4, {total_mp4_mb:.1f} MB)...")
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.mp4"],
        commit_message=f"Add surgical videos ({len(mp4_files)} mp4, {total_mp4_mb:.1f} MB @ 10fps)",
        ignore_patterns=[".git/*", "*.jsonl", "*.md"],
    )
    print("Stage 2 complete: Videos uploaded.")

    # 6. Verify Remote Repository
    verify_remote(api, repo_id, len(mp4_files), len(jsonl_files))


if __name__ == "__main__":
    main()
