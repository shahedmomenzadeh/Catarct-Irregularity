"""
hf_loader.py - Hugging Face Hub Dataset Downloader and Resolver for CSI-Bench
Supports:
1. Downloading from Hugging Face repository (e.g. 'username/cataract_irregularity_eval')
2. Caching or saving into local Irregularity_dataset directory
3. Seamlessly resolving local and remote dataset paths
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("hf_loader")


def download_hf_dataset(
    repo_id: str,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """
    Downloads the cataract irregularity dataset from Hugging Face Hub.
    Expects repository to contain Irregularity_dataset/ or flat case_*.mp4 / case_*.jsonl files.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download datasets from Hugging Face. "
            "Please install it using: pip install huggingface_hub"
        )

    target_dir = Path(local_dir).resolve() if local_dir else Path("./Irregularity_dataset").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    auth_token = token or os.environ.get("HF_TOKEN")
    log.info(f"Downloading Hugging Face dataset '{repo_id}' into {target_dir}...")

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=auth_token,
        revision=revision,
    )

    return Path(downloaded_path)


def resolve_dataset_root(
    dataset_root: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_token: Optional[str] = None,
    download_dir: Optional[str] = None,
) -> Path:
    """
    Resolves the active dataset root, downloading from Hugging Face if hf_dataset is given,
    or checking local paths (Irregularity_dataset/, data/, etc.).
    """
    if hf_dataset:
        dl_target = download_dir or "./Irregularity_dataset"
        downloaded = download_hf_dataset(
            repo_id=hf_dataset,
            local_dir=dl_target,
            token=hf_token
        )
        return downloaded

    if dataset_root:
        p = Path(dataset_root).resolve()
        if (p / "Irregularity_dataset").is_dir():
            return p / "Irregularity_dataset"
        if (p / "data").is_dir():
            return p / "data"
        return p

    # Default fallback: check Irregularity_dataset or data in current working directory or script directory
    cwd = Path.cwd()
    if (cwd / "Irregularity_dataset").is_dir():
        return cwd / "Irregularity_dataset"
    if (cwd / "data").is_dir():
        return cwd / "data"

    script_dir = Path(__file__).parent.resolve()
    if (script_dir / "Irregularity_dataset").is_dir():
        return script_dir / "Irregularity_dataset"
    if (script_dir / "data").is_dir():
        return script_dir / "data"

    return script_dir
