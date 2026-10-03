"""
downsample_videos.py
Fast, parallelized video downsampler for Cataract Surgery Irregularity dataset.
Downsamples 60fps/25fps high-bitrate surgical videos to a VLM-optimized temporal rate
and standardized vertical resolution (e.g. 2 fps or 1 fps at 360p).

Usage:
  python downsample_videos.py --fps 2.0 --scale 360 --output-dir data_downsampled
  python downsample_videos.py --fps 1.0 --scale 360 --output-dir data_downsampled_1fps
  python downsample_videos.py --in-place (replaces or creates beside)
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=0, desc="", unit="", **kwargs):
            self.total = total
            self.n = 0
            self.desc = desc
            print(f"[{desc}] Starting {total} tasks...")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            print(f"\n[{self.desc}] Completed {self.n}/{self.total}.")
        def update(self, n=1):
            self.n += n
            print(f"\rProgress: {self.n}/{self.total}...", end="", flush=True)
        def set_postfix(self, **kwargs):
            pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def downsample_single_video(
    src_video: Path,
    dst_video: Path,
    target_fps: float = 10.0,
    target_scale: int = 0,
    crf: int = 24,
    preset: str = "fast"
) -> dict:
    """Downsamples a single video file using ffmpeg."""
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    orig_size_mb = src_video.stat().st_size / (1024 * 1024)

    # Scale filter: if target_scale > 0, scale preserving aspect ratio; else keep native resolution
    if target_scale and target_scale > 0:
        vf_filter = f"fps={target_fps},scale=-2:{target_scale}"
    else:
        vf_filter = f"fps={target_fps}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_video),
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst_video)
    ]

    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t1 = time.time()

    if res.returncode != 0 or not dst_video.exists() or dst_video.stat().st_size == 0:
        return {
            "name": src_video.name,
            "success": False,
            "error": res.stderr[:200] if res.stderr else "Unknown error"
        }

    new_size_mb = dst_video.stat().st_size / (1024 * 1024)
    return {
        "name": src_video.name,
        "success": True,
        "orig_mb": round(orig_size_mb, 2),
        "new_mb": round(new_size_mb, 2),
        "ratio": round(orig_size_mb / max(0.01, new_size_mb), 2),
        "elapsed_sec": round(t1 - t0, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Downsample cataract surgery videos for VLM evaluation.")
    parser.add_argument("--src-dir", type=str, default=str(DATA_DIR), help="Source directory containing category subfolders.")
    parser.add_argument("--dst-dir", type=str, default=str(BASE_DIR / "data_downsampled"), help="Destination directory.")
    parser.add_argument("--fps", type=float, default=10.0, help="Target frame rate (default: 10.0 fps).")
    parser.add_argument("--scale", type=int, default=0, help="Target height resolution (default: 0 = keep native resolution; e.g. 360 for 360p).")
    parser.add_argument("--crf", type=int, default=24, help="H.264 CRF quality (default: 24).")
    parser.add_argument("--preset", type=str, default="fast", help="ffmpeg preset (default: fast).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent ffmpeg worker threads (default: 4).")
    parser.add_argument("--copy-jsonl", action="store_true", default=True, help="Copy paired .jsonl metadata files to destination.")
    args = parser.parse_args()

    src_root = Path(args.src_dir).resolve()
    dst_root = Path(args.dst_dir).resolve()

    print("=" * 75)
    print("CATARACT SURGERY VIDEO DOWNSAMPLER")
    print(f"Source Directory      : {src_root}")
    print(f"Destination Directory : {dst_root}")
    print(f"Target FPS            : {args.fps} fps")
    print(f"Target Height Scale   : {args.scale}p (aspect ratio preserved)")
    print(f"Parallel Workers      : {args.workers}")
    print("=" * 75)

    categories = ["Normal", "Lens_irregularity", "Pupil_Contraction"]
    tasks = []

    for cat in categories:
        cat_src = src_root / cat
        cat_dst = dst_root / cat
        if not cat_src.is_dir():
            continue

        for mp4_file in sorted(cat_src.glob("*.mp4")):
            dst_file = cat_dst / mp4_file.name
            tasks.append((mp4_file, dst_file, cat))

            # Copy paired jsonl files if requested
            if args.copy_jsonl:
                for jf in cat_src.glob(f"{mp4_file.stem}*.jsonl"):
                    cat_dst.mkdir(parents=True, exist_ok=True)
                    dst_jf = cat_dst / jf.name
                    if not dst_jf.exists() or dst_jf.stat().st_mtime < jf.stat().st_mtime:
                        import shutil
                        shutil.copy2(jf, dst_jf)

    print(f"\nFound {len(tasks)} videos to process across {len(categories)} categories.\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                downsample_single_video,
                src,
                dst,
                target_fps=args.fps,
                target_scale=args.scale,
                crf=args.crf,
                preset=args.preset
            ): (src, dst, cat)
            for src, dst, cat in tasks
        }

        with tqdm(total=len(tasks), desc="Downsampling", unit="video") as pbar:
            for future in as_completed(future_map):
                res = future.result()
                results.append(res)
                if res["success"]:
                    pbar.set_postfix(
                        last=res["name"],
                        orig=f"{res['orig_mb']}MB",
                        new=f"{res['new_mb']}MB",
                        ratio=f"{res['ratio']}x"
                    )
                else:
                    print(f"\n[FAIL] {res['name']}: {res.get('error')}")
                pbar.update(1)

    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    total_orig_mb = sum(r["orig_mb"] for r in succeeded)
    total_new_mb = sum(r["new_mb"] for r in succeeded)
    total_time = sum(r["elapsed_sec"] for r in succeeded)

    print("\n" + "=" * 75)
    print(f"DOWNSAMPLING COMPLETE: {len(succeeded)}/{len(tasks)} succeeded.")
    if failed:
        print(f"Failed count: {len(failed)}")
    print(f"Original Total Size   : {total_orig_mb:.1f} MB ({total_orig_mb/1024:.2f} GB)")
    print(f"Downsampled Total Size: {total_new_mb:.1f} MB ({total_new_mb/1024:.2f} GB)")
    print(f"Overall Compression   : {total_orig_mb/max(0.1, total_new_mb):.1f}x reduction")
    print(f"Total Video Transcode Time: {total_time:.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
