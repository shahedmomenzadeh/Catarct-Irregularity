"""
api_inference.py
Inference and deterministic evaluation execution for API-based Vision-Language Models.
Supports OpenAI-compatible endpoints (such as local 9router serving ag/gemini-3.8-flash at http://localhost:20128/v1).
Enforces pure visual evaluation: 100% silent video (audio stripped via ffmpeg -an), zero transcripts,
and evaluates questions strictly one-by-one.
"""

import os
import json
import base64
import subprocess
import tempfile
import shutil
import time
import logging
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path
import httpx
import openai

from eval_common import run_evaluation_loop

log = logging.getLogger("api_inference")

_B64_CACHE = {}
_B64_LOCK = threading.Lock()


def is_h264(video_path: Path) -> bool:
    """Checks if the video stream is already standard H.264."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "json",
        str(video_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout or "{}")
        cname = data.get("streams", [{}])[0].get("codec_name", "")
        return cname.lower() in ("h264", "avc1")
    except Exception:
        return False


def get_base64_silent_video(video_path: str, max_size_mb: float = 12.0) -> str:
    """
    Prepares a pure silent video by stripping audio tracks with ffmpeg -an.
    Ensures standard H.264 encoding required by multimodal vision APIs.
    - If already H.264 and <= max_size_mb: uses fast stream copy (-c:v copy -an).
    - If non-H.264 or > max_size_mb: transcodes to 2fps 360p H.264 with ultrafast preset.
    Caches results in memory to eliminate redundant encoding across runs.
    """
    vpath_str = str(Path(video_path).resolve())
    with _B64_LOCK:
        if vpath_str in _B64_CACHE:
            return _B64_CACHE[vpath_str]

    vpath = Path(video_path)
    sz_mb = vpath.stat().st_size / (1024 * 1024)
    temp_dir = tempfile.mkdtemp()
    out_silent_mp4 = os.path.join(temp_dir, "silent_video.mp4")

    dur = 0.0
    try:
        dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(vpath)]
        r = subprocess.run(dur_cmd, capture_output=True, text=True)
        dur = float(json.loads(r.stdout or "{}").get("format", {}).get("duration", 0))
    except Exception:
        pass

    target_fps = "1"
    if dur > 360:
        target_fps = "0.5"

    cmd = [
        "ffmpeg", "-y", "-i", str(vpath),
        "-vf", f"fps={target_fps},scale=-2:360",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-pix_fmt", "yuv420p",
        "-an",
        out_silent_mp4
    ]

    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    if not os.path.exists(out_silent_mp4) or os.path.getsize(out_silent_mp4) == 0:
        log.warning(f"Initial silent video transcode failed ({res.stderr[:100] if res.stderr else 'empty'}). Retrying with fallback transcode...")
        cmd_fallback = [
            "ffmpeg", "-y", "-i", str(vpath),
            "-vf", "fps=2,scale=-2:360",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-an",
            out_silent_mp4
        ]
        res_fb = subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if not os.path.exists(out_silent_mp4) or os.path.getsize(out_silent_mp4) == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"ffmpeg failed to process {video_path}: {res_fb.stderr[:200] if res_fb.stderr else 'unknown error'}")

    with open(out_silent_mp4, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("utf-8")

    shutil.rmtree(temp_dir, ignore_errors=True)
    with _B64_LOCK:
        _B64_CACHE[vpath_str] = b64_str
    return b64_str


def run_api_generation(
    client: openai.OpenAI,
    model_name: str,
    video_path: str,
    question_text: str,
    max_new_tokens: int = 400,
    temperature: float = 0.1,
    max_retries: int = 5,
    timeout_sec: float = 600.0,
    log_id: str = ""
) -> Optional[str]:
    """Sends a single question and silent video to an OpenAI-compatible vision endpoint with robust network retry."""
    try:
        b64_video = get_base64_silent_video(video_path)
    except Exception as e:
        log.error(f"[{log_id}] Video preparation failed: {e}")
        return None

    # Construct messages with data URI
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:video/mp4;base64,{b64_video}"
                    }
                },
                {
                    "type": "text",
                    "text": question_text
                }
            ]
        }
    ]

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                timeout=timeout_sec
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
            log.warning(f"[{log_id}] Empty response on attempt {attempt}/{max_retries}")
        except Exception as e:
            err_msg = str(e)
            log.warning(f"[{log_id}] API call attempt {attempt}/{max_retries} failed: {err_msg[:140]}")
            if attempt < max_retries:
                # Progressive backoff for high latency / poor network recovery
                wait_time = 5 * attempt
                log.info(f"[{log_id}] Poor connection backoff: waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                log.error(f"[{log_id}] All {max_retries} attempts failed.")
                return None

    return None


def run(args, records: List[Dict[str, Any]]) -> dict:
    """Entry point for API-based model evaluation."""
    base_url = args.api_base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "none")
    model_name = args.model_id or "ag/gemini-3.8-flash"
    timeout_sec = getattr(args, "api_timeout", 600.0)
    retries_count = getattr(args, "api_retries", 5)

    log.info(
        f"Connecting to API endpoint: {base_url} (Model: {model_name}, "
        f"Timeout: {timeout_sec:.1f}s, Max Retries: {retries_count})"
    )

    # Configure httpx client with generous timeouts to tolerate high latency & slow uploads
    http_client = httpx.Client(
        timeout=httpx.Timeout(
            timeout=timeout_sec,
            connect=120.0,
            read=timeout_sec,
            write=timeout_sec,
            pool=60.0
        ),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
    )
    client = openai.OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    def generate_fn(video_path: str, question_text: str, log_id: str) -> Optional[str]:
        return run_api_generation(
            client=client,
            model_name=model_name,
            video_path=video_path,
            question_text=question_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_retries=retries_count,
            timeout_sec=timeout_sec,
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

    return summary
