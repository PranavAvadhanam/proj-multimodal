from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def _scan_code(root: Path) -> dict[str, list[str]]:
    py_files = [p for p in root.rglob("*.py") if ".venv" not in str(p)]
    frame_like: list[str] = []
    video_upload_like: list[str] = []
    ffmpeg_audio_drop: list[str] = []
    ffmpeg_audio_codec: list[str] = []

    for p in py_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = str(p.relative_to(root))
        if rel.startswith("test_scripts/"):
            continue
        low = text.lower()
        if "extract_frames" in low or "image.open(" in low:
            if "generate_content" in low or "model.generate(" in low:
                frame_like.append(rel)
        if "upload_file(" in low and ("video/mp4" in low or "video" in low):
            video_upload_like.append(rel)
        if "ffmpeg" in low and re.search(r"(^|\\s)-an(\\s|$)", low):
            ffmpeg_audio_drop.append(rel)
        if "ffmpeg" in low and ("-c:a" in low or "aac" in low):
            ffmpeg_audio_codec.append(rel)

    return {
        "frame_like_inference_files": sorted(frame_like),
        "video_upload_files": sorted(video_upload_like),
        "ffmpeg_with_an_files": sorted(ffmpeg_audio_drop),
        "ffmpeg_with_audio_codec_files": sorted(ffmpeg_audio_codec),
    }


def _ffprobe_audio(video_path: Path) -> dict[str, object]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-select_streams",
        "a",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": proc.stderr.strip() or proc.stdout.strip() or "ffprobe failed",
            "audio_stream_count": 0,
            "audio_streams": [],
        }
    try:
        parsed = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        parsed = {}
    streams = parsed.get("streams") or []
    return {
        "ok": True,
        "audio_stream_count": len(streams),
        "audio_streams": streams,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether inference likely sends images-only and whether a video retains audio streams."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file to inspect with ffprobe.",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repo root to scan for inference/ffmpeg patterns.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    code_scan = _scan_code(root)
    probe = _ffprobe_audio(video)

    print("=== Code Scan ===")
    print(json.dumps(code_scan, indent=2))
    print("\n=== ffprobe Audio Check ===")
    print(json.dumps({"video": str(video), **probe}, indent=2))

    print("\n=== Summary ===")
    if code_scan["frame_like_inference_files"]:
        print(
            f"[WARN] Found frame-like inference patterns in {len(code_scan['frame_like_inference_files'])} file(s)."
        )
    else:
        print("[OK] No obvious frame-only inference patterns found.")
    if probe.get("ok") and int(probe.get("audio_stream_count", 0)) > 0:
        print("[OK] ffprobe detected audio stream(s) in the inspected video.")
    else:
        print("[FAIL] ffprobe found no audio stream in the inspected video.")


if __name__ == "__main__":
    main()
