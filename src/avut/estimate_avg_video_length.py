#!/usr/bin/env python3
"""Estimate average AVUT video length from a small sampled subset.

This script:
1) streams AVUT rows from Hugging Face,
2) scans a limited pool,
3) selects the N smallest videos by byte size,
4) computes durations with ffprobe, and
5) prints summary stats.

It is intentionally self-contained and does not depend on project modules.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


def parse_hf_uri(hf_uri: str) -> tuple[str, str]:
    if not hf_uri.startswith("hf://"):
        raise ValueError(f"Expected hf:// URI, got: {hf_uri}")
    repo_and_split = hf_uri.removeprefix("hf://")
    if "@" in repo_and_split:
        repo_id, split = repo_and_split.split("@", 1)
    else:
        repo_id, split = repo_and_split, "train"
    return repo_id, split


def ffprobe_duration_seconds(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")

    payload = json.loads(proc.stdout)
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError("ffprobe returned no duration")
    return float(duration)


def to_local_video_file(video_obj: object) -> tuple[str, str | None]:
    """Return (path_for_ffprobe, temp_file_to_cleanup_or_none)."""
    if not isinstance(video_obj, dict):
        raise ValueError("Unexpected video object format (expected dict).")

    path = video_obj.get("path")
    data = video_obj.get("bytes")

    if isinstance(path, str) and path and Path(path).exists():
        return path, None

    if isinstance(data, (bytes, bytearray)) and len(data) > 0:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return tmp.name, tmp.name

    # Some datasets return a path-like URL or unresolved location when bytes are absent.
    if isinstance(path, str) and path:
        return path, None

    raise ValueError("Video dict has neither readable local path nor bytes.")


def iter_dataset_rows(repo_id: str, split: str) -> Iterable[dict]:
    try:
        from datasets import Video, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ds = load_dataset(repo_id, split=split, streaming=True, token=token)
    # Keep decode disabled so we can measure raw bytes size directly.
    ds = ds.cast_column("video", Video(decode=False))
    for row in ds:
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate AVUT average video length from N smallest videos."
    )
    parser.add_argument(
        "--hf-uri",
        default="hf://tsinghua-ee/AVUTBenchmark@train",
        help="Hugging Face dataset URI, e.g. hf://org/name@train",
    )
    parser.add_argument(
        "--scan-count",
        type=int,
        default=60,
        help="How many rows to scan before choosing the smallest videos.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of smallest videos to use for average duration.",
    )
    args = parser.parse_args()

    if args.scan_count <= 0 or args.sample_size <= 0:
        raise SystemExit("--scan-count and --sample-size must be positive.")

    repo_id, split = parse_hf_uri(args.hf_uri)
    candidates: list[tuple[int, int, object]] = []  # (byte_size, row_idx, video_obj)

    print(f"Scanning {args.scan_count} rows from {repo_id}@{split} ...")
    for idx, row in enumerate(iter_dataset_rows(repo_id, split)):
        if idx >= args.scan_count:
            break
        video_obj = row.get("video")
        if not isinstance(video_obj, dict):
            continue
        raw = video_obj.get("bytes")
        if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
            candidates.append((len(raw), idx, video_obj))

    if not candidates:
        raise SystemExit(
            "No videos with embedded bytes found in scanned rows. "
            "Try increasing --scan-count or check dataset access/token."
        )

    candidates.sort(key=lambda x: x[0])
    chosen = candidates[: min(args.sample_size, len(candidates))]

    durations: list[float] = []
    print(f"Selected {len(chosen)} smallest videos. Measuring durations ...")
    for byte_size, row_idx, video_obj in chosen:
        temp_path: str | None = None
        try:
            video_path, temp_path = to_local_video_file(video_obj)
            duration_s = ffprobe_duration_seconds(video_path)
            durations.append(duration_s)
            print(
                f"- row={row_idx:>4} size={byte_size/1024/1024:>6.2f} MB "
                f"duration={duration_s:>6.2f}s"
            )
        except Exception as exc:  # noqa: BLE001 - keep script robust for quick analysis
            print(f"- row={row_idx:>4} size={byte_size/1024/1024:>6.2f} MB ERROR: {exc}")
        finally:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink(missing_ok=True)

    if not durations:
        raise SystemExit("Failed to measure durations for all sampled videos.")

    avg_s = sum(durations) / len(durations)
    print()
    print(f"Average duration across {len(durations)} sampled videos: {avg_s:.2f} seconds")
    print(f"Approximate average: {avg_s / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
