"""Audio extraction helpers for AVUT samples."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from google.genai import types


def _hf_to_resolve_url(hf_path: str) -> str | None:
    prefix = "hf://datasets/"
    if not hf_path.startswith(prefix):
        return None
    rest = hf_path[len(prefix) :]
    if "@" not in rest:
        return None
    repo, tail = rest.split("@", 1)
    if "/" not in tail:
        return None
    rev, rel = tail.split("/", 1)
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{rel}"


def _video_path_from_input(video_input: object) -> str | None:
    """Best-effort extraction of an on-disk video path from media input."""
    hf_encoded = getattr(video_input, "_hf_encoded", None)
    if isinstance(hf_encoded, dict):
        p = hf_encoded.get("path")
        if isinstance(p, str):
            if p.startswith("hf://datasets/"):
                return _hf_to_resolve_url(p) or p
            return p
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        path_val = video_input.get("path")
        if isinstance(path_val, str):
            if path_val.startswith("hf://datasets/"):
                return _hf_to_resolve_url(path_val) or path_val
            return path_val
    path_attr = getattr(video_input, "path", None)
    if isinstance(path_attr, str):
        if path_attr.startswith("hf://datasets/"):
            return _hf_to_resolve_url(path_attr) or path_attr
        return path_attr
    return None


def extract_audio_part_from_video_input(
    *,
    video_input: object | None,
    sample_id: str,
    cache_dir: Path,
) -> types.Part | None:
    """Extract mono 16k WAV via ffmpeg and return a Gemini audio Part.

    Returns ``None`` if no usable video path exists or ffmpeg extraction fails.
    """
    wav_bytes = extract_audio_wav_bytes_from_video_input(
        video_input=video_input,
        sample_id=sample_id,
        cache_dir=cache_dir,
    )
    if not wav_bytes:
        return None
    return types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")


def extract_audio_wav_bytes_from_video_input(
    *,
    video_input: object | None,
    sample_id: str,
    cache_dir: Path,
) -> bytes | None:
    """Extract mono 16k WAV via ffmpeg and return raw wav bytes.

    Returns ``None`` if no usable video path exists or ffmpeg extraction fails.
    """
    if video_input is None:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    src_path: Path | None = None
    src_uri: str | None = None
    src = _video_path_from_input(video_input)
    if src:
        if "://" in src:
            src_uri = src
        else:
            p = Path(src)
            if p.exists():
                src_path = p

    # HF video feature may carry bytes with a non-local/invalid path.
    raw_bytes = None
    if src_path is None and src_uri is None:
        if isinstance(video_input, dict):
            raw_b = video_input.get("bytes")
            if isinstance(raw_b, (bytes, bytearray)) and raw_b:
                raw_bytes = raw_b
        if raw_bytes is None:
            hf_encoded = getattr(video_input, "_hf_encoded", None)
            if isinstance(hf_encoded, dict):
                hb = hf_encoded.get("bytes")
                if isinstance(hb, (bytes, bytearray)) and hb:
                    raw_bytes = hb
    if raw_bytes is not None:
        key_b = hashlib.sha1(f"{sample_id}:bytes".encode("utf-8")).hexdigest()[:12]
        tmp_mp4 = cache_dir / f"{sample_id}_{key_b}.mp4"
        if not tmp_mp4.exists():
            try:
                tmp_mp4.write_bytes(bytes(raw_bytes))
            except Exception:
                return None
        src_path = tmp_mp4

    if src_path is None and src_uri is None:
        return None

    key_src = src_uri if src_uri is not None else str(src_path.resolve())
    key = hashlib.sha1(f"{sample_id}:{key_src}".encode("utf-8")).hexdigest()[:12]
    out_wav = cache_dir / f"{sample_id}_{key}.wav"

    if not out_wav.exists():
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            src_uri if src_uri is not None else str(src_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(out_wav),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return None

    try:
        wav_bytes = out_wav.read_bytes()
    except Exception:
        return None
    if not wav_bytes:
        return None
    return wav_bytes

