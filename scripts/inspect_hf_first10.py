from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv()


def _parse_hf_uri(hf_uri: str) -> tuple[str, str]:
    if not hf_uri.startswith("hf://"):
        raise ValueError(f"Expected hf:// URI, got: {hf_uri}")
    repo_and_split = hf_uri.removeprefix("hf://")
    if "@" in repo_and_split:
        repo_id, split = repo_and_split.split("@", 1)
    else:
        repo_id, split = repo_and_split, "train"
    return repo_id, split


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _summarize_video_field(video_obj: Any) -> Any:
    if isinstance(video_obj, dict):
        b = video_obj.get("bytes")
        return {
            "field_type": "dict",
            "keys": sorted(video_obj.keys()),
            "path": video_obj.get("path"),
            "has_bytes": isinstance(b, (bytes, bytearray)),
            "bytes_len": len(b) if isinstance(b, (bytes, bytearray)) else 0,
        }
    return {"field_type": type(video_obj).__name__, "repr": repr(video_obj)}


def _read_first_rows(hf_uri: str, rows: int) -> dict[str, Any]:
    try:
        from datasets import Video, load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency 'datasets'. Install with: pip install datasets") from exc

    repo_id, split = _parse_hf_uri(hf_uri)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    ds = load_dataset(repo_id, split=split, streaming=True, token=token)
    if "video" in getattr(ds, "features", {}):
        ds = ds.cast_column("video", Video(decode=False))

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if idx >= rows:
            break
        normalized: dict[str, Any] = {"_row_index": idx}
        for key, value in row.items():
            if key == "video":
                normalized[key] = _summarize_video_field(value)
            else:
                normalized[key] = _json_safe(value)
        out_rows.append(normalized)

    return {
        "hf_uri": hf_uri,
        "repo_id": repo_id,
        "split": split,
        "rows_requested": rows,
        "rows_captured": len(out_rows),
        "rows": out_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect first N Hugging Face AVUT rows and write metadata snapshot."
    )
    parser.add_argument(
        "--hf-uri",
        default="hf://tsinghua-ee/AVUTBenchmark@train",
        help="Dataset URI to inspect.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="How many rows to capture from stream (default: 10).",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "hf_first10_metadata.json"),
        help="Output JSON file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise SystemExit("--rows must be > 0")

    payload = _read_first_rows(args.hf_uri, args.rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({payload['rows_captured']} row(s)).")


if __name__ == "__main__":
    main()

