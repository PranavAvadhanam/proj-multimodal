from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


# remote filename (from HF repo) -> local filename (in data/avut/hf_raw)
TARGET_FILE_MAP: dict[str, str] = {
    "AV_Human_filtered_data.json": "AV_Human_filtered_data.json",
    "AV_Gemini_filtered_data.json": "AV_Gemini_filtered_data.json",
}
OUTPUT_JSONL_BY_RAW_JSON: dict[str, str] = {
    "AV_Human_filtered_data.json": "avut_human_filtered.jsonl",
    "AV_Gemini_filtered_data.json": "avut_gemini_filtered.jsonl",
}


def _resolve_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _find_remote_path(repo_files: list[str], target_name: str) -> str:
    """Find a repo file path ending with the expected target filename."""
    matches = [p for p in repo_files if p.endswith(target_name)]
    if not matches:
        raise FileNotFoundError(f"Could not find {target_name!r} in Hugging Face repo listing.")
    # Prefer top-level path when multiple files share name.
    matches.sort(key=lambda p: (p.count("/"), p))
    return matches[0]


def download_hf_raw_metadata(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    out_dir: Path,
    token: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[HF] Listing files for {repo_type}:{repo_id}@{revision} ...")
    repo_files = list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision, token=token)

    for remote_name, local_name in TARGET_FILE_MAP.items():
        remote_path = _find_remote_path(repo_files, remote_name)
        print(f"[HF] Downloading {remote_path} ...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=remote_path,
            token=token,
        )
        dst = out_dir / local_name
        shutil.copyfile(downloaded, dst)
        print(f"[OK] Overwrote {dst}")


def _row_to_jsonl_record(row: dict) -> dict:
    return {
        # Preserve original identifiers/metadata.
        "qa_id": row.get("QA_id"),
        "video_id": row.get("video_id"),
        "url": row.get("url"),
        "video_type": row.get("video_type"),
        "task_type": row.get("task_type"),
        "video_path": row.get("video_path"),
        "audio_path": row.get("audio_path"),
        "transcript": row.get("transcript"),
        # Keep normalized fields expected by current pipeline code.
        "sample_id": str(row.get("QA_id", "")),
        "question": row.get("question"),
        "option_a": row.get("option_A"),
        "option_b": row.get("option_B"),
        "option_c": row.get("option_C"),
        "option_d": row.get("option_D"),
        "answer": row.get("answer"),
    }


def build_filtered_jsonl_from_hf_raw(*, hf_raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw_name, out_jsonl_name in OUTPUT_JSONL_BY_RAW_JSON.items():
        raw_path = hf_raw_dir / raw_name
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing required hf_raw file: {raw_path}")
        rows = json.loads(raw_path.read_text(encoding="utf-8"))
        out_path = out_dir / out_jsonl_name
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                rec = _row_to_jsonl_record(row)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] Overwrote {out_path} from {raw_path.name} ({len(rows)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload AVUT hf_raw filtered metadata JSON files from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default="tsinghua-ee/AVUTBenchmark",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset"],
        help="Hugging Face repo type.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repo revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "hf_raw"),
        help="Output directory for overwritten hf_raw metadata files.",
    )
    parser.add_argument(
        "--jsonl-out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output directory for regenerated filtered JSONL files.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Hugging Face download and only regenerate JSONL from existing hf_raw files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_raw_out_dir = Path(args.out_dir)
    jsonl_out_dir = Path(args.jsonl_out_dir)
    token = _resolve_token()
    if not args.skip_download:
        download_hf_raw_metadata(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            out_dir=hf_raw_out_dir,
            token=token,
        )
    build_filtered_jsonl_from_hf_raw(hf_raw_dir=hf_raw_out_dir, out_dir=jsonl_out_dir)


if __name__ == "__main__":
    main()

