from __future__ import annotations

import argparse
import os
import pprint
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# Load .env before Hugging Face / datasets (so HF_TOKEN and GEMINI_* are visible on first Hub use).
load_dotenv(ROOT / ".env")
load_dotenv()

# Optional faster Hub downloads (install `hf-transfer`); safe if package missing.
try:
    import hf_transfer  # noqa: F401

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

# Silence per-file Hugging Face download bars; we use one tqdm for prefetch in src.main.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import TranscriptionFailure, run_idea2_pipeline


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal AVUT Idea 2 pipeline.")
    parser.add_argument(
        "--input",
        required=False,
        default=None,
        help="Optional QA JSONL: single-pass debug override (disables dual AV-Human / AV-Gemini runs).",
    )
    parser.add_argument("--output-dir", default=None, help="Optional override for output dir.")
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help=(
            "Sample cap. Default mode runs AV-Human only with up to N rows "
            "(task-balanced). With --split-max-samples, N is total rows split across "
            "AV-Human and AV-Gemini."
        ),
    )
    parser.add_argument(
        "--split-max-samples",
        action="store_true",
        help="Run both AV-Human and AV-Gemini, splitting --max-samples across the two passes.",
    )
    parser.add_argument(
        "--run-sample",
        type=str,
        default=None,
        help="Run one AV-Human sample_id as a minimal end-to-end pass (replaces old --max-samples=-1 behavior).",
    )
    parser.add_argument(
        "--prefetch-videos",
        type=int,
        default=None,
        help="Max distinct HF videos to prefetch (one progress bar). Default: all QA_ids needed for selected passes.",
    )
    parser.add_argument(
        "--no-prefetch-videos",
        action="store_true",
        help="Skip Hugging Face video prefetch (much faster; pipeline runs without video_input).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_sample is not None and args.input is not None:
        raise ValueError("--run-sample cannot be used with --input.")
    if args.split_max_samples and args.max_samples is None:
        raise ValueError("--split-max-samples requires --max-samples.")
    try:
        metrics = run_idea2_pipeline(
            args.input,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            run_sample=args.run_sample,
            prefetch_videos=args.prefetch_videos,
            no_prefetch_videos=args.no_prefetch_videos,
            split_max_samples=args.split_max_samples,
        )
    except TranscriptionFailure as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()
