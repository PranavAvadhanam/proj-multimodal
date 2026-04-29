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

# Silence per-file Hugging Face download bars; we use one tqdm for prefetch in src pipeline.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vanilla.pipeline import run_vanilla_pipeline


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vanilla AVUT baseline pipeline.")
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
        help="Per-pass cap (default: two passes). Up to N rows from AV-Human pass AND up to N from AV-Gemini pass; each pass task-balanced. Must be positive.",
    )
    parser.add_argument(
        "--run-sample",
        type=str,
        default=None,
        help="Run one AV-Human sample_id as a minimal end-to-end pass.",
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
        help="Skip Hugging Face video prefetch (faster, but baseline video QA will fail without video_input).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_sample is not None and args.input is not None:
        raise ValueError("--run-sample cannot be used with --input.")
    metrics = run_vanilla_pipeline(
        args.input,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        run_sample=args.run_sample,
        prefetch_videos=args.prefetch_videos,
        no_prefetch_videos=args.no_prefetch_videos,
    )
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()

