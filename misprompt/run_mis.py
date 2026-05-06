#!/usr/bin/env python3
"""Run the Modality Importance Score (MIS) calibration.

Samples X videos randomly from the AVUT dataset (enforcing separation from the
idea2 test set), evaluates all 7 non-empty modality subsets per question, and
produces softmax-normalized token allocations.

Usage:
    python -m misprompt.run_mis --mis-samples 30 --prefetch-videos 30
    python -m misprompt.run_mis --mis-samples 10 --no-prefetch-videos
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.avut.audio_extractor import (
    extract_audio_part_from_video_input,
    extract_audio_wav_bytes_from_video_input,
)
from src.avut.dataset import (
    MCQSample,
    attach_prefetched_videos,
    enrich_samples_from_metadata,
    load_samples,
    prefetch_hf_avut_train_videos,
    representative_even_sample,
)
from src.avut.prompts import audio_perception_prompt, video_perception_prompt
from src.config import Settings, get_settings
from src.mspragcot.client import GeminiClient
from src.mspragcot.modality_describer import ModalityDescriber

from misprompt.compute_mis import (
    MODALITIES,
    mis_to_token_allocation,
    run_mis_evaluation,
    softmax,
)

MIS_EXCLUSION_FILE = "mis_excluded_sample_ids.json"


def _load_existing_exclusions(output_dir: Path) -> set[str]:
    """Load previously used MIS sample IDs to maintain separation across runs."""
    path = output_dir / MIS_EXCLUSION_FILE
    if path.exists():
        return set(json.loads(path.read_text(encoding="utf-8")))
    return set()


def _save_exclusions(output_dir: Path, sample_ids: set[str]) -> None:
    path = output_dir / MIS_EXCLUSION_FILE
    path.write_text(json.dumps(sorted(sample_ids), indent=2), encoding="utf-8")


def _select_mis_samples(
    all_samples: list[MCQSample],
    n_mis: int,
    idea2_excluded_ids: set[str],
    seed: int = 42,
) -> list[MCQSample]:
    """Select MIS calibration samples, enforcing separation from idea2 test set.

    Selects from samples whose IDs are NOT in idea2_excluded_ids. Uses task-balanced
    random sampling.
    """
    eligible = [s for s in all_samples if str(s.sample_id) not in idea2_excluded_ids]
    if len(eligible) < n_mis:
        print(
            f"[MIS] WARNING: only {len(eligible)} eligible samples after exclusion "
            f"(requested {n_mis}). Using all eligible."
        )
        n_mis = len(eligible)

    selected = representative_even_sample(eligible, n_mis)

    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected[:n_mis]


def _prepare_descriptions(
    samples: list[MCQSample],
    settings: Settings,
    audio_cache_dir: Path,
) -> dict[str, dict[str, str]]:
    """Generate modality descriptions for all MIS samples.

    For each sample, produces {"text": ..., "audio": ..., "visual": ...} descriptions.
    Mirrors the idea2 pipeline's describe phase.
    """
    client = GeminiClient(settings)
    describer = ModalityDescriber(client, settings)
    descriptions: dict[str, dict[str, str]] = {}

    from tqdm import tqdm

    pbar = tqdm(total=len(samples), desc="MIS descriptions", unit="sample", colour="blue")

    for sample in samples:
        sid = str(sample.sample_id)
        client.set_run_context(
            pass_label="MIS_describe",
            sample_id=sid,
            task_code=sample.task_code or "UNKNOWN",
            task_type=sample.task_type,
        )

        descs: dict[str, str] = {"text": "", "audio": "", "visual": ""}

        has_video = client.has_usable_media(sample.video_input)

        if sample.video_input is None and sample.video_path:
            sample.video_input = sample.video_path
            has_video = client.has_usable_media(sample.video_input)

        if not has_video:
            print(f"  [MIS] SKIP sample_id={sid}: no usable video input")
            pbar.update(1)
            continue

        text_ready = bool(sample.transcript)
        audio_ready = sample.audio_input is not None

        if not text_ready and has_video:
            try:
                import base64
                import urllib.error
                import urllib.request

                import google.auth
                import google.auth.transport.requests

                audio_wav = extract_audio_wav_bytes_from_video_input(
                    video_input=sample.video_input,
                    sample_id=sid,
                    cache_dir=audio_cache_dir,
                    max_duration_seconds=settings.max_audio_duration_seconds,
                )
                if audio_wav:
                    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
                    if project_id:
                        from src.main import _transcribe_with_stt_v2

                        transcript = _transcribe_with_stt_v2(audio_wav, project_id)
                        if transcript:
                            sample.transcript = transcript
                            text_ready = True
            except Exception as exc:
                print(f"  [MIS] Transcription failed for {sid}: {exc}")

        if not audio_ready and has_video:
            audio_part = extract_audio_part_from_video_input(
                video_input=sample.video_input,
                sample_id=sid,
                cache_dir=audio_cache_dir,
            )
            if audio_part is not None:
                sample.audio_input = audio_part
                audio_ready = True

        if text_ready:
            descs["text"] = describer.describe_text(sample)

        if audio_ready:
            a_prompt = audio_perception_prompt(sample)
            text_ref = descs["text"] or "No text available."
            try:
                descs["audio"] = describer.describe_audio(sample, a_prompt, text_ref)
            except Exception as exc:
                print(f"  [MIS] Audio describe failed for {sid}: {exc}")
                descs["audio"] = ""

        if has_video:
            v_prompt = video_perception_prompt(sample)
            text_ref = descs["text"] or "No text available."
            try:
                descs["visual"] = describer.describe_video(sample, v_prompt, text_ref)
            except Exception as exc:
                print(f"  [MIS] Video describe failed for {sid}: {exc}")
                descs["visual"] = ""

        descriptions[sid] = descs
        pbar.update(1)

    pbar.close()
    return descriptions


def _get_idea2_used_ids(output_dir: Path) -> set[str]:
    """Collect sample IDs already used in idea2 predictions to enforce separation."""
    ids: set[str] = set()
    for pred_file in output_dir.glob("idea2_predictions_*.jsonl"):
        with pred_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    sid = row.get("sample_id")
                    if sid and not row.get("_run_divider"):
                        ids.add(str(sid))
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Compute Modality Importance Scores via subset ablation."
    )
    parser.add_argument(
        "--mis-samples", type=int, default=30,
        help="Number of calibration samples for MIS (default: 30).",
    )
    parser.add_argument(
        "--prefetch-videos", type=int, default=None,
        help="Max videos to prefetch from HF (default: all needed).",
    )
    parser.add_argument(
        "--no-prefetch-videos", action="store_true",
        help="Skip HF video prefetch entirely.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection (default: 42).",
    )
    parser.add_argument(
        "--total-token-budget", type=int, default=768,
        help="Total token budget to split across modalities (default: 768).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/mis/).",
    )
    args = parser.parse_args()

    settings = get_settings()
    base_out = Path(args.output_dir or os.path.join(settings.output_dir, "mis"))
    base_out.mkdir(parents=True, exist_ok=True)
    audio_cache_dir = base_out / "audio_cache"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODALITY IMPORTANCE SCORE (MIS) CALIBRATION")
    print("=" * 70)
    print(f"  MIS samples:       {args.mis_samples}")
    print(f"  Total token budget: {args.total_token_budget}")
    print(f"  Seed:              {args.seed}")
    print(f"  Output:            {base_out}")
    print(f"  Model:             {settings.gemini_model}")
    print()

    all_samples = load_samples(settings.qa_human_filtered_jsonl)
    enrich_samples_from_metadata(all_samples, settings.video_metadata_human_json)
    print(f"[MIS] Loaded {len(all_samples)} total AV-Human samples.")

    idea2_used = _get_idea2_used_ids(Path(settings.output_dir))
    prev_mis_ids = _load_existing_exclusions(base_out)
    excluded_ids = idea2_used | prev_mis_ids
    print(f"[MIS] Excluding {len(excluded_ids)} IDs (idea2={len(idea2_used)}, prev_mis={len(prev_mis_ids)}).")

    mis_samples = _select_mis_samples(all_samples, args.mis_samples, excluded_ids, seed=args.seed)
    mis_ids = {str(s.sample_id) for s in mis_samples}
    print(f"[MIS] Selected {len(mis_samples)} calibration samples.")

    _save_exclusions(base_out, prev_mis_ids | mis_ids)

    if not args.no_prefetch_videos:
        n_vids = len({s.video_id or s.sample_id for s in mis_samples})
        print(f"[MIS] Prefetching up to {n_vids} videos from HF...")
        vmap_h, _ = prefetch_hf_avut_train_videos(
            mis_samples,
            None,
            settings.hf_video_dataset_uri,
            max_videos=args.prefetch_videos,
            desc="MIS prefetch",
        )
        attach_prefetched_videos(mis_samples, vmap_h)
    else:
        print("[MIS] Skipping video prefetch (--no-prefetch-videos).")

    print("\n[MIS] Phase 1: Generating modality descriptions...")
    descriptions = _prepare_descriptions(mis_samples, settings, audio_cache_dir)

    usable_samples = [s for s in mis_samples if str(s.sample_id) in descriptions]
    print(f"[MIS] {len(usable_samples)}/{len(mis_samples)} samples have descriptions ready.")

    if not usable_samples:
        print("[MIS] ERROR: No usable samples. Check video availability.")
        sys.exit(1)

    print(f"\n[MIS] Phase 2: Evaluating {len(usable_samples)} samples x 7 subsets = {len(usable_samples)*7} inferences...")
    t0 = time.perf_counter()

    output = run_mis_evaluation(
        usable_samples,
        descriptions,
        settings,
        base_out,
    )

    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("MIS RESULTS")
    print("=" * 70)
    print(f"\n  Raw MIS scores:")
    for m in MODALITIES:
        print(f"    {m:>7}: {output['raw_mis_scores'][m]:+.4f}")

    print(f"\n  Softmax weights:")
    for m in MODALITIES:
        print(f"    {m:>7}: {output['softmax_weights'][m]:.4f}")

    print(f"\n  Token allocation (budget={args.total_token_budget}):")
    token_alloc = mis_to_token_allocation(output["raw_mis_scores"], args.total_token_budget)
    for m in MODALITIES:
        print(f"    {m:>7}: {token_alloc[m]} tokens")

    print(f"\n  Per-subset accuracy:")
    for subset_key, acc in sorted(output["per_subset_accuracy"].items()):
        print(f"    {subset_key:>20}: {acc:.4f}")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Outputs: {base_out}/mis_results.json, {base_out}/mis_detail.jsonl")
    print(f"  Exclusion list: {base_out}/{MIS_EXCLUSION_FILE}")
    print("=" * 70)

    alloc_path = base_out / "token_allocation.json"
    alloc_path.write_text(json.dumps({
        "total_budget": args.total_token_budget,
        "allocation": token_alloc,
        "softmax_weights": output["softmax_weights"],
        "raw_mis_scores": output["raw_mis_scores"],
    }, indent=2), encoding="utf-8")
    print(f"\n  Token allocation saved: {alloc_path}")


if __name__ == "__main__":
    main()
