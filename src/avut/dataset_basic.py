"""Minimal one-sample AVUT loader for quick smoke tests."""

from __future__ import annotations

from src.avut.dataset import (
    MCQSample,
    attach_prefetched_videos,
    enrich_samples_from_metadata,
    load_samples,
    prefetch_hf_avut_train_videos,
)
from src.config import Settings


def load_basic_human_sample(settings: Settings, sample_id: str | None = None) -> list[MCQSample]:
    """Load exactly one AV-Human QA sample and attach one matching HF video.

    This intentionally keeps the flow minimal for command-line smoke tests that still
    exercise the same QA/video path as the full pipeline.

    Args:
        settings: Runtime settings.
        sample_id: Optional AV-Human sample_id to run. If omitted, selects the first row.
    """
    samples = load_samples(settings.qa_human_filtered_jsonl)
    if not samples:
        return []

    enrich_samples_from_metadata(samples, settings.video_metadata_human_json)
    if sample_id is None:
        sample = samples[0]
    else:
        target = str(sample_id)
        sample = next((s for s in samples if str(s.sample_id) == target), None)
        if sample is None:
            raise ValueError(
                f"AV-Human sample_id={sample_id!r} not found in {settings.qa_human_filtered_jsonl}."
            )
    human_map, _ = prefetch_hf_avut_train_videos(
        [sample],
        None,
        settings.hf_video_dataset_uri,
        max_videos=1,
        desc="Prefetch videos (HF basic)",
    )
    attach_prefetched_videos([sample], human_map)
    return [sample]
