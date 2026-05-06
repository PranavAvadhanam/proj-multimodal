"""AVUT QA loading, metadata enrichment, HF video prefetch, and Table-3 task-balanced sampling."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class MCQSample:
    """One multiple-choice AVUT row (JSONL) plus optional paths and decoded media handles."""

    sample_id: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    answer: str
    video_id: str | None = None
    transcript: str | None = None
    video_path: str | None = None
    audio_path: str | None = None
    video_input: object | None = None
    audio_input: object | None = None
    task_type: str | None = None
    task_code: str | None = None

    @property
    def options(self) -> dict[str, str]:
        """Map option letter ``A``–``D`` to choice text."""
        return {
            "A": self.option_a,
            "B": self.option_b,
            "C": self.option_c,
            "D": self.option_d,
        }


def load_samples(jsonl_path: str | Path) -> list[MCQSample]:
    """Read AVUT-style JSONL into ``MCQSample`` rows (skips blank lines)."""
    path = Path(jsonl_path)
    rows: list[MCQSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            vid_raw = row.get("video_id")
            rows.append(
                MCQSample(
                    sample_id=str(row["sample_id"]),
                    question=row["question"],
                    option_a=row["option_a"],
                    option_b=row["option_b"],
                    option_c=row["option_c"],
                    option_d=row["option_d"],
                    answer=row["answer"],
                    video_id=str(vid_raw) if vid_raw is not None else None,
                    transcript=row.get("transcript"),
                    video_path=row.get("video_path"),
                    audio_path=row.get("audio_path"),
                    video_input=row.get("video_input"),
                    audio_input=row.get("audio_input"),
                    task_type=row.get("task_type"),
                    task_code=canonical_task_code(row.get("task_type")),
                )
            )
    return rows


def iter_samples(samples: list[MCQSample]) -> Iterable[MCQSample]:
    """Yield each sample (thin iterator over a list)."""
    for sample in samples:
        yield sample


def enrich_samples_from_metadata(samples: list[MCQSample], metadata_json_path: str | Path) -> None:
    """Fill missing task_type, task_code, video_path, audio_path by ``sample_id`` ↔ ``QA_id`` match.

    No-op if the JSON file is missing. Only overwrites fields that are still empty on each sample.
    """
    path = Path(metadata_json_path)
    if not path.exists():
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_id = {str(r.get("QA_id")): r for r in rows if r.get("QA_id") is not None}
    for s in samples:
        meta = by_id.get(str(s.sample_id))
        if not meta:
            continue
        if not s.video_id and meta.get("video_id") is not None:
            s.video_id = str(meta["video_id"])
        if not s.task_type:
            s.task_type = meta.get("task_type")
        if not s.task_code:
            s.task_code = canonical_task_code(s.task_type)
        if not s.video_path:
            s.video_path = meta.get("video_path")
        if not s.audio_path:
            s.audio_path = meta.get("audio_path")


def _hf_token() -> str | bool | None:
    """HF Hub token: prefer HF_TOKEN, then HUGGING_FACE_HUB_TOKEN. None = anonymous."""
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t
    return None


def _parse_hf_uri(hf_uri: str) -> tuple[str, str]:
    """Parse ``hf://org/name@split`` into ``(repo_id, split)``; default split ``train``."""
    repo_and_split = hf_uri.removeprefix("hf://")
    if "@" in repo_and_split:
        repo_id, split = repo_and_split.split("@", 1)
    else:
        repo_id, split = repo_and_split, "train"
    return repo_id, split


def _video_filename_from_obj(video_obj: object) -> str | None:
    """Best-effort basename extraction from datasets video feature payload."""
    if isinstance(video_obj, dict):
        p = video_obj.get("path")
        if isinstance(p, str) and p:
            return Path(p).name
    p_attr = getattr(video_obj, "path", None)
    if isinstance(p_attr, str) and p_attr:
        return Path(p_attr).name
    if isinstance(video_obj, str) and video_obj:
        return Path(video_obj).name
    return None


def _build_video_id_maps(
    samples: list[MCQSample] | None,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    """From a list of samples build three lookup dicts keyed by ``video_id``.

    Returns:
        ``(vid_to_sids, vid_to_filename, sid_to_vid)``
        - ``vid_to_sids``: ``video_id`` → list of ``sample_id`` strings needing that video.
        - ``vid_to_filename``: ``video_id`` → expected video basename (from ``video_path``).
        - ``sid_to_vid``: ``sample_id`` → ``video_id`` (reverse lookup).
    """
    vid_to_sids: dict[str, list[str]] = defaultdict(list)
    vid_to_filename: dict[str, str] = {}
    sid_to_vid: dict[str, str] = {}
    for s in (samples or []):
        vid = s.video_id or str(s.sample_id)
        sid = str(s.sample_id)
        vid_to_sids[vid].append(sid)
        sid_to_vid[sid] = vid
        if s.video_path and vid not in vid_to_filename:
            vid_to_filename[vid] = Path(s.video_path).name
    return dict(vid_to_sids), vid_to_filename, sid_to_vid


def prefetch_hf_avut_train_videos(
    human_samples: list[MCQSample] | None,
    gemini_samples: list[MCQSample] | None,
    hf_uri: str,
    max_videos: int | None = None,
    desc: str = "Prefetch videos (HF)",
) -> tuple[dict[str, object], dict[str, object]]:
    """Stream Hub ``train`` into two ``sample_id`` → ``video`` maps (Human vs Gemini).

    Keyed entirely by ``video_id``: one HF video per unique ``video_id``, then fanned out
    to every ``sample_id`` sharing that ``video_id``. ``sample_id`` is **not** used for
    matching HF rows.

    Matching priority per HF row:
      1. Video filename from the HF row's video object matches an expected filename
         derived from ``video_path`` in the QA metadata.
      2. Explicit ``video_id`` column in the HF row matches a target ``video_id``.
      3. ``QA_id`` / ``sample_id`` column in the HF row → reverse-lookup to ``video_id``.

    Args:
        human_samples: AV-Human ``MCQSample`` list (or ``None``).
        gemini_samples: AV-Gemini ``MCQSample`` list (or ``None``).
        hf_uri: e.g. ``hf://tsinghua-ee/AVUTBenchmark@train``.
        max_videos: Cap distinct *video_ids* per side (``None`` = all needed).
        desc: tqdm description.

    Returns:
        ``(human_sid_to_video, gemini_sid_to_video)``; empty dicts on bad URI or import
        failure.
    """
    if not hf_uri.startswith("hf://"):
        return {}, {}

    h_vid_to_sids, h_vid_to_fn, h_sid_to_vid = _build_video_id_maps(human_samples)
    g_vid_to_sids, g_vid_to_fn, g_sid_to_vid = _build_video_id_maps(gemini_samples)

    target_h_vids = set(sorted(h_vid_to_sids.keys()))
    target_g_vids = set(sorted(g_vid_to_sids.keys()))
    if max_videos is not None:
        target_h_vids = set(sorted(target_h_vids)[:max_videos])
        target_g_vids = set(sorted(target_g_vids)[:max_videos])

    if not target_h_vids and not target_g_vids:
        return {}, {}

    # Reverse filename → video_id for fast lookup during streaming.
    h_fn_to_vid: dict[str, str] = {}
    for vid, fn in h_vid_to_fn.items():
        if vid in target_h_vids:
            h_fn_to_vid.setdefault(fn, vid)
    g_fn_to_vid: dict[str, str] = {}
    for vid, fn in g_vid_to_fn.items():
        if vid in target_g_vids:
            g_fn_to_vid.setdefault(fn, vid)

    try:
        from datasets import DownloadConfig, Video, load_dataset
        from tqdm import tqdm
    except Exception:
        return {}, {}

    repo_id, split = _parse_hf_uri(hf_uri)
    token = _hf_token()
    dl_cfg = DownloadConfig(max_retries=5)

    found_h: dict[str, object] = {}  # video_id → video obj
    found_g: dict[str, object] = {}

    ds = load_dataset(repo_id, split=split, streaming=True, token=token, download_config=dl_cfg)
    if "video" in getattr(ds, "features", {}):
        ds = ds.cast_column("video", Video(decode=False))

    total_target = len(target_h_vids) + len(target_g_vids)
    pbar = tqdm(total=total_target, desc=desc, unit="video", dynamic_ncols=True)

    for _idx, row in enumerate(ds):
        if len(found_h) >= len(target_h_vids) and len(found_g) >= len(target_g_vids):
            break
        vid_obj = row.get("video")
        if vid_obj is None:
            continue

        vid_name = _video_filename_from_obj(vid_obj)
        matched_h: str | None = None
        matched_g: str | None = None

        # Strategy 1: match by video filename → video_id
        if vid_name:
            h_cand = h_fn_to_vid.get(vid_name)
            if h_cand and h_cand in target_h_vids and h_cand not in found_h:
                matched_h = h_cand
            g_cand = g_fn_to_vid.get(vid_name)
            if g_cand and g_cand in target_g_vids and g_cand not in found_g:
                matched_g = g_cand

        # Strategy 2: explicit video_id column in the HF row
        if matched_h is None or matched_g is None:
            row_vid = row.get("video_id")
            if row_vid is not None:
                k = str(row_vid)
                if matched_h is None and k in target_h_vids and k not in found_h:
                    matched_h = k
                if matched_g is None and k in target_g_vids and k not in found_g:
                    matched_g = k

        # Strategy 3: QA_id / sample_id column → reverse-lookup to video_id
        if matched_h is None or matched_g is None:
            for col in ("QA_id", "sample_id", "qa_id", "id"):
                if col in row and row[col] is not None:
                    qa_key = str(row[col])
                    if matched_h is None:
                        h_vid_for_qa = h_sid_to_vid.get(qa_key)
                        if h_vid_for_qa and h_vid_for_qa in target_h_vids and h_vid_for_qa not in found_h:
                            matched_h = h_vid_for_qa
                    if matched_g is None:
                        g_vid_for_qa = g_sid_to_vid.get(qa_key)
                        if g_vid_for_qa and g_vid_for_qa in target_g_vids and g_vid_for_qa not in found_g:
                            matched_g = g_vid_for_qa
                    break

        if matched_h is not None:
            found_h[matched_h] = vid_obj
            pbar.update(1)
        if matched_g is not None:
            found_g[matched_g] = vid_obj
            pbar.update(1)

    pbar.close()

    # --- Safety net: verify filename consistency across video_ids ---
    for label, found, vid_to_fn in [
        ("Human", found_h, h_vid_to_fn),
        ("Gemini", found_g, g_vid_to_fn),
    ]:
        for vid, obj in found.items():
            actual = _video_filename_from_obj(obj)
            expected = vid_to_fn.get(vid)
            if actual and expected and actual != expected:
                import warnings
                warnings.warn(
                    f"[{label}] video_id={vid}: HF filename {actual!r} != "
                    f"expected {expected!r} from metadata. Using HF object anyway.",
                    stacklevel=2,
                )

    # --- Fan-out: video_id → video to sample_id → video ---
    out_h: dict[str, object] = {}
    for vid, obj in found_h.items():
        for sid in h_vid_to_sids.get(vid, []):
            out_h[sid] = obj

    out_g: dict[str, object] = {}
    for vid, obj in found_g.items():
        for sid in g_vid_to_sids.get(vid, []):
            out_g[sid] = obj

    h_found_sids = len(out_h)
    h_total_sids = sum(len(sids) for v, sids in h_vid_to_sids.items() if v in target_h_vids)
    g_found_sids = len(out_g)
    g_total_sids = sum(len(sids) for v, sids in g_vid_to_sids.items() if v in target_g_vids)
    print(
        f"[Prefetch] Resolved: Human {len(found_h)}/{len(target_h_vids)} video_ids "
        f"→ {h_found_sids}/{h_total_sids} samples, "
        f"Gemini {len(found_g)}/{len(target_g_vids)} video_ids "
        f"→ {g_found_sids}/{g_total_sids} samples"
    )

    return out_h, out_g


def attach_prefetched_videos(samples: list[MCQSample], video_by_sid: dict[str, object]) -> None:
    """Set ``video_input`` on each sample whose ``sample_id`` appears in ``video_by_sid``."""
    for s in samples:
        v = video_by_sid.get(str(s.sample_id))
        if v is not None:
            s.video_input = v


def canonical_task_code(task_type: str | None) -> str | None:
    """Normalize free-text ``task_type`` to AVUT Table-3 code (``AIE``, ``ACC``, …) or ``None``."""
    if not task_type:
        return None
    normalized = task_type.strip().lower()
    mapping = {
        "audio information extraction": "AIE",
        "audio content counting": "ACC",
        "audio event location": "AEL",
        "audio event localization": "AEL",
        "audio character matching": "AVCM",
        "audio visual character matching": "AVCM",
        "audio object matching": "AVOM",
        "audio visual object matching": "AVOM",
        "audio ocr matching": "AVTM",
        "audio text matching": "AVTM",
        "audio visual text matching": "AVTM",
    }
    return mapping.get(normalized)


def representative_even_sample(samples: list[MCQSample], max_samples: int) -> list[MCQSample]:
    """Pick up to ``max_samples`` rows spread across Table-3 task codes when labels exist.

    Partitions by ``task_code``, allocates roughly ``max_samples // 6`` per known code with
    remainder distributed one-by-one in fixed code order, then backfills from leftovers and
    unknown-task rows if groups are short. If no known codes, returns the first ``max_samples``.
    """
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples[:max_samples] if max_samples > 0 else []

    grouped: dict[str, list[MCQSample]] = defaultdict(list)
    unknown: list[MCQSample] = []
    ordered_codes = ["AIE", "ACC", "AEL", "AVCM", "AVOM", "AVTM"]
    for s in samples:
        if s.task_code in ordered_codes:
            grouped[s.task_code].append(s)
        else:
            unknown.append(s)

    # If no valid task labels exist, fallback to deterministic slice.
    if not any(grouped.values()):
        return samples[:max_samples]

    target_per_group = max_samples // len(ordered_codes)
    remainder = max_samples % len(ordered_codes)

    selected: list[MCQSample] = []
    for code in ordered_codes:
        take_n = target_per_group + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        selected.extend(grouped[code][:take_n])

    # Backfill if some groups are smaller than target.
    if len(selected) < max_samples:
        leftovers: list[MCQSample] = []
        for code in ordered_codes:
            already = sum(1 for s in selected if s.task_code == code)
            leftovers.extend(grouped[code][already:])
        leftovers.extend(unknown)
        need = max_samples - len(selected)
        selected.extend(leftovers[:need])

    return selected[:max_samples]
