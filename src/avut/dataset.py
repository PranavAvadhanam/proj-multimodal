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
            rows.append(
                MCQSample(
                    sample_id=str(row["sample_id"]),
                    question=row["question"],
                    option_a=row["option_a"],
                    option_b=row["option_b"],
                    option_c=row["option_c"],
                    option_d=row["option_d"],
                    answer=row["answer"],
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


def _avut_human_prefix_rows() -> int:
    """How many leading ``train`` rows belong to AV-Human (1-based ``sample_id`` = row+1).

    Remaining rows are AV-Gemini (``sample_id`` = row_index − this value). Override if your
    Hub revision differs: ``HF_AVUT_HUMAN_ROW_COUNT``.
    """
    return int(os.environ.get("HF_AVUT_HUMAN_ROW_COUNT", "1734"))


def _uniq_ids(ids: list[str] | None, max_videos: int | None) -> set[str]:
    """Dedupe string ids, sort for stable cap, then keep at most ``max_videos`` (per side)."""
    u = sorted({str(x) for x in (ids or []) if x})
    if max_videos is not None:
        u = u[: max(0, max_videos)]
    return set(u)


def _row_keys_human_gemini(
    idx: int,
    target_h: set[str],
    target_g: set[str],
    *,
    human_rows: int,
    row: dict,
) -> tuple[list[str], list[str]]:
    """Map stream row ``idx`` to Human and/or Gemini ``sample_id`` keys still wanted.

    Prefer explicit id columns when present. Otherwise use AVUT train layout: Human
    ``sample_id`` = ``idx + 1`` for ``idx < human_rows``; Gemini ``sample_id`` = ``idx - human_rows``.
    Returns ``(human_keys, gemini_keys)`` to fetch for this row (may be empty).
    """
    for col in ("QA_id", "sample_id", "qa_id", "id"):
        if col in row and row[col] is not None:
            k = str(row[col])
            if k not in target_h and k not in target_g:
                return [], []
            kh: list[str] = []
            kg: list[str] = []
            if k in target_h:
                kh.append(k)
            if k in target_g:
                kg.append(k)
            return kh, kg
    if "video" not in row:
        return [], []
    kh: list[str] = []
    kg: list[str] = []
    if idx < human_rows:
        h = str(idx + 1)
        if h in target_h:
            kh.append(h)
    g = idx - human_rows
    if g >= 0:
        gs = str(g)
        if gs in target_g:
            kg.append(gs)
    return kh, kg


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


def prefetch_hf_avut_train_videos(
    human_qa_ids: list[str] | None,
    gemini_qa_ids: list[str] | None,
    hf_uri: str,
    max_videos: int | None,
    desc: str = "Prefetch videos (HF)",
    human_expected_video_by_id: dict[str, str] | None = None,
    gemini_expected_video_by_id: dict[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Stream Hub ``train`` into two ``sample_id`` → ``video`` maps (Human vs Gemini).
    
    ``hf_uri`` must start with ``hf://``. Uses authenticated Hub token when set. Stops once
    both target sets are filled. Reads ``row[\"video\"]`` only when that row matches at least
    one requested id.

    Args:
        human_qa_ids: AV-Human ``sample_id`` strings to resolve (or ``None``).
        gemini_qa_ids: AV-Gemini ``sample_id`` strings to resolve (or ``None``).
        hf_uri: e.g. ``hf://tsinghua-ee/AVUTBenchmark@train``.
        max_videos: Cap distinct ids per side after dedupe (``None`` = all listed).
        desc: tqdm description.

    Returns:
        ``(human_id_to_video, gemini_id_to_video)``; empty dicts on bad URI or import failure.
    """
    if not hf_uri.startswith("hf://"):
        return {}, {}

    target_h = _uniq_ids(human_qa_ids, max_videos)
    target_g = _uniq_ids(gemini_qa_ids, max_videos)
    expected_h = {
        str(k): Path(v).name
        for k, v in (human_expected_video_by_id or {}).items()
        if str(k) in target_h and isinstance(v, str) and v
    }
    expected_g = {
        str(k): Path(v).name
        for k, v in (gemini_expected_video_by_id or {}).items()
        if str(k) in target_g and isinstance(v, str) and v
    }
    if not target_h and not target_g:
        return {}, {}

    try:
        from datasets import DownloadConfig, Video, load_dataset
        from tqdm import tqdm
    except Exception:
        return {}, {}

    repo_id, split = _parse_hf_uri(hf_uri)
    token = _hf_token()
    dl_cfg = DownloadConfig(max_retries=5)
    human_rows = _avut_human_prefix_rows()

    out_h: dict[str, object] = {}
    out_g: dict[str, object] = {}
    ds = load_dataset(
        repo_id,
        split=split,
        streaming=True,
        token=token,
        download_config=dl_cfg,
    )
    # Avoid expensive video decoding during prefetch; we only need lightweight path metadata.
    if "video" in getattr(ds, "features", {}):
        ds = ds.cast_column("video", Video(decode=False))

    total = len(target_h) + len(target_g)
    pbar = tqdm(total=total, desc=desc, unit="video", dynamic_ncols=True)
    for idx, row in enumerate(ds):
        if len(out_h) >= len(target_h) and len(out_g) >= len(target_g):
            break
        kh, kg = _row_keys_human_gemini(idx, target_h, target_g, human_rows=human_rows, row=row)
        vid = row.get("video")
        if vid is None:
            continue
        vid_name = _video_filename_from_obj(vid)
        if expected_h:
            # Prefer metadata filename matching, but retain row-index/id fallback when no
            # filename match is found (dataset revisions can rename/rehash media paths).
            matched_h = [
                sid for sid, expected_name in expected_h.items() if sid not in out_h and expected_name == vid_name
            ]
            if matched_h:
                kh = matched_h
        elif vid_name:
            for sid, expected_name in expected_h.items():
                if sid not in out_h and expected_name == vid_name:
                    kh.append(sid)
        if expected_g:
            # Prefer metadata filename matching, but retain row-index/id fallback when no
            # filename match is found (dataset revisions can rename/rehash media paths).
            matched_g = [
                sid for sid, expected_name in expected_g.items() if sid not in out_g and expected_name == vid_name
            ]
            if matched_g:
                kg = matched_g
        elif vid_name:
            for sid, expected_name in expected_g.items():
                if sid not in out_g and expected_name == vid_name:
                    kg.append(sid)
        if not kh and not kg:
            continue
        for k in kh:
            if k not in out_h:
                out_h[k] = vid
                pbar.update(1)
        for k in kg:
            if k not in out_g:
                out_g[k] = vid
                pbar.update(1)
        if len(out_h) >= len(target_h) and len(out_g) >= len(target_g):
            break
    pbar.close()
    return out_h, out_g


def attach_prefetched_videos(samples: list[MCQSample], video_by_qa: dict[str, object]) -> None:
    """Set ``video_input`` on each sample whose ``sample_id`` appears in ``video_by_qa``."""
    for s in samples:
        v = video_by_qa.get(str(s.sample_id))
        if v is not None:
            # Prefer the prefetched HF object for this QA id even when the metadata basename
            # does not match the Hub filename (e.g. stale server-side paths vs re-keyed uploads).
            # Skipping attachment left ``video_input`` unset and routed inference to bogus
            # metadata paths such as ``/mnt/...`` that exist only on upstream hosts, which
            # breaks ffmpeg-based transcription while Gemini may still appear "usable".
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
