from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.avut.dataset import (
    MCQSample,
    attach_prefetched_videos,
    enrich_samples_from_metadata,
    iter_samples,
    load_samples,
    prefetch_hf_avut_train_videos,
    representative_even_sample,
)
from src.avut.dataset_basic import load_basic_human_sample
from src.config import Settings, get_settings
from src.avut.prompts import (
    final_mcq_answer_format_prompt,
    vanilla_mcq_answer_preamble_prompt,
)
from src.eval.metrics import accuracy
from src.mspragcot.client import GeminiClient


def _hf_to_hub_url(hf_path: str, mode: str = "blob") -> str | None:
    """Convert hf://datasets/... path to a Hugging Face Hub URL."""
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
    return f"https://huggingface.co/datasets/{repo}/{mode}/{rev}/{rel}"


def _sample_media_uri(sample: MCQSample) -> str:
    """Best-effort URI/path string for the sample's video (metadata-first)."""
    if sample.video_path:
        return _hf_to_hub_url(sample.video_path, mode="blob") or sample.video_path
    v = sample.video_input
    hf_encoded = getattr(v, "_hf_encoded", None)
    if isinstance(hf_encoded, dict):
        hfp = hf_encoded.get("path")
        if isinstance(hfp, str) and hfp:
            return _hf_to_hub_url(hfp, mode="blob") or hfp
    if isinstance(v, dict):
        p = v.get("path")
        if isinstance(p, str) and p:
            return _hf_to_hub_url(p, mode="blob") or p
    if isinstance(v, str) and v:
        return _hf_to_hub_url(v, mode="blob") or v
    p_attr = getattr(v, "path", None)
    if isinstance(p_attr, str) and p_attr:
        return _hf_to_hub_url(p_attr, mode="blob") or p_attr
    return ""


def _sample_effective_video_url(sample: MCQSample) -> str:
    """Best-effort https URL for the exact media object used at inference time."""
    v = sample.video_input
    hf_encoded = getattr(v, "_hf_encoded", None)
    if isinstance(hf_encoded, dict):
        hfp = hf_encoded.get("path")
        if isinstance(hfp, str) and hfp:
            if hfp.startswith("hf://datasets/"):
                return _hf_to_hub_url(hfp, mode="resolve") or ""
            if hfp.startswith("https://"):
                return hfp
    if isinstance(v, dict):
        p = v.get("path")
        if isinstance(p, str) and p:
            if p.startswith("hf://datasets/"):
                return _hf_to_hub_url(p, mode="resolve") or ""
            if p.startswith("https://"):
                return p
    if isinstance(v, str):
        if v.startswith("hf://datasets/"):
            return _hf_to_hub_url(v, mode="resolve") or ""
        if v.startswith("https://"):
            return v
    p_attr = getattr(v, "path", None)
    if isinstance(p_attr, str):
        if p_attr.startswith("hf://datasets/"):
            return _hf_to_hub_url(p_attr, mode="resolve") or ""
        if p_attr.startswith("https://"):
            return p_attr
    return ""


def _sample_pbar(total: int, desc: str) -> tqdm:
    try:
        return tqdm(total=total, desc=desc, unit="sample", dynamic_ncols=True, colour="cyan")
    except TypeError:
        return tqdm(total=total, desc=desc, unit="sample", dynamic_ncols=True)


def _prepare_pass_samples(
    *,
    pass_label: str,
    qa_jsonl_path: str,
    metadata_paths: tuple[str, ...],
    max_samples: int | None,
) -> list[MCQSample]:
    samples = load_samples(qa_jsonl_path)
    for metadata_path in metadata_paths:
        enrich_samples_from_metadata(samples, metadata_path)
    _print_task_distribution(f"[{pass_label}] Loaded dataset", samples)
    if max_samples is not None:
        samples = representative_even_sample(samples, max_samples)
        _print_task_distribution(f"[{pass_label}] Selected subset", samples)
    return samples


def _build_vanilla_prompt(sample: MCQSample) -> str:
    return (
        f"{vanilla_mcq_answer_preamble_prompt(sample)}\n\n"
        f"{final_mcq_answer_format_prompt()}"
    )


def run_vanilla_pipeline(
    input_jsonl: str | None = None,
    output_dir: str | None = None,
    max_samples: int | None = None,
    run_sample: str | None = None,
    prefetch_videos: int | None = None,
    no_prefetch_videos: bool = False,
    split_max_samples: bool = False,
) -> dict:
    """Run vanilla baseline: one direct video+question prompt per sample.

    Default mode runs AV-Human only. Optional split mode runs AV-Human and AV-Gemini
    with --max-samples interpreted as a total budget split between passes.
    """
    settings = get_settings()
    base_out = Path(output_dir or settings.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    if run_sample is not None:
        if input_jsonl is not None:
            raise ValueError("--run-sample cannot be used with --input.")
        print(
            f"[Basic mode] --run-sample={run_sample} loads one AV-Human QA row and one "
            "matching HF video for a minimal end-to-end baseline run.\n"
        )
        samples = load_basic_human_sample(settings, sample_id=run_sample)
        basic_sample_id = str(samples[0].sample_id) if samples else ""
        basic_sample_uri = _sample_media_uri(samples[0]) if samples else ""
        basic_effective_video_url = _sample_effective_video_url(samples[0]) if samples else ""
        if samples:
            print(
                f"[AV-Human-basic] sample_id={basic_sample_id} "
                f"video_url_used={basic_effective_video_url or '<unknown>'} "
                f"metadata_video_uri={basic_sample_uri or '<unknown>'}\n"
            )
        metrics = _run_pass_inference(
            settings=settings,
            pass_label="AV-Human-basic",
            table3_note="single sample + single HF video",
            samples=samples,
            qa_jsonl_path=settings.qa_human_filtered_jsonl,
            metadata_paths=(settings.video_metadata_human_json,),
            out_dir=base_out,
            file_slug="av_human_basic",
            extra_metrics={
                "sample_id": basic_sample_id,
                "sample_video_uri": basic_sample_uri,
                "sample_video_url_used": basic_effective_video_url,
            },
        )
        return {"AV_Human_basic": metrics}

    if split_max_samples and max_samples is None:
        raise ValueError("--split-max-samples requires --max-samples.")

    if input_jsonl is not None:
        print(
            "[Single-pass mode] --input overrides dual AV-Human / AV-Gemini passes.\n"
            f"  QA: {input_jsonl}\n"
            "  Metadata: both human + Gemini filtered JSONs (for QA_id lookup only)."
        )
        samples = _prepare_pass_samples(
            pass_label="single",
            qa_jsonl_path=input_jsonl,
            metadata_paths=(
                settings.video_metadata_human_json,
                settings.video_metadata_gemini_json,
            ),
            max_samples=max_samples,
        )
        if no_prefetch_videos:
            print("[Prefetch] Skipped (--no-prefetch-videos).\n")
            vmap = {}
        else:
            n_vids = len({s.video_id or s.sample_id for s in samples})
            print(
                f"[Prefetch] Resolving up to {n_vids} distinct video_id(s) for {len(samples)} sample row(s) "
                f"(streaming, one bar).\n"
            )
            use_gemini = "gemini" in Path(input_jsonl).name.lower()
            oh, og = prefetch_hf_avut_train_videos(
                None if use_gemini else samples,
                samples if use_gemini else None,
                settings.hf_video_dataset_uri,
                max_videos=prefetch_videos,
                desc="Prefetch videos (HF)",
            )
            vmap = og if use_gemini else oh
        attach_prefetched_videos(samples, vmap)
        metrics = _run_pass_inference(
            settings=settings,
            pass_label="single",
            table3_note="custom QA path",
            samples=samples,
            qa_jsonl_path=input_jsonl,
            metadata_paths=(
                settings.video_metadata_human_json,
                settings.video_metadata_gemini_json,
            ),
            out_dir=base_out,
            file_slug="single",
        )
        return {"single": metrics}

    if split_max_samples:
        assert max_samples is not None
        max_h = (max_samples + 1) // 2
        max_g = max_samples // 2
        print(
            "AVUT split mode: metrics are reported for both AV-Human and AV-Gemini.\n"
            f"--max-samples={max_samples!r}: total budget split as AV-Human={max_h}, AV-Gemini={max_g}.\n"
            f"--prefetch-videos={prefetch_videos!r}: caps distinct ids prefetched per side "
            "(default: all needed on each side). Use --no-prefetch-videos to skip Hub entirely.\n"
        )

        samples_human = _prepare_pass_samples(
            pass_label="AV-Human",
            qa_jsonl_path=settings.qa_human_filtered_jsonl,
            metadata_paths=(settings.video_metadata_human_json,),
            max_samples=max_h,
        )
        samples_gemini = _prepare_pass_samples(
            pass_label="AV-Gemini",
            qa_jsonl_path=settings.qa_gemini_filtered_jsonl,
            metadata_paths=(settings.video_metadata_gemini_json,),
            max_samples=max_g,
        )
        if no_prefetch_videos:
            print("[Prefetch] Skipped (--no-prefetch-videos).\n")
            vmap_h, vmap_g = {}, {}
        else:
            nh = len({s.video_id or s.sample_id for s in samples_human})
            ng = len({s.video_id or s.sample_id for s in samples_gemini})
            cap_note = (
                f" (each side capped to {prefetch_videos} video_id(s))"
                if prefetch_videos is not None
                else ""
            )
            print(
                f"[Prefetch] AV-Human {nh} distinct video_id(s), AV-Gemini {ng} distinct video_id(s); "
                f"one HF stream / one bar{cap_note}.\n"
            )
            vmap_h, vmap_g = prefetch_hf_avut_train_videos(
                samples_human,
                samples_gemini,
                settings.hf_video_dataset_uri,
                max_videos=prefetch_videos,
                desc="Prefetch videos (HF)",
            )
        attach_prefetched_videos(samples_human, vmap_h)
        attach_prefetched_videos(samples_gemini, vmap_g)

        m_human = _run_pass_inference(
            settings=settings,
            pass_label="AV-Human",
            table3_note="Table 3 left of '/'",
            samples=samples_human,
            qa_jsonl_path=settings.qa_human_filtered_jsonl,
            metadata_paths=(settings.video_metadata_human_json,),
            out_dir=base_out,
            file_slug="av_human",
        )
        m_gemini = _run_pass_inference(
            settings=settings,
            pass_label="AV-Gemini",
            table3_note="Table 3 right of '/'",
            samples=samples_gemini,
            qa_jsonl_path=settings.qa_gemini_filtered_jsonl,
            metadata_paths=(settings.video_metadata_gemini_json,),
            out_dir=base_out,
            file_slug="av_gemini",
        )
        return {"AV_Human": m_human, "AV_Gemini": m_gemini}

    # Default mode: AV-Human only.
    n = max_samples if max_samples is not None else "all"
    print(
        "AVUT default mode: running AV-Human only.\n"
        f"--max-samples={max_samples!r}: up to {n} AV-Human QA rows (task-balanced).\n"
        f"--prefetch-videos={prefetch_videos!r}: caps AV-Human distinct ids prefetched "
        "(default: all needed). Use --no-prefetch-videos to skip Hub entirely.\n"
    )

    samples_human = _prepare_pass_samples(
        pass_label="AV-Human",
        qa_jsonl_path=settings.qa_human_filtered_jsonl,
        metadata_paths=(settings.video_metadata_human_json,),
        max_samples=max_samples,
    )
    if no_prefetch_videos:
        print("[Prefetch] Skipped (--no-prefetch-videos).\n")
        vmap_h = {}
    else:
        n_vids = len({s.video_id or s.sample_id for s in samples_human})
        cap_note = f" (capped to {prefetch_videos} video_id(s))" if prefetch_videos is not None else ""
        print(f"[Prefetch] AV-Human {n_vids} distinct video_id(s); one HF stream / one bar{cap_note}.\n")
        vmap_h, _ = prefetch_hf_avut_train_videos(
            samples_human,
            None,
            settings.hf_video_dataset_uri,
            max_videos=prefetch_videos,
            desc="Prefetch videos (HF)",
        )
    attach_prefetched_videos(samples_human, vmap_h)

    m_human = _run_pass_inference(
        settings=settings,
        pass_label="AV-Human",
        table3_note="AV-Human only (default)",
        samples=samples_human,
        qa_jsonl_path=settings.qa_human_filtered_jsonl,
        metadata_paths=(settings.video_metadata_human_json,),
        out_dir=base_out,
        file_slug="av_human",
    )
    return {"AV_Human": m_human}


def _run_pass_inference(
    *,
    settings: Settings,
    pass_label: str,
    table3_note: str,
    samples: list[MCQSample],
    qa_jsonl_path: str,
    metadata_paths: tuple[str, ...],
    out_dir: Path,
    file_slug: str,
    extra_metrics: dict | None = None,
) -> dict:
    t0 = time.perf_counter()
    run_logs: list[str] = []

    def _log(msg: str) -> None:
        print(msg)
        run_logs.append(msg)

    _log(f"[{pass_label}] Start | note={table3_note} | samples={len(samples)}")
    _log(f"[{pass_label}] QA={qa_jsonl_path}")
    _log(f"[{pass_label}] Metadata={', '.join(metadata_paths)}")
    _log(f"[{pass_label}] HF_URI={settings.hf_video_dataset_uri}")
    _log(f"[{pass_label}] Model={settings.gemini_model}")

    client = GeminiClient(settings)
    preds: list[str] = []
    correct: list[str] = []
    per_task_gold: dict[str, list[str]] = {}
    per_task_pred: dict[str, list[str]] = {}
    rows: list[dict] = []
    media_ready_count = 0
    media_missing_count = 0
    media_kind_counts: Counter[str] = Counter()

    safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", file_slug).strip("_") or "pass"
    pred_path = out_dir / f"vanilla_predictions_{safe_slug}.jsonl"
    metrics_path = out_dir / f"vanilla_metrics_{safe_slug}.json"

    pbar = _sample_pbar(len(samples), desc=f"Gemini vanilla {pass_label}")
    for sample in iter_samples(samples):
        client.set_run_context(
            pass_label=pass_label,
            sample_id=str(sample.sample_id),
            task_code=sample.task_code or "UNKNOWN",
            task_type=sample.task_type,
        )
        raw_pred = ""
        pred = ""
        status = "ok"
        error = ""
        media_kind = "none"
        has_video = client.has_usable_media(sample.video_input)
        if has_video:
            media_ready_count += 1
            media_kind = "usable_video_input"
            media_kind_counts[media_kind] += 1
        else:
            media_missing_count += 1
            status = "error"
            error = "video_input unavailable; pass --no-prefetch-videos disables baseline media QA."

        try:
            if has_video:
                prompt = _build_vanilla_prompt(sample)
                pred, raw_text = client.generate_answer_letter(
                    prompt,
                    media_input=sample.video_input,
                    stage="vanilla_answer",
                    extraction_mode="answer_is",
                    max_output_tokens=settings.max_output_tokens_vanilla_answer,
                    format_retry_attempts=settings.format_retry_attempts,
                    max_repair_attempts=settings.max_repair_attempts,
                )
                raw_pred = raw_text
                if pred in {"A", "B", "C", "D"}:
                    preds.append(pred)
                    correct.append(sample.answer)
                    task_code = sample.task_code or "UNKNOWN"
                    per_task_gold.setdefault(task_code, []).append(sample.answer)
                    per_task_pred.setdefault(task_code, []).append(pred)
                else:
                    status = "error"
                    error = "parse_error: no valid [ANSWER] letter extracted"
        except Exception as exc:
            status = "error"
            error = str(exc)
        finally:
            rows.append(
                {
                    "pass": pass_label,
                    "sample_id": sample.sample_id,
                    "task_type": sample.task_type,
                    "task_code": sample.task_code,
                    "gold": sample.answer,
                    "pred": pred,
                    "raw_pred": raw_pred,
                    "status": status,
                    "error": error,
                    "has_video_input": has_video,
                    "video_media_kind": media_kind,
                }
            )
            pbar.update(1)
    pbar.close()

    _log(
        f"[{pass_label}] Video availability: ready={media_ready_count}/{len(samples)}, "
        f"missing={media_missing_count}/{len(samples)}"
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    metrics = {
        "pass": pass_label,
        "table3_note": table3_note,
        "qa_jsonl": qa_jsonl_path,
        "metadata_json": list(metadata_paths),
        "accuracy": accuracy(correct, preds),
        "task_accuracy": {
            code: accuracy(per_task_gold.get(code, []), per_task_pred.get(code, []))
            for code in ["AIE", "ACC", "AEL", "AVCM", "AVOM", "AVTM"]
        },
        "task_counts_scored": {
            code: len(per_task_gold.get(code, []))
            for code in ["AIE", "ACC", "AEL", "AVCM", "AVOM", "AVTM"]
        },
        "n_samples_total": len(samples),
        "n_samples_scored": len(correct),
        "n_samples_failed": len(samples) - len(correct),
        "video_ready_count": media_ready_count,
        "video_missing_count": media_missing_count,
        "video_media_kinds": dict(media_kind_counts),
        "gemini_model_configured": settings.gemini_model,
        "gemini_model_effective": client.active_model,
        "pipeline_latency_ms": elapsed_ms,
        "pipeline_latency_s": round(elapsed_ms / 1000.0, 3),
        "run_logs": run_logs,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    _write_jsonl(pred_path, rows)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log(
        f"[{pass_label}] Done | scored={len(correct)}/{len(samples)} "
        f"| acc={metrics['accuracy']:.4f} | latency_ms={elapsed_ms} "
        f"| outputs={pred_path.name},{metrics_path.name}"
    )
    metrics["run_logs"] = run_logs
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    divider = {
        "_run_divider": "============================================================",
        "run_started_unix_ms": int(time.time() * 1000),
        "file": path.name,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(divider, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_task_distribution(label: str, samples: list[MCQSample]) -> None:
    codes = ["AIE", "ACC", "AEL", "AVCM", "AVOM", "AVTM", "UNKNOWN"]
    counts = Counter((s.task_code or "UNKNOWN") for s in samples)
    pieces = [f"{code}={counts.get(code, 0)}" for code in codes]
    print(f"{label} n={len(samples)} | " + " | ".join(pieces))
