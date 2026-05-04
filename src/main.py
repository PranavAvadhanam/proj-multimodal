from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import google.auth
import google.auth.transport.requests
from tqdm import tqdm

from src.avut.audio_extractor import (
    extract_audio_part_from_video_input,
    extract_audio_wav_bytes_from_video_input,
)
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
from src.avut.prompts import (
    audio_perception_prompt,
    build_fixed_mcq_prompt,
    FINAL_MCQ_ANSWER_MAX_OUTPUT_TOKENS,
    text_perception_prompt,
    video_perception_prompt,
)
from src.config import Settings, get_settings
from src.eval.metrics import accuracy
from src.mspragcot.client import GeminiClient
from src.mspragcot.modality_describer import ModalityDescriber
from src.mspragcot.reasoner import PragReasoner


class TranscriptionFailure(RuntimeError):
    """Fatal transcription error that should abort the current run."""


def _transcribe_with_stt_v2(audio_wav_bytes: bytes, project_id: str) -> str:
    """Transcribe WAV bytes using Google Cloud Speech-to-Text v2 REST API."""
    if not project_id:
        raise TranscriptionFailure(
            "Missing GOOGLE_CLOUD_PROJECT for Speech-to-Text v2 recognizer path."
        )

    # Support either OAuth (service-account ADC) or API-key auth from env.
    # Preferred explicit env var is GOOGLE_STT_API_KEY; as a convenience fallback, if
    # GOOGLE_APPLICATION_CREDENTIALS contains an API-key-like value (not a JSON path),
    # treat it as an STT API key.
    stt_api_key = (os.environ.get("GOOGLE_STT_API_KEY") or "").strip()
    if not stt_api_key:
        gac = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        if gac and not gac.endswith(".json") and gac.startswith("AIza"):
            stt_api_key = gac

    endpoint = (
        "https://speech.googleapis.com/v2/projects/"
        f"{project_id}/locations/global/recognizers/_:recognize"
    )
    if stt_api_key:
        endpoint = f"{endpoint}?key={stt_api_key}"

    payload = {
        "config": {
            "autoDecodingConfig": {},
            "languageCodes": ["en-US"],
            "model": "long",
        },
        "content": base64.b64encode(audio_wav_bytes).decode("ascii"),
    }
    headers = {"Content-Type": "application/json"}
    if not stt_api_key:
        try:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except Exception as exc:
            raise TranscriptionFailure(
                "Failed to load Google application default credentials. "
                "Set GOOGLE_APPLICATION_CREDENTIALS to a valid service-account JSON, "
                "or set GOOGLE_STT_API_KEY for API-key auth."
            ) from exc

        if not credentials.valid:
            try:
                credentials.refresh(google.auth.transport.requests.Request())
            except Exception as exc:
                raise TranscriptionFailure(
                    "Failed to refresh Google OAuth token for Speech-to-Text v2."
                ) from exc
        access_token = credentials.token
        if not access_token:
            raise TranscriptionFailure("Google OAuth token is empty for Speech-to-Text v2 call.")
        headers["Authorization"] = f"Bearer {access_token}"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TranscriptionFailure(f"Speech-to-Text v2 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise TranscriptionFailure(f"Speech-to-Text v2 request failed: {exc}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TranscriptionFailure(f"Speech-to-Text v2 returned invalid JSON: {body[:300]}") from exc

    results = parsed.get("results") or []
    parts: list[str] = []
    for r in results:
        alts = r.get("alternatives") or []
        if not alts:
            continue
        t = (alts[0].get("transcript") or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


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


def _gemini_sample_pbar(total: int, desc: str) -> tqdm:
    """Single tqdm for one pass over samples (each step = full Gemini pipeline for that row)."""
    try:
        return tqdm(total=total, desc=desc, unit="sample", dynamic_ncols=True, colour="green")
    except TypeError:
        return tqdm(total=total, desc=desc, unit="sample", dynamic_ncols=True)


def _prepare_modalities_for_sample(
    sample: MCQSample,
    client: GeminiClient,
    audio_cache_dir: Path,
) -> dict[str, bool]:
    """Prepare transcript/audio inputs before describe_* calls.

    - If transcript is missing and video is present, request lexical transcription from
      Speech-to-Text v2.
    - If audio_input is missing but video_input exists, reuse video media object as audio source.
      (Gemini can still reason over the audio channel from container media.)
    """
    if sample.video_input is None and sample.video_path:
        sample.video_input = sample.video_path

    video_ready = client.has_usable_media(sample.video_input)
    text_ready = bool(sample.transcript)
    audio_ready = sample.audio_input is not None

    prep_notes: list[str] = []
    if sample.video_input is None and not sample.video_path:
        prep_notes.append("no_video_source")
    elif not video_ready:
        prep_notes.append("video_unusable")

    if not text_ready and video_ready:
        try:
            audio_wav = extract_audio_wav_bytes_from_video_input(
                video_input=sample.video_input,
                sample_id=str(sample.sample_id),
                cache_dir=audio_cache_dir,
            )
            if not audio_wav:
                raise TranscriptionFailure(
                    f"Failed to extract audio for transcription (sample_id={sample.sample_id})."
                )
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
            transcript = _transcribe_with_stt_v2(
                audio_wav_bytes=audio_wav,
                project_id=project_id or "",
            )
            if transcript:
                sample.transcript = transcript
                text_ready = True
            else:
                prep_notes.append("transcription_empty")
        except TranscriptionFailure:
            text_ready = False
            prep_notes.append("transcription_failed")
            raise
        except Exception as exc:
            text_ready = False
            prep_notes.append("transcription_failed")
            raise TranscriptionFailure(
                "Transcription failed for "
                f"sample_id={sample.sample_id} task_code={sample.task_code or 'UNKNOWN'}: {exc}"
            ) from exc

    if not audio_ready and video_ready:
        audio_part = extract_audio_part_from_video_input(
            video_input=sample.video_input,
            sample_id=str(sample.sample_id),
            cache_dir=audio_cache_dir,
        )
        if audio_part is not None:
            sample.audio_input = audio_part
            audio_ready = True
        else:
            prep_notes.append("audio_extract_failed")

    return {
        "text": text_ready,
        "audio": audio_ready,
        "video": video_ready,
        "prep_notes": ";".join(prep_notes),
    }


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


def run_idea2_pipeline(
    input_jsonl: str | None = None,
    output_dir: str | None = None,
    max_samples: int | None = None,
    run_sample: str | None = None,
    prefetch_videos: int | None = None,
    no_prefetch_videos: bool = False,
    split_max_samples: bool = False,
) -> dict:
    """Run Idea 2 pipeline.

    Default: AV-Human only. Optional split mode runs both AV-Human and AV-Gemini with
    --max-samples interpreted as a total budget split between passes.

    Args:
        input_jsonl: Single-pass QA JSONL override.
        max_samples: Per-pass QA row cap (task-balanced).
        run_sample: Optional AV-Human sample_id for one-row basic run.
        prefetch_videos: Max distinct QA videos to resolve from HF (one prefetch bar). None = all
            unique QA_ids needed for the selected passes.
        no_prefetch_videos: Skip Hugging Face streaming entirely (fast for Gemini-only debugging).
        split_max_samples: If True, run both AV-Human and AV-Gemini and split max_samples across
            the two passes (requires max_samples). If False (default), run AV-Human only.
    """
    settings = get_settings()
    base_out = Path(output_dir or settings.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    if run_sample is not None:
        if input_jsonl is not None:
            raise ValueError("--run-sample cannot be used with --input.")
        print(
            f"[Basic mode] --run-sample={run_sample} loads one AV-Human QA row and one "
            "matching HF video for a minimal end-to-end run.\n"
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
        qa_ids = [s.sample_id for s in samples]
        if no_prefetch_videos:
            print("[Prefetch] Skipped (--no-prefetch-videos).\n")
            vmap = {}
        else:
            cap = prefetch_videos if prefetch_videos is not None else len(set(qa_ids))
            print(
                f"[Prefetch] Resolving up to {cap} distinct HF video(s) for {len(samples)} sample row(s) "
                f"(streaming, one bar).\n"
            )
            use_gemini = "gemini" in Path(input_jsonl).name.lower()
            oh, og = prefetch_hf_avut_train_videos(
                None if use_gemini else qa_ids,
                qa_ids if use_gemini else None,
                settings.hf_video_dataset_uri,
                max_videos=prefetch_videos,
                desc="Prefetch videos (HF)",
                human_expected_video_by_id=(
                    {str(s.sample_id): s.video_path or "" for s in samples} if not use_gemini else None
                ),
                gemini_expected_video_by_id=(
                    {str(s.sample_id): s.video_path or "" for s in samples} if use_gemini else None
                ),
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
        # Total max_samples budget split across AV-Human and AV-Gemini.
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
        qh = [str(s.sample_id) for s in samples_human]
        qg = [str(s.sample_id) for s in samples_gemini]
        if no_prefetch_videos:
            print("[Prefetch] Skipped (--no-prefetch-videos).\n")
            vmap_h, vmap_g = {}, {}
        else:
            nh, ng = len(set(qh)), len(set(qg))
            cap_note = (
                f" (each side capped to {prefetch_videos} id(s))"
                if prefetch_videos is not None
                else ""
            )
            print(
                f"[Prefetch] AV-Human {nh} distinct id(s), AV-Gemini {ng} distinct id(s); "
                f"one HF stream / one bar{cap_note}.\n"
            )
            vmap_h, vmap_g = prefetch_hf_avut_train_videos(
                qh,
                qg,
                settings.hf_video_dataset_uri,
                max_videos=prefetch_videos,
                desc="Prefetch videos (HF)",
                human_expected_video_by_id={str(s.sample_id): s.video_path or "" for s in samples_human},
                gemini_expected_video_by_id={str(s.sample_id): s.video_path or "" for s in samples_gemini},
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
    qh = [str(s.sample_id) for s in samples_human]
    if no_prefetch_videos:
        print("[Prefetch] Skipped (--no-prefetch-videos).\n")
        vmap_h = {}
    else:
        nh = len(set(qh))
        cap_note = f" (capped to {prefetch_videos} id(s))" if prefetch_videos is not None else ""
        print(f"[Prefetch] AV-Human {nh} distinct id(s); one HF stream / one bar{cap_note}.\n")
        vmap_h, _ = prefetch_hf_avut_train_videos(
            qh,
            None,
            settings.hf_video_dataset_uri,
            max_videos=prefetch_videos,
            desc="Prefetch videos (HF)",
            human_expected_video_by_id={str(s.sample_id): s.video_path or "" for s in samples_human},
            gemini_expected_video_by_id=None,
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
    """Run Gemini pipeline over prepared samples (videos already attached)."""
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
    describer = ModalityDescriber(client)
    reasoner = PragReasoner(client)

    preds: list[str] = []
    correct: list[str] = []
    per_task_gold: dict[str, list[str]] = {}
    per_task_pred: dict[str, list[str]] = {}
    rows: list[dict] = []
    transcript_rows: list[dict] = []
    modality_samples = {"text": 0, "audio": 0, "video": 0}
    modality_combo_counts: Counter[str] = Counter()

    safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", file_slug).strip("_") or "pass"
    pred_path = out_dir / f"idea2_predictions_{safe_slug}.jsonl"
    metrics_path = out_dir / f"idea2_metrics_{safe_slug}.json"
    transcript_path = out_dir / "transcript.py"
    audio_cache_dir = out_dir / "audio_cache"

    pbar = _gemini_sample_pbar(len(samples), desc=f"Gemini {pass_label}")
    for sample in iter_samples(samples):
        client.set_run_context(
            pass_label=pass_label,
            sample_id=str(sample.sample_id),
            task_code=sample.task_code or "UNKNOWN",
            task_type=sample.task_type,
        )
        used_text = False
        used_audio = False
        used_video = False
        prep_notes = ""
        text_desc = ""
        audio_desc = ""
        video_desc = ""
        context_block = ""
        reasoning_cot = ""
        try:
            prepared = _prepare_modalities_for_sample(sample, client, audio_cache_dir)
            used_text = prepared["text"]
            used_audio = prepared["audio"]
            used_video = prepared["video"]
            prep_notes = prepared.get("prep_notes", "")
            if used_text:
                modality_samples["text"] += 1
            if used_audio:
                modality_samples["audio"] += 1
            if used_video:
                modality_samples["video"] += 1
            combo_bits = []
            if used_text:
                combo_bits.append("text")
            if used_audio:
                combo_bits.append("audio")
            if used_video:
                combo_bits.append("video")
            combo_key = "+".join(combo_bits) if combo_bits else "none"
            modality_combo_counts[combo_key] += 1
            transcript_rows.append(
                {
                    "pass": pass_label,
                    "sample_id": sample.sample_id,
                    "task_type": sample.task_type,
                    "task_code": sample.task_code,
                    "transcript": sample.transcript or "",
                }
            )

            t_prompt = text_perception_prompt(sample)
            a_prompt = audio_perception_prompt(sample)
            v_prompt = video_perception_prompt(sample)
            text_desc = describer.describe_text(sample, t_prompt) if used_text else "Text unavailable."
            audio_desc = (
                describer.describe_audio(sample, a_prompt, text_desc)
                if used_audio
                else "Audio unavailable."
            )
            video_desc = (
                describer.describe_video(sample, v_prompt, text_desc)
                if used_video
                else "Video unavailable."
            )

            # Decode stage removed: feed modality-specific descriptions directly to reasoner.
            context_block = (
                f"[TEXT]\n{text_desc}\n\n"
                f"[AUDIO]\n{audio_desc}\n\n"
                f"[VISUAL]\n{video_desc}\n"
            )

            final_prompt = build_fixed_mcq_prompt(sample, context_block=context_block)
            pred, reasoning_cot = client.generate_answer_letter(
                final_prompt,
                media_input=None,
                stage="reason_and_answer",
                extraction_mode="answer_is",
                max_output_tokens=FINAL_MCQ_ANSWER_MAX_OUTPUT_TOKENS,
                format_retry_attempts=3,
            )
            raw_pred = reasoning_cot

            preds.append(pred)
            correct.append(sample.answer)
            task_code = sample.task_code or "UNKNOWN"
            per_task_gold.setdefault(task_code, []).append(sample.answer)
            per_task_pred.setdefault(task_code, []).append(pred)
            rows.append(
                {
                    "pass": pass_label,
                    "sample_id": sample.sample_id,
                    "task_type": sample.task_type,
                    "task_code": sample.task_code,
                    "gold": sample.answer,
                    "pred": pred,
                    "raw_pred": raw_pred,
                    "status": "ok",
                    "used_text": used_text,
                    "used_audio": used_audio,
                    "used_video": used_video,
                    "prep_notes": prep_notes,
                    "text_desc": text_desc,
                    "audio_desc": audio_desc,
                    "video_desc": video_desc,
                    "reasoning_cot": reasoning_cot,
                }
            )
        except TranscriptionFailure as exc:
            rows.append(
                {
                    "pass": pass_label,
                    "sample_id": sample.sample_id,
                    "task_type": sample.task_type,
                    "task_code": sample.task_code,
                    "gold": sample.answer,
                    "pred": "",
                    "raw_pred": "",
                    "status": "fatal_error",
                    "error": str(exc),
                    "used_text": used_text,
                    "used_audio": used_audio,
                    "used_video": used_video,
                    "prep_notes": prep_notes,
                    "text_desc": text_desc,
                    "audio_desc": audio_desc,
                    "video_desc": video_desc,
                    "reasoning_cot": reasoning_cot,
                }
            )
            raise
        except Exception as exc:
            rows.append(
                {
                    "pass": pass_label,
                    "sample_id": sample.sample_id,
                    "task_type": sample.task_type,
                    "task_code": sample.task_code,
                    "gold": sample.answer,
                    "pred": "",
                    "raw_pred": "",
                    "status": "error",
                    "error": str(exc),
                    "used_text": used_text,
                    "used_audio": used_audio,
                    "used_video": used_video,
                    "prep_notes": prep_notes,
                    "text_desc": text_desc,
                    "audio_desc": audio_desc,
                    "video_desc": video_desc,
                    "reasoning_cot": reasoning_cot,
                }
            )
        finally:
            pbar.update(1)
    pbar.close()
    total = len(samples)
    if total > 0:
        combo_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(modality_combo_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        _log(
            f"[{pass_label}] Modality usage: "
            f"text={modality_samples['text']}/{total} ({100.0*modality_samples['text']/total:.1f}%), "
            f"audio={modality_samples['audio']}/{total} ({100.0*modality_samples['audio']/total:.1f}%), "
            f"video={modality_samples['video']}/{total} ({100.0*modality_samples['video']/total:.1f}%)"
        )
        _log(f"[{pass_label}] Modality combos: {combo_summary}")

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
        "modality_usage_counts": dict(modality_samples),
        "modality_usage_combos": dict(modality_combo_counts),
        "gemini_model_configured": settings.gemini_model,
        "gemini_model_effective": client.active_model,
        "pipeline_latency_ms": elapsed_ms,
        "pipeline_latency_s": round(elapsed_ms / 1000.0, 3),
        "run_logs": run_logs,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    _write_jsonl(pred_path, rows)
    _append_transcripts_py(transcript_path, transcript_rows, pass_label)
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
    """Write list of dict rows to JSONL.

    Args:
        path: Destination file path.
        rows: Records to serialize, one JSON object per line.

    Returns:
        None.
    """
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_transcripts_py(path: Path, rows: list[dict], pass_label: str) -> None:
    """Append transcript records for this pass into a Python-readable file."""
    if not rows:
        return
    if not path.exists():
        path.write_text("TRANSCRIPTS = []\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"# === {pass_label} ===\n")
        f.write("TRANSCRIPTS.extend([\n")
        for row in rows:
            f.write(f"    {repr(row)},\n")
        f.write("])\n\n")


def _print_task_distribution(label: str, samples: list[MCQSample]) -> None:
    """Print task-code distribution summary for quick representativeness checks."""
    codes = ["AIE", "ACC", "AEL", "AVCM", "AVOM", "AVTM", "UNKNOWN"]
    counts = Counter((s.task_code or "UNKNOWN") for s in samples)
    pieces = [f"{code}={counts.get(code, 0)}" for code in codes]
    print(f"{label} n={len(samples)} | " + " | ".join(pieces))
