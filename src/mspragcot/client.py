from __future__ import annotations

import json
import hashlib
import mimetypes
import re
import subprocess
import time
import uuid
from pathlib import Path

from google import genai
from google.genai import types

from src.config import Settings
from src.eval.metrics import extract_final_answer_letter


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        """Args: settings with API key/model. Returns: None."""
        self._settings = settings
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.timeout_ms),
        )
        self._active_model = settings.gemini_model
        self._metrics_path = Path(settings.output_dir) / "gemini_call_metrics_detailed.jsonl"
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_context: dict[str, object] = {}
        self._media_cache_dir = Path(settings.output_dir) / "gemini_media_cache"
        self._media_cache_dir.mkdir(parents=True, exist_ok=True)

    def _generation_config(
        self,
        *,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
    ) -> types.GenerateContentConfig:
        tb = thinking_budget if thinking_budget is not None else self._settings.thinking_budget
        thinking = types.ThinkingConfig(thinking_budget=tb) if tb >= 0 else None
        return types.GenerateContentConfig(
            temperature=self._settings.temperature,
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else self._settings.max_output_tokens
            ),
            system_instruction=self._settings.system_instruction,
            thinking_config=thinking,
        )

    @property
    def active_model(self) -> str:
        return self._active_model

    def set_run_context(self, **context: object) -> None:
        """Attach context fields included on every subsequent call metric event."""
        self._run_context = dict(context)

    def _task_code(self) -> str:
        tc = str(self._run_context.get("task_code") or "").upper().strip()
        return tc

    def _is_alignment_task(self) -> bool:
        return self._task_code() in {"AVOM", "AVSM", "AEL"}

    def _media_kind(self, media_input: object | None) -> str:
        if media_input is None:
            return "none"
        if isinstance(media_input, types.Part):
            return "genai_part"
        hf_encoded = getattr(media_input, "_hf_encoded", None)
        if isinstance(hf_encoded, dict):
            if isinstance(hf_encoded.get("bytes"), (bytes, bytearray)):
                return "hf_dict_bytes"
            if isinstance(hf_encoded.get("path"), str):
                return "hf_dict_path"
        if isinstance(media_input, dict):
            if isinstance(media_input.get("bytes"), (bytes, bytearray)):
                return "dict_bytes"
            if isinstance(media_input.get("path"), str):
                return "dict_path"
            return "dict_other"
        if isinstance(media_input, str):
            if media_input.startswith("hf://"):
                return "hf_uri"
            if "://" in media_input:
                return "uri"
            return "file_path"
        p_attr = getattr(media_input, "path", None)
        if isinstance(p_attr, str):
            if p_attr.startswith("hf://"):
                return "obj_hf_path"
            if "://" in p_attr:
                return "obj_uri_path"
            return "obj_file_path"
        return type(media_input).__name__

    def _append_metric(self, event: str, data: dict[str, object]) -> None:
        row = {
            "event_id": f"ev_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
            "ts_unix_ms": int(time.time() * 1000),
            "event": event,
            "model": self._settings.gemini_model,
            "timeout_ms": self._settings.timeout_ms,
            "temperature": self._settings.temperature,
            "max_output_tokens": self._settings.max_output_tokens,
            "run_context": self._run_context,
            **data,
        }
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _guess_mime(self, path: str, fallback: str = "application/octet-stream") -> str:
        mime, _ = mimetypes.guess_type(path)
        return mime or fallback

    def _hf_to_https(self, hf_path: str, mode: str = "resolve") -> str | None:
        """Convert hf://datasets/<repo>@<rev>/<path> to https URL."""
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

    def _sampled_video_path(self, src: str) -> str | None:
        """Create a low-fps, lower-resolution cached MP4 for faster Gemini uploads."""
        is_alignment = self._is_alignment_task()
        fps = (
            self._settings.video_sample_fps_alignment
            if is_alignment
            else self._settings.video_sample_fps
        )
        if fps <= 0:
            return src
        key = hashlib.sha1(
            (
                f"{src}|fps={fps}|w={self._settings.video_max_width}|"
                f"crf={self._settings.video_crf}|task={self._task_code()}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        out_path = self._media_cache_dir / f"sampled_{key}.mp4"
        if out_path.exists():
            return str(out_path)
        # Keep aspect ratio, force even dimensions for encoder compatibility.
        vf = (
            f"fps={fps},"
            f"scale='min(iw,{self._settings.video_max_width})':-2:flags=bicubic"
        )
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            src,
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(self._settings.video_crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return None
        return str(out_path) if out_path.exists() else None

    def _part_from_uri_with_sampling(self, uri: str, mime_hint: str) -> types.Part:
        sampled = self._sampled_video_path(uri)
        if sampled:
            p = Path(sampled)
            return types.Part.from_bytes(
                data=p.read_bytes(),
                mime_type=self._guess_mime(str(p), "video/mp4"),
            )
        return types.Part.from_uri(
            file_uri=uri,
            mime_type=self._guess_mime(mime_hint, "video/mp4"),
        )

    def _process_video_bytes(self, raw: bytes, _name_hint: str = "") -> types.Part:
        """Save raw video bytes to disk, run through ffmpeg preprocessing, return Part."""
        key = hashlib.sha1(raw[:4096]).hexdigest()[:16]
        tmp = self._media_cache_dir / f"hfraw_{key}.mp4"
        if not tmp.exists():
            tmp.write_bytes(raw)
        sampled = self._sampled_video_path(str(tmp))
        if sampled:
            sp = Path(sampled)
            if sp.exists():
                return types.Part.from_bytes(
                    data=sp.read_bytes(),
                    mime_type=self._guess_mime(str(sp), "video/mp4"),
                )
        return types.Part.from_bytes(data=raw, mime_type="video/mp4")

    def _as_media_part(self, media_input: object) -> object | None:
        """Normalize local-path / dict media to a Gemini Part.

        HF video features may arrive as dicts like {"path": "...", "bytes": ...}. Passing those
        raw can trigger SDK validation errors ("file uri and mime_type are required"), so we
        convert to bytes-backed Part explicitly.
        """
        if isinstance(media_input, types.Part):
            return media_input

        hf_encoded = getattr(media_input, "_hf_encoded", None)
        if isinstance(hf_encoded, dict):
            hfp = hf_encoded.get("path")
            hfb = hf_encoded.get("bytes")
            if isinstance(hfb, (bytes, bytearray)):
                return self._process_video_bytes(bytes(hfb), _name_hint=str(hfp))
            if isinstance(hfp, str):
                if hfp.startswith("hf://datasets/"):
                    resolved = self._hf_to_https(hfp, mode="resolve")
                    if resolved:
                        return self._part_from_uri_with_sampling(resolved, hfp)
                if "://" in hfp:
                    return self._part_from_uri_with_sampling(hfp, hfp)

        if isinstance(media_input, dict):
            b = media_input.get("bytes")
            p = media_input.get("path")
            if isinstance(b, (bytes, bytearray)):
                return self._process_video_bytes(bytes(b), _name_hint=str(p))
            if isinstance(p, str):
                if p.startswith("hf://datasets/"):
                    resolved = self._hf_to_https(p, mode="resolve")
                    if resolved:
                        return self._part_from_uri_with_sampling(resolved, p)
                if "://" in p:
                    return self._part_from_uri_with_sampling(p, p)
                path = Path(p)
                if path.exists():
                    sampled = self._sampled_video_path(str(path)) or str(path)
                    sp = Path(sampled)
                    return types.Part.from_bytes(
                        data=sp.read_bytes(),
                        mime_type=self._guess_mime(str(sp), fallback="video/mp4"),
                    )

        if isinstance(media_input, str):
            if "://" in media_input:
                return self._part_from_uri_with_sampling(media_input, media_input)
            path = Path(media_input)
            if path.exists():
                sampled = self._sampled_video_path(str(path)) or str(path)
                sp = Path(sampled)
                return types.Part.from_bytes(
                    data=sp.read_bytes(),
                    mime_type=self._guess_mime(str(sp), fallback="video/mp4"),
                )

        p_attr = getattr(media_input, "path", None)
        if isinstance(p_attr, str):
            if "://" in p_attr:
                return self._part_from_uri_with_sampling(p_attr, p_attr)
            path = Path(p_attr)
            if path.exists():
                sampled = self._sampled_video_path(str(path)) or str(path)
                sp = Path(sampled)
                return types.Part.from_bytes(
                    data=sp.read_bytes(),
                    mime_type=self._guess_mime(str(sp), fallback="video/mp4"),
                )

        return None

    def has_usable_media(self, media_input: object | None) -> bool:
        """Whether media can be normalized into a Gemini-supported Part."""
        if media_input is None:
            return False
        return self._as_media_part(media_input) is not None

    @staticmethod
    def _extract_single_letter(text: str) -> str:
        m = re.match(r"^\s*([ABCD])\s*$", text or "", flags=re.IGNORECASE)
        return m.group(1).upper() if m else ""

    @staticmethod
    def _fallback_first_option(text: str) -> str:
        t = (text or "").upper()
        for ch in t:
            if ch in {"A", "B", "C", "D"}:
                return ch
        return ""

    def generate(
        self,
        prompt: str,
        media_input: object | None = None,
        *,
        stage: str = "unspecified",
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        """Args: prompt text and optional single-modality media input. Returns: model text output."""
        t0 = time.perf_counter()
        call_id = f"call_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        eff_max_out = (
            max_output_tokens
            if max_output_tokens is not None
            else self._settings.max_output_tokens
        )
        self._append_metric(
            "call_start",
            {
                "call_id": call_id,
                "stage": stage,
                "prompt_chars": len(prompt),
                "has_media_input": media_input is not None,
                "media_kind": self._media_kind(media_input),
                "max_output_tokens_effective": eff_max_out,
                "thinking_budget_effective": thinking_budget if thinking_budget is not None else self._settings.thinking_budget,
            },
        )
        contents: list[object] = [prompt]
        if media_input is not None:
            part = self._as_media_part(media_input)
            if part is None:
                self._append_metric(
                    "call_error",
                    {
                        "call_id": call_id,
                        "stage": stage,
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        "error_type": "ValueError",
                        "error": "Unsupported media_input format; expected bytes/path/uri with mime.",
                        "failure_phase": "normalize_media",
                    },
                )
                raise ValueError("Unsupported media_input format; expected bytes/path/uri with mime.")
            contents = [part, prompt]
        self._append_metric(
            "request_dispatch",
            {
                "call_id": call_id,
                "stage": stage,
                "content_parts": len(contents),
                "has_media_part": len(contents) > 1,
            },
        )
        # Exponential backoff for transient API failures, capped at 64s wait interval.
        backoff_schedule_s = [1, 2, 4, 8, 16, 32, 64]
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._client.models.generate_content(
                    model=self._settings.gemini_model,
                    contents=contents,
                    config=self._generation_config(
                        max_output_tokens=max_output_tokens,
                        thinking_budget=thinking_budget,
                    ),
                )
                break
            except Exception as exc:
                err = str(exc).upper()
                retryable = any(
                    token in err
                    for token in (
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "RESOURCE_EXHAUSTED",
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                        "TIMEOUT",
                        "TIMED OUT",
                        "READ OPERATION TIMED OUT",
                    )
                )
                if retryable and attempt <= len(backoff_schedule_s):
                    wait_s = backoff_schedule_s[attempt - 1]
                    self._append_metric(
                        "call_retry",
                        {
                            "call_id": call_id,
                            "stage": stage,
                            "attempt": attempt,
                            "retry_wait_s": wait_s,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    time.sleep(wait_s)
                    continue
                self._append_metric(
                    "call_error",
                    {
                        "call_id": call_id,
                        "stage": stage,
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        "attempts": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failure_phase": "generate_content",
                    },
                )
                raise
        self._active_model = self._settings.gemini_model
        text = getattr(response, "text", None)
        self._append_metric(
            "call_success",
            {
                "call_id": call_id,
                "stage": stage,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                "attempts": attempt,
                "response_text_chars": len(text) if isinstance(text, str) else 0,
                "response_has_text": bool(text),
            },
        )
        return text.strip() if text else ""

    def generate_answer_letter(
        self,
        prompt: str,
        media_input: object | None = None,
        *,
        stage: str = "answer_letter",
        max_repair_attempts: int = 3,
        extraction_mode: str = "single_letter",
        max_output_tokens: int | None = None,
        format_retry_attempts: int = 3,
        thinking_budget: int | None = None,
    ) -> tuple[str, str]:
        """Generate final MCQ letter with retries.

        extraction_mode ``single_letter``: reply must be one letter; repair fixes format.
        extraction_mode ``answer_is``: reply must contain substring ``Answer is X``.

        Token caps are resolved from ``max_output_tokens`` (caller must supply via
        settings.max_output_tokens_idea2_answer or settings.max_output_tokens_vanilla_answer)
        or falls back to the global settings.max_output_tokens.

        Returns:
            (letter, raw_text_for_logging)
        """
        if extraction_mode == "answer_is":
            cap = max_output_tokens
            raw = ""
            for i in range(format_retry_attempts):
                st = stage if i == 0 else f"{stage}_format_retry_{i+1}"
                raw = self.generate(
                    prompt,
                    media_input=media_input,
                    stage=st,
                    max_output_tokens=cap,
                    thinking_budget=thinking_budget,
                )
                letter = extract_final_answer_letter(raw)
                if letter:
                    return letter, raw

            last_text = raw
            for i in range(max_repair_attempts):
                repair_prompt = (
                    "Your previous reply did not contain a valid substring of the form "
                    "'Answer is X' where X is exactly one uppercase letter among A, B, C, D.\n"
                    "Write a short reply that MUST include that exact substring (e.g. \"Answer is B\").\n\n"
                    f"Previous reply:\n{last_text}"
                )
                repaired = self.generate(
                    repair_prompt,
                    media_input=None,
                    stage=f"{stage}_repair_{i+1}",
                    max_output_tokens=cap,
                )
                last_text = repaired
                letter = extract_final_answer_letter(repaired)
                if letter:
                    return letter, repaired

            # IMPORTANT: avoid biased fallbacks (responses often contain many 'A/B/C/D' tokens).
            return "", last_text

        raw = self.generate(
            prompt,
            media_input=media_input,
            stage=stage,
            max_output_tokens=max_output_tokens,
        )
        letter = self._extract_single_letter(raw)
        if letter:
            return letter, raw

        last_text = raw
        for i in range(max_repair_attempts):
            repair_prompt = (
                "Convert the following model response into exactly one uppercase option letter.\n"
                "Valid outputs: A, B, C, D.\n"
                "Output exactly one letter and nothing else.\n\n"
                f"Response:\n{last_text}"
            )
            repaired = self.generate(
                repair_prompt,
                media_input=None,
                stage=f"{stage}_repair_{i+1}",
                max_output_tokens=max_output_tokens,
            )
            last_text = repaired
            letter = self._extract_single_letter(repaired)
            if letter:
                return letter, repaired

        fallback = self._fallback_first_option(last_text) or self._fallback_first_option(raw)
        return fallback, last_text
