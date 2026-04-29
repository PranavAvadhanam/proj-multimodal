from __future__ import annotations

import json
import mimetypes
import time
import uuid
from pathlib import Path

from google import genai
from google.genai import types

from src.config import Settings


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        """Args: settings with API key/model. Returns: None."""
        self._settings = settings
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.timeout_ms),
        )
        self._active_model = settings.gemini_model
        self._generation_config = types.GenerateContentConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            system_instruction=settings.system_instruction,
        )
        self._metrics_path = Path(settings.output_dir) / "gemini_call_metrics_detailed.jsonl"
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_context: dict[str, object] = {}

    @property
    def active_model(self) -> str:
        return self._active_model

    def set_run_context(self, **context: object) -> None:
        """Attach context fields included on every subsequent call metric event."""
        self._run_context = dict(context)

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
                mime = self._guess_mime(str(hfp), fallback="video/mp4")
                return types.Part.from_bytes(data=bytes(hfb), mime_type=mime)
            if isinstance(hfp, str):
                if hfp.startswith("hf://datasets/"):
                    resolved = self._hf_to_https(hfp, mode="resolve")
                    if resolved:
                        return types.Part.from_uri(
                            file_uri=resolved,
                            mime_type=self._guess_mime(hfp, "video/mp4"),
                        )
                if "://" in hfp:
                    return types.Part.from_uri(
                        file_uri=hfp,
                        mime_type=self._guess_mime(hfp, "video/mp4"),
                    )

        if isinstance(media_input, dict):
            b = media_input.get("bytes")
            p = media_input.get("path")
            if isinstance(b, (bytes, bytearray)):
                mime = self._guess_mime(str(p), fallback="video/mp4")
                return types.Part.from_bytes(data=bytes(b), mime_type=mime)
            if isinstance(p, str):
                if p.startswith("hf://datasets/"):
                    resolved = self._hf_to_https(p, mode="resolve")
                    if resolved:
                        return types.Part.from_uri(
                            file_uri=resolved,
                            mime_type=self._guess_mime(p, "video/mp4"),
                        )
                if "://" in p:
                    return types.Part.from_uri(file_uri=p, mime_type=self._guess_mime(p, "video/mp4"))
                path = Path(p)
                if path.exists():
                    return types.Part.from_bytes(
                        data=path.read_bytes(),
                        mime_type=self._guess_mime(str(path), fallback="video/mp4"),
                    )
                if path.is_absolute():
                    # Best-effort file URI for non-workspace absolute paths.
                    return types.Part.from_uri(
                        file_uri=f"file://{path}",
                        mime_type=self._guess_mime(str(path), "video/mp4"),
                    )

        if isinstance(media_input, str):
            if "://" in media_input:
                return types.Part.from_uri(
                    file_uri=media_input,
                    mime_type=self._guess_mime(media_input, "video/mp4"),
                )
            path = Path(media_input)
            if path.exists():
                return types.Part.from_bytes(
                    data=path.read_bytes(),
                    mime_type=self._guess_mime(str(path), fallback="video/mp4"),
                )
            if path.is_absolute():
                return types.Part.from_uri(
                    file_uri=f"file://{path}",
                    mime_type=self._guess_mime(str(path), "video/mp4"),
                )

        p_attr = getattr(media_input, "path", None)
        if isinstance(p_attr, str):
            if "://" in p_attr:
                return types.Part.from_uri(
                    file_uri=p_attr,
                    mime_type=self._guess_mime(p_attr, "video/mp4"),
                )
            path = Path(p_attr)
            if path.exists():
                return types.Part.from_bytes(
                    data=path.read_bytes(),
                    mime_type=self._guess_mime(str(path), fallback="video/mp4"),
                )
            if path.is_absolute():
                return types.Part.from_uri(
                    file_uri=f"file://{path}",
                    mime_type=self._guess_mime(str(path), "video/mp4"),
                )

        return None

    def has_usable_media(self, media_input: object | None) -> bool:
        """Whether media can be normalized into a Gemini-supported Part."""
        if media_input is None:
            return False
        return self._as_media_part(media_input) is not None

    def generate(
        self,
        prompt: str,
        media_input: object | None = None,
        *,
        stage: str = "unspecified",
    ) -> str:
        """Args: prompt text and optional single-modality media input. Returns: model text output."""
        t0 = time.perf_counter()
        call_id = f"call_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self._append_metric(
            "call_start",
            {
                "call_id": call_id,
                "stage": stage,
                "prompt_chars": len(prompt),
                "has_media_input": media_input is not None,
                "media_kind": self._media_kind(media_input),
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
        try:
            response = self._client.models.generate_content(
                model=self._settings.gemini_model,
                contents=contents,
                config=self._generation_config,
            )
        except Exception as exc:
            self._append_metric(
                "call_error",
                {
                    "call_id": call_id,
                    "stage": stage,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
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
                "response_text_chars": len(text) if isinstance(text, str) else 0,
                "response_has_text": bool(text),
            },
        )
        return text.strip() if text else ""
