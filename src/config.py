from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


def _normalize_google_credentials_env() -> None:
    """Normalize GOOGLE_APPLICATION_CREDENTIALS path for local .env usage.

    python-dotenv does not expand '~', but Google auth expects a real filesystem path.
    """
    raw = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not raw:
        return
    expanded = os.path.abspath(os.path.expanduser(raw))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = expanded


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    temperature: float
    max_output_tokens: int
    timeout_ms: int
    system_instruction: str
    output_dir: str
    # Table 3 (AVUT): left of "/" = AV-Human filtered; right = AV-Gemini filtered — separate QA + metadata per pass.
    qa_human_filtered_jsonl: str
    qa_gemini_filtered_jsonl: str
    video_metadata_human_json: str
    video_metadata_gemini_json: str
    hf_video_dataset_uri: str


def get_settings() -> Settings:
    load_dotenv()
    _normalize_google_credentials_env()
    default_system_instruction = (
        "You are solving multimodal multiple-choice QA on 1-2 minute videos. "
        "Be factual and use complete grammatical sentences with no unfinished fragments. "
        "Prioritize modality/stage-specific prompt requirements first, then follow general style guidance. "
        "When producing descriptive content, use the available output budget thoroughly and aim to use the 256-token window as fully as useful without padding. "
        "Do not stop early unless the response is complete and all required parts are covered."
    )
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=DEFAULT_GEMINI_MODEL,
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
        max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "256")),
        timeout_ms=int(os.getenv("GEMINI_TIMEOUT_MS", "90000")),
        system_instruction=os.getenv("GEMINI_SYSTEM_INSTRUCTION", default_system_instruction),
        output_dir=os.getenv("OUTPUT_DIR", "outputs"),
        qa_human_filtered_jsonl=os.getenv(
            "QA_HUMAN_FILTERED_JSONL", "data/avut/avut_human_filtered.jsonl"
        ),
        qa_gemini_filtered_jsonl=os.getenv(
            "QA_GEMINI_FILTERED_JSONL", "data/avut/avut_gemini_filtered.jsonl"
        ),
        video_metadata_human_json=os.getenv(
            "VIDEO_METADATA_HUMAN_JSON", "data/avut/hf_raw/AV_Human_filtered_data.json"
        ),
        video_metadata_gemini_json=os.getenv(
            "VIDEO_METADATA_GEMINI_JSON", "data/avut/hf_raw/AV_Gemini_filtered_data.json"
        ),
        hf_video_dataset_uri=os.getenv(
            "HF_VIDEO_DATASET_URI", "hf://tsinghua-ee/AVUTBenchmark@train"
        ),
    )
