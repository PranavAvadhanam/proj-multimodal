from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


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
    video_sample_fps: float
    video_sample_fps_alignment: float
    video_max_width: int
    video_crf: int
    temperature: float
    timeout_ms: int
    system_instruction: str
    output_dir: str
    # Token budgets — every Gemini call resolves its cap from one of these.
    max_output_tokens: int               # global fallback
    max_output_tokens_describe: int      # baseline per modality when MIS is off; MIS sums to 3× this across text+audio+video
    max_output_tokens_idea2_answer: int  # idea2 final MCQ answer step
    max_output_tokens_vanilla_answer: int  # vanilla single-call MCQ answer
    thinking_budget: int                 # Gemini thinking tokens; 0 = disabled, -1 = dynamic
    thinking_budget_idea2: int           # thinking tokens for idea2 reasoning step only
    # Retry / robustness knobs
    format_retry_attempts: int           # answer-format retries before falling back to repair
    max_repair_attempts: int             # repair-prompt retries after format retries exhausted
    max_audio_duration_seconds: int      # cap on extracted audio fed to Speech-to-Text
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
        "You are solving multimodal multiple-choice QA on short videos. "
        "Be factual and concise. Follow prompt-specific format instructions exactly."
    )
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=DEFAULT_GEMINI_MODEL,
        video_sample_fps=float(os.getenv("VIDEO_SAMPLE_FPS", "2")),
        video_sample_fps_alignment=float(os.getenv("VIDEO_SAMPLE_FPS_ALIGNMENT", "3")),
        video_max_width=int(os.getenv("VIDEO_MAX_WIDTH", "640")),
        video_crf=int(os.getenv("VIDEO_CRF", "20")),
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
        timeout_ms=int(os.getenv("GEMINI_TIMEOUT_MS", "90000")),
        max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "256")),
        # Per-mod describe cap (no MIS). With MIS calibration, allocations sum to 3× this env value.
        max_output_tokens_describe=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS_DESCRIBE", "1024")),
        # sometimes takes time before it converges on "Answer is X", 
        # doesn't always say the "answer is X" right away, even if wasn't instructed to do CoT
        max_output_tokens_idea2_answer=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS_IDEA2_ANSWER", "256")), 
        max_output_tokens_vanilla_answer=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS_VANILLA_ANSWER", "256")),
        thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "0")),
        thinking_budget_idea2=int(os.getenv("GEMINI_THINKING_BUDGET_IDEA2", "0")),
        format_retry_attempts=int(os.getenv("FORMAT_RETRY_ATTEMPTS", "3")),
        max_repair_attempts=int(os.getenv("MAX_REPAIR_ATTEMPTS", "3")),
        max_audio_duration_seconds=int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "60")),
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
