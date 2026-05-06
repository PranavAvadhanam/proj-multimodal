from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.avut.dataset import MCQSample
from src.config import Settings
from src.mspragcot.client import GeminiClient

_FULLNESS_PROMPT = (
    "\n\n[OUTPUT BUDGET]\n"
    "You have a token budget of {budget} tokens for this description. "
    "Use the full budget: be thorough and detailed. Include timestamps, "
    "specific entities, quantities, spatial relationships, and any nuance "
    "that could help answer the question. Do not leave room unused — a "
    "longer, richer description is better than a terse one.\n"
    "Output format requirements:\n"
    "- Write as many detailed bullet points as the budget allows.\n"
    "- Each bullet must be a complete sentence.\n"
    "- Do not end with unfinished words, dangling punctuation, or broken quotes."
)


@dataclass
class ModalityDescriptions:
    text: str
    audio: str
    video: str


@dataclass(frozen=True)
class PerModalityBudget:
    """Per-modality token budgets derived from MIS weighting."""
    text: int
    audio: int
    visual: int


class ModalityDescriber:
    def __init__(
        self,
        client: GeminiClient,
        settings: Settings,
        per_modality_budget: Optional[PerModalityBudget] = None,
    ) -> None:
        self.client = client
        self._default_budget = settings.max_output_tokens_describe
        self._budget = per_modality_budget

    @property
    def text_budget(self) -> int:
        return self._budget.text if self._budget else self._default_budget

    @property
    def audio_budget(self) -> int:
        return self._budget.audio if self._budget else self._default_budget

    @property
    def visual_budget(self) -> int:
        return self._budget.visual if self._budget else self._default_budget

    def describe_text(self, sample: MCQSample) -> str:
        if not sample.transcript:
            raise ValueError("describe_text requires transcript text input.")
        return f"{sample.transcript}\n"

    def describe_audio(self, sample: MCQSample, prompt: str, text_reference: str) -> str:
        if sample.audio_input is None:
            raise ValueError(
                "describe_audio requires audio_input in sample (audio-only media object)."
            )
        if not text_reference:
            raise ValueError("describe_audio requires text_reference for cross-modal de-duplication.")
        budget = self.audio_budget
        audio_only_prompt = (
            f"{prompt}\n\n"
            "[TEXT_REFERENCE_ALREADY_COVERED]\n"
            f"{text_reference}\n\n"
            "Use only the provided audio input.\n"
            "Do not include any lexical/semantic transcript content (no quotes, paraphrases, or word-level meaning).\n"
            "Describe only non-lexical audio evidence: prosody, intonation, pacing, pauses, overlap/turn-taking, "
            "laughter, sighs, cries, music, SFX, loudness, pitch, and timing.\n"
            "Explicitly avoid repeating anything already covered in TEXT_REFERENCE_ALREADY_COVERED."
            + _FULLNESS_PROMPT.format(budget=budget)
        )
        return self.client.generate(
            audio_only_prompt,
            media_input=sample.audio_input,
            stage="describe_audio",
            max_output_tokens=budget,
        )

    def describe_video(self, sample: MCQSample, prompt: str, text_reference: str) -> str:
        if sample.video_input is None:
            raise ValueError(
                "describe_video requires video_input in sample (video media object)."
            )
        if not text_reference:
            raise ValueError("describe_video requires text_reference for cross-modal de-duplication.")
        budget = self.visual_budget
        video_only_prompt = (
            f"{prompt}\n\n"
            "[TEXT_REFERENCE_ALREADY_COVERED]\n"
            f"{text_reference}\n\n"
            "Use only the provided visual/video input for this description.\n"
            "Describe only visual evidence (entities, gestures, objects, scene/layout, on-screen text, "
            "actions, camera/shot changes, and timing).\n"
            "Do not restate transcript semantics or any information already present in TEXT_REFERENCE_ALREADY_COVERED."
            + _FULLNESS_PROMPT.format(budget=budget)
        )
        return self.client.generate(
            video_only_prompt,
            media_input=sample.video_input,
            stage="describe_video",
            max_output_tokens=budget,
        )
