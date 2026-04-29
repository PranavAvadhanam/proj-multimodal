from __future__ import annotations

from dataclasses import dataclass

from src.avut.dataset import MCQSample
from src.mspragcot.client import GeminiClient


@dataclass
class ModalityDescriptions:
    text: str
    audio: str
    video: str


class ModalityDescriber:
    def __init__(self, client: GeminiClient) -> None:
        """Args: GeminiClient instance. Returns: None."""
        self.client = client

    def describe_text(self, sample: MCQSample, prompt: str) -> str:
        """Args: AVUT sample + text-perception prompt. Returns: text modality description."""
        if not sample.transcript:
            raise ValueError("describe_text requires transcript text input.")
        text_only_prompt = (
            f"{prompt}\n\n"
            "[TEXT_ONLY_INPUT]\n"
            f"{sample.transcript}\n\n"
            "Use only this text input. Do not infer from audio or visuals.\n"
            "Output format requirements:\n"
            "- Provide 4-5 bullet points.\n"
            "- Each bullet must be a complete sentence.\n"
            "- Do not end with unfinished words, dangling punctuation, or broken quotes."
        )
        return self.client.generate(text_only_prompt, stage="describe_text")

    def describe_audio(self, sample: MCQSample, prompt: str, text_reference: str) -> str:
        """Args: AVUT sample + audio-perception prompt. Returns: audio modality description."""
        if sample.audio_input is None:
            raise ValueError(
                "describe_audio requires audio_input in sample (audio-only media object)."
            )
        if not text_reference:
            raise ValueError("describe_audio requires text_reference for cross-modal de-duplication.")
        audio_only_prompt = (
            f"{prompt}\n\n"
            "[TEXT_REFERENCE_ALREADY_COVERED]\n"
            f"{text_reference}\n\n"
            "Use only the provided audio input.\n"
            "Do not include any lexical/semantic transcript content (no quotes, paraphrases, or word-level meaning).\n"
            "Describe only non-lexical audio evidence: prosody, intonation, pacing, pauses, overlap/turn-taking, laughter, sighs, cries, music, SFX, loudness, pitch, and timing.\n"
            "Explicitly avoid repeating anything already covered in TEXT_REFERENCE_ALREADY_COVERED.\n"
            "Output format requirements:\n"
            "- Provide 4-5 bullet points.\n"
            "- Each bullet must be a complete sentence.\n"
            "- Do not end with unfinished words, dangling punctuation, or broken quotes."
        )
        return self.client.generate(
            audio_only_prompt,
            media_input=sample.audio_input,
            stage="describe_audio",
        )

    def describe_video(self, sample: MCQSample, prompt: str, text_reference: str) -> str:
        """Args: AVUT sample + video-perception prompt. Returns: visual modality description."""
        if sample.video_input is None:
            raise ValueError(
                "describe_video requires video_input in sample (video media object)."
            )
        if not text_reference:
            raise ValueError("describe_video requires text_reference for cross-modal de-duplication.")
        video_only_prompt = (
            f"{prompt}\n\n"
            "[TEXT_REFERENCE_ALREADY_COVERED]\n"
            f"{text_reference}\n\n"
            "Use only the provided visual/video input for this description.\n"
            "Describe only visual evidence (entities, gestures, objects, scene/layout, on-screen text, actions, camera/shot changes, and timing).\n"
            "Do not restate transcript semantics or any information already present in TEXT_REFERENCE_ALREADY_COVERED.\n"
            "Output format requirements:\n"
            "- Provide 4-5 bullet points.\n"
            "- Each bullet must be a complete sentence.\n"
            "- Do not end with unfinished words, dangling punctuation, or broken quotes."
        )
        return self.client.generate(
            video_only_prompt,
            media_input=sample.video_input,
            stage="describe_video",
        )
