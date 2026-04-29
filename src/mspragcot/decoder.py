from __future__ import annotations

from dataclasses import dataclass

from src.mspragcot.client import GeminiClient


@dataclass(frozen=True)
class DecoderStateSpace:
    # From sketch: "Audio: irritated, amused, ..."
    audio_states: tuple[str, ...] = (
        "irritated",
        "amused",
        "angry",
        "neutral",
        "sarcastic",
        "frustrated",
        "dismissive",
        "enthusiastic",
        "sad",
        "surprised",
        "deadpan",
        "confused",
    )
    # From sketch: "Image: facial action coding / facial AUs"
    image_states: tuple[str, ...] = (
        "au_inner_brow_raise",
        "au_brow_lower",
        "au_cheek_raise",
        "au_lid_tighten",
        "au_lip_corner_pull",
        "au_lip_corner_depress",
        "au_jaw_drop",
        "au_none",
    )
    # From sketch: "Text: irony, metaphor, ..."
    text_states: tuple[str, ...] = (
        "irony",
        "metaphor",
        "literal",
        "hyperbole",
        "understatement",
        "rhetorical_question",
    )

    def as_instruction_block(self) -> str:
        """Args: none. Returns: decoder instruction text listing state spaces."""
        return (
            "Classify each modality into one or more classes from these state spaces.\n"
            f"Audio states: {', '.join(self.audio_states)}\n"
            f"Image states: {', '.join(self.image_states)}\n"
            f"Text states: {', '.join(self.text_states)}\n"
            "Return a compact structured output with modality labels and a short decoded summary."
        )


class PragDecoder:
    def __init__(self, client: GeminiClient) -> None:
        """Args: GeminiClient instance. Returns: None."""
        self.client = client
        self.state_space = DecoderStateSpace()
        
    def decode(self, text_desc: str, audio_desc: str, video_desc: str, prompt: str) -> str:
        """Args: modality descriptions + decode prompt. Returns: compact decoded summary string."""
        text = self.client.generate(prompt, stage="decode")
        return text
