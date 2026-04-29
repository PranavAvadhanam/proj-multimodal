from __future__ import annotations

from .dataset import MCQSample


def _context_block_prompt(context_block: str) -> str:
    return f"Context:\n{context_block}\n"


def _fixed_question_prompt(sample: MCQSample) -> str:
    return (
        f"Question: {sample.question}\n"
        f"A. {sample.option_a}\n"
        f"B. {sample.option_b}\n"
        f"C. {sample.option_c}\n"
        f"D. {sample.option_d}\n"
    )


def _reasoning_strategy_prompt() -> str:
    return (
        "Select the best answer to the multiple-choice question using the provided context.\n"
        "Use the context to reason internally, but do not output your reasoning."
    )


def _single_letter_answer_prompt() -> str:
    return (
        "Output format requirements:\n"
        "Return exactly one uppercase letter: A, B, C, or D.\n"
        "Do not output any other text."
    )


def build_fixed_mcq_prompt(sample: MCQSample, context_block: str) -> str:
    """Compose final reasoner prompt in order: context -> question -> CoT strategy -> answer format."""
    return (
        f"{_context_block_prompt(context_block)}\n"
        f"{_fixed_question_prompt(sample)}\n"
        # f"{_reasoning_strategy_prompt()}\n"
        f"{_single_letter_answer_prompt()}"
    )


def text_perception_prompt(sample: MCQSample) -> str:
    return (
        "Perceive textual cues only (other modalities masked).\n"
        "Use up to ~256 tokens and maximize information density.\n"
        "Include as many distinct, question-relevant details as possible.\n"
        "Typical textual features to extract: entities/participants, actions/events, relations, attributes/adjectives, quantities/comparatives, negation/uncertainty/modality, discourse structure (cause-contrast-condition), sentiment/stance, intent/goals, and key quoted phrases.\n"
        "Summarize literal meaning and salient lexical cues relevant to the question.\n"
        f"Transcript:\n{sample.transcript or 'N/A'}"
    )


def audio_perception_prompt(sample: MCQSample) -> str:
    return (
        "Perceive audio cues only (other modalities masked).\n"
        "Use up to ~256 tokens and maximize information density.\n"
        "Include as many distinct, question-relevant details as possible.\n"
        "Describe only non-lexical audio evidence relevant to the question.\n"
        "Do NOT include lexical/transcript meaning, quotes, paraphrases, or word-level semantic content.\n"
        "Typical audio features to extract: prosody/intonation contours, speaking rate/rhythm, pauses/silences, overlap/turn-taking, speaker affect/emotion, loudness and pitch dynamics, voice quality/timbre, background ambiance/noise, music characteristics (presence/style/energy), and notable non-speech sound classes.\n"
        f"Audio path hint:\n{sample.audio_path or 'N/A'}"
    )


def video_perception_prompt(sample: MCQSample) -> str:
    return (
        "Perceive visual cues only (other modalities masked).\n"
        "Use up to ~256 tokens and maximize information density.\n"
        "Include as many distinct, question-relevant details as possible.\n"
        "Typical visual features to extract: scene/layout, entities and attributes (appearance/pose), objects/tools/materials, actions/interactions, motion/dynamics, spatial relations, salience/focus changes, camera/viewpoint cues (shot scale/angle/transitions), and lighting/color/style context.\n"
        "Describe entities, actions, and scene context relevant to the question.\n"
        f"Video path hint:\n{sample.video_path or 'N/A'}"
    )


