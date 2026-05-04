from __future__ import annotations

from .dataset import MCQSample

# Budget for optional chain-of-thought before the anchored "Answer is X" line (vanilla + idea2 final step).
FINAL_MCQ_ANSWER_MAX_OUTPUT_TOKENS = 50


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


def final_mcq_answer_format_prompt() -> str:
    """Shared final-step instructions: optional CoT, anchored answer line, shared token budget."""
    n = FINAL_MCQ_ANSWER_MAX_OUTPUT_TOKENS
    return (
        "Output format:\n"
        "You may output chain-of-thought reasoning (open-ended, ideally 1-3 short sentences), but your response MUST "
        "contain the exact substring 'Answer is ' immediately followed by a single uppercase option letter, "
        "with no other letters between—one of A, B, C, or D.\n"
        "Critical constraints:\n"
        "- Do NOT restate the answer choices (do not write lines like 'A.' / 'B.' / 'C.' / 'D.').\n"
        "- Do NOT mention standalone option letters anywhere else in your response.\n"
        "- The final line must be exactly: Answer is X  (where X is A/B/C/D)\n"
        "Correct examples:\n"
        "1) Answer is A\n"
        "2) Based on the evidence, Answer is C\n"
        "3) I considered the options and the best match is clear. Answer is D\n"
        "4) Short rationale here. Answer is B\n"
        "5) Final decision: Answer is A\n"
        "Invalid examples:\n"
        "1) A\n"
        "2) The answer is B\n"
        "3) Answer: C\n"
        "4) Answer is E\n"
        "5) No explicit final substring\n"
        f"Stay within ~{n} output tokens.\n"
        "Wrong example (invalid): omitting \"Answer is X\" entirely."
    )


def build_fixed_mcq_prompt(sample: MCQSample, context_block: str) -> str:
    """Compose final reasoner prompt in order: context -> question -> answer format."""
    return (
        f"{_context_block_prompt(context_block)}\n"
        f"{_fixed_question_prompt(sample)}\n"
        # f"{_reasoning_strategy_prompt()}\n"
        f"{final_mcq_answer_format_prompt()}"
    )


def vanilla_mcq_answer_preamble_prompt(sample: MCQSample) -> str:
    """Video+audio preamble + MCQ stems; pair with ``final_mcq_answer_format_prompt()`` for the full vanilla prompt."""
    return (
        "Answer the multiple-choice question using the attached video clip.\n"
        "Use everything available in that clip—including both visuals and synchronized audio "
        "(speech, music, ambient sound)—not picture-only reasoning.\n\n"
        f"{_fixed_question_prompt(sample)}"
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


