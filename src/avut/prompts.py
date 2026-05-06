from __future__ import annotations

from .dataset import MCQSample


def _format_options(sample: MCQSample) -> str:
    return (
        f"(A) {sample.option_a}\n"
        f"(B) {sample.option_b}\n"
        f"(C) {sample.option_c}\n"
        f"(D) {sample.option_d}"
    )


def final_mcq_answer_format_prompt() -> str:
    return (
        "Reply with EXACTLY this format and nothing else:\n"
        "[ANSWER] <letter>\n\n"
        "where <letter> is one of A, B, C, D.\n\n"
        "Example response: [ANSWER] C"
    )


def vanilla_mcq_answer_preamble_prompt(sample: MCQSample) -> str:
    return (
        "You are given a video with audio. Watch and listen carefully, "
        "then answer the following question.\n\n"
        f"Question: {sample.question}\n\n"
        f"Options:\n{_format_options(sample)}\n\n"
        "Think step by step, then give your answer."
    )


def build_fixed_mcq_prompt(
    sample: MCQSample, context_block: str, *, is_idea2: bool = False
) -> str:
    context = context_block.strip()
    if is_idea2 and context:
        head = (
            "You have been provided with three expert descriptions of the same "
            "short video, each covering a different modality:\n\n"
            f"{context}\n\n"
            "Use ALL three descriptions as evidence. For each answer option, "
            "identify which specific details from [TEXT], [AUDIO], and [VISUAL] "
            "support or contradict it. Pay special attention to timestamps, "
            "counts, spatial relationships, and any cross-modal correspondences "
            "(e.g. what is visually on screen when a particular sound occurs). "
            "Prefer the option with the strongest multi-modal support.\n\n"
        )
    elif context:
        head = (
            "You are given a video with audio. A transcript of the spoken "
            "content is also provided below.\n\n"
            f'Transcript:\n"{context}"\n\n'
        )
    else:
        head = ""
    return (
        f"{head}"
        "Answer the following multiple-choice question.\n\n"
        f"Question: {sample.question}\n\n"
        f"Options:\n{_format_options(sample)}\n\n"
        f"{final_mcq_answer_format_prompt()}"
    )


def text_perception_prompt(sample: MCQSample) -> str:
    transcript = sample.transcript or "N/A"
    return (
        f"Question: {sample.question}\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Based on the transcript, describe the key details relevant to answering the question."
    )


def audio_perception_prompt(sample: MCQSample) -> str:
    return (
        f"Question: {sample.question}\n\n"
        "Describe each distinct sound event you hear, noting approximately when "
        "it occurs in the clip (e.g. 'at the start', 'around 0:15', 'near the end'). "
        "Include: speaker tone, speech rate, emotion, background sounds, music, "
        "sound effects, and what object or action likely produces each sound."
    )


def video_perception_prompt(sample: MCQSample) -> str:
    return (
        f"Question: {sample.question}\n\n"
        "Describe what you see in the video, noting approximately when key events "
        "occur (e.g. 'at 0:05', 'midway through', 'at the end'). "
        "Include: people, actions, objects, on-screen text, scene layout, "
        "and any visual details relevant to the question."
    )
