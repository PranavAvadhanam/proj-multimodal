from __future__ import annotations

import re

_ANSWER_IS_PATTERN = re.compile(r"Answer\s+is\s*:?\s*([ABCD])\b", re.IGNORECASE)
_FINAL_ANSWER_PATTERN = re.compile(r"FINAL[_\s-]*ANSWER\s*:?\s*([ABCD])\b", re.IGNORECASE)


def extract_answer_is_letter(pred: str) -> str:
    """Parse MCQ letter from substring 'Answer is X' where X ∈ {A,B,C,D}; uses last valid match."""
    if not pred:
        return ""
    matches = list(_ANSWER_IS_PATTERN.finditer(pred))
    if not matches:
        return ""
    return matches[-1].group(1).upper()


def extract_final_answer_letter(pred: str) -> str:
    """Parse a final MCQ letter from anchored patterns (Answer is X, FINAL_ANSWER: X)."""
    if not pred:
        return ""
    ans = extract_answer_is_letter(pred)
    if ans:
        return ans
    matches = list(_FINAL_ANSWER_PATTERN.finditer(pred))
    if not matches:
        return ""
    return matches[-1].group(1).upper()


def normalize_option(pred: str) -> str:
    pred = (pred or "").strip().upper()
    if pred in {"A", "B", "C", "D"}:
        return pred
    for ch in pred:
        if ch in {"A", "B", "C", "D"}:
            return ch
    return ""


def accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return correct / len(gold)
