from __future__ import annotations

import re

_ANSWER_IS_PATTERN = re.compile(r"Answer\s+is\s*:?\s*([ABCD])\b", re.IGNORECASE)
_FINAL_ANSWER_PATTERN = re.compile(r"FINAL[_\s-]*ANSWER\s*:?\s*([ABCD])\b", re.IGNORECASE)
_BRACKET_ANSWER_PATTERN = re.compile(r"\[ANSWER\]\s*([A-Da-d])")
_COMPACT_ANSWER_PATTERN = re.compile(r"^\s*\(?([A-Da-d])\)?\s*[.,:\-]?\s*(\d{1,3})(?:\s|$|%)")
_ANSWER_IS_FALLBACK_PATTERN = re.compile(r"answer\s+is\s*[:\-]?\s*\(?([A-Da-d])\)?", re.IGNORECASE)
_PAREN_ANSWER_PATTERN = re.compile(r"\(([A-Da-d])\)")
_SINGLE_LETTER_PATTERN = re.compile(r"\b([A-D])\b")


def extract_answer_is_letter(pred: str) -> str:
    """Parse MCQ letter from substring 'Answer is X' where X ∈ {A,B,C,D}; uses last valid match."""
    if not pred:
        return ""
    matches = list(_ANSWER_IS_PATTERN.finditer(pred))
    if not matches:
        return ""
    return matches[-1].group(1).upper()


def extract_final_answer_letter(pred: str) -> str:
    """Parse final MCQ answer letter with strict-first and robust fallbacks."""
    if not pred:
        return ""
    m = _BRACKET_ANSWER_PATTERN.search(pred)
    if m:
        return m.group(1).upper()
    ans = extract_answer_is_letter(pred)
    if ans:
        return ans
    m2 = _COMPACT_ANSWER_PATTERN.match(pred)
    if m2:
        return m2.group(1).upper()
    m3 = _ANSWER_IS_FALLBACK_PATTERN.search(pred)
    if m3:
        return m3.group(1).upper()
    m4 = _PAREN_ANSWER_PATTERN.search(pred)
    if m4:
        return m4.group(1).upper()
    matches = list(_FINAL_ANSWER_PATTERN.finditer(pred))
    if matches:
        return matches[-1].group(1).upper()
    m5 = _SINGLE_LETTER_PATTERN.search(pred.upper())
    if m5:
        return m5.group(1).upper()
    return ""


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
