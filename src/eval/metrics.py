from __future__ import annotations


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
