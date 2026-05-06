"""Core MIS (Modality Importance Score) computation.

For modalities M = {text, audio, visual}, computes:
    MIS(mj) = perf(qi | M+_j) - perf(qi | M-_j)

where:
    M+_j = all non-empty subsets containing mj, excluding singleton {mj}
    M-_j = all non-empty subsets excluding mj

Performance is average correctness (1/0) across the relevant subsets.
Final MIS values are softmax-normalized to produce token-budget weights.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.avut.dataset import MCQSample
from src.avut.prompts import build_fixed_mcq_prompt, final_mcq_answer_format_prompt
from src.config import Settings
from src.eval.metrics import extract_final_answer_letter
from src.mspragcot.client import GeminiClient

MODALITIES = ("text", "audio", "visual")

ALL_NONEMPTY_SUBSETS: list[frozenset[str]] = []
for r in range(1, len(MODALITIES) + 1):
    for combo in itertools.combinations(MODALITIES, r):
        ALL_NONEMPTY_SUBSETS.append(frozenset(combo))


def _m_plus(mj: str) -> list[frozenset[str]]:
    """Non-empty subsets containing mj, excluding singleton {mj}."""
    return [s for s in ALL_NONEMPTY_SUBSETS if mj in s and len(s) > 1]


def _m_minus(mj: str) -> list[frozenset[str]]:
    """Non-empty subsets not containing mj."""
    return [s for s in ALL_NONEMPTY_SUBSETS if mj not in s]


def softmax(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=np.float64)
    shifted = arr - arr.max()
    exp_vals = np.exp(shifted)
    return (exp_vals / exp_vals.sum()).tolist()


def _build_subset_prompt(
    sample: MCQSample,
    subset: frozenset[str],
    descriptions: dict[str, str],
) -> str:
    """Build MCQ prompt using only the modality descriptions in `subset`."""
    parts = []
    if "text" in subset and descriptions.get("text"):
        parts.append(f"[TEXT]\n{descriptions['text']}")
    if "audio" in subset and descriptions.get("audio"):
        parts.append(f"[AUDIO]\n{descriptions['audio']}")
    if "visual" in subset and descriptions.get("visual"):
        parts.append(f"[VISUAL]\n{descriptions['visual']}")

    context_block = "\n\n".join(parts) + "\n" if parts else ""

    modality_names = sorted(subset)
    head = (
        f"You have been provided with descriptions from the following modalities: "
        f"{', '.join(modality_names)}.\n\n"
        f"{context_block}\n"
        "Use ONLY the provided descriptions as evidence to answer the question. "
        "Prefer the option with the strongest support from the available evidence.\n\n"
    )
    return (
        f"{head}"
        "Answer the following multiple-choice question.\n\n"
        f"Question: {sample.question}\n\n"
        f"Options:\n"
        f"(A) {sample.option_a}\n"
        f"(B) {sample.option_b}\n"
        f"(C) {sample.option_c}\n"
        f"(D) {sample.option_d}\n\n"
        f"{final_mcq_answer_format_prompt()}"
    )


def evaluate_subset(
    client: GeminiClient,
    sample: MCQSample,
    subset: frozenset[str],
    descriptions: dict[str, str],
    settings: Settings,
) -> tuple[bool, str, str]:
    """Run MCQ inference for a single (sample, modality_subset) pair.

    Returns: (is_correct, predicted_letter, raw_response)
    """
    prompt = _build_subset_prompt(sample, subset, descriptions)
    subset_label = "+".join(sorted(subset))

    pred_letter, raw_text = client.generate_answer_letter(
        prompt,
        media_input=None,
        stage=f"mis_ablation_{subset_label}",
        extraction_mode="answer_is",
        max_output_tokens=settings.max_output_tokens_idea2_answer,
        format_retry_attempts=settings.format_retry_attempts,
        max_repair_attempts=settings.max_repair_attempts,
        thinking_budget=0,
    )
    is_correct = pred_letter == sample.answer
    return is_correct, pred_letter, raw_text


def compute_mis_scores(
    results: dict[str, dict[frozenset[str], list[bool]]],
) -> dict[str, float]:
    """Compute MIS for each modality from per-sample, per-subset correctness results.

    Args:
        results: {sample_id: {subset: [correct_bool]}} — one entry per sample per subset.
                 In practice each list has exactly one element.

    Returns:
        {modality: MIS_score} where MIS = avg_perf(M+) - avg_perf(M-)
    """
    mis_scores: dict[str, float] = {}

    for mj in MODALITIES:
        m_plus_subsets = _m_plus(mj)
        m_minus_subsets = _m_minus(mj)

        per_sample_mis: list[float] = []
        for sample_id, subset_results in results.items():
            plus_correct = []
            for subset in m_plus_subsets:
                entries = subset_results.get(subset, [])
                plus_correct.extend(entries)

            minus_correct = []
            for subset in m_minus_subsets:
                entries = subset_results.get(subset, [])
                minus_correct.extend(entries)

            if plus_correct and minus_correct:
                perf_plus = sum(plus_correct) / len(plus_correct)
                perf_minus = sum(minus_correct) / len(minus_correct)
                per_sample_mis.append(perf_plus - perf_minus)

        mis_scores[mj] = sum(per_sample_mis) / len(per_sample_mis) if per_sample_mis else 0.0

    return mis_scores


def mis_to_token_allocation(
    mis_scores: dict[str, float],
    total_budget: int = 768,
) -> dict[str, int]:
    """Convert raw MIS scores to per-modality token allocations via softmax."""
    ordered = [mis_scores.get(m, 0.0) for m in MODALITIES]
    weights = softmax(ordered)

    raw_allocations = [w * total_budget for w in weights]
    allocations = [int(round(a)) for a in raw_allocations]

    diff = total_budget - sum(allocations)
    if diff != 0:
        idx = int(np.argmax(raw_allocations))
        allocations[idx] += diff

    return {
        MODALITIES[i]: allocations[i]
        for i in range(len(MODALITIES))
    }


def run_mis_evaluation(
    samples: list[MCQSample],
    descriptions_by_sample: dict[str, dict[str, str]],
    settings: Settings,
    output_dir: Path,
) -> dict[str, Any]:
    """Full MIS pipeline: ablate all subsets for all samples, compute scores, allocate tokens.

    Args:
        samples: MIS evaluation samples (must be separate from idea2 test set).
        descriptions_by_sample: {sample_id: {"text": ..., "audio": ..., "visual": ...}}
        settings: Project settings.
        output_dir: Where to write MIS artifacts.

    Returns:
        Dict with raw scores, weights, and token allocations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiClient(settings)

    results: dict[str, dict[frozenset[str], list[bool]]] = defaultdict(dict)
    detail_rows: list[dict] = []

    total_evals = len(samples) * len(ALL_NONEMPTY_SUBSETS)
    pbar = tqdm(total=total_evals, desc="MIS ablation", unit="eval", colour="magenta")

    for sample in samples:
        sid = str(sample.sample_id)
        descs = descriptions_by_sample.get(sid, {})
        client.set_run_context(
            pass_label="MIS",
            sample_id=sid,
            task_code=sample.task_code or "UNKNOWN",
            task_type=sample.task_type,
        )

        for subset in ALL_NONEMPTY_SUBSETS:
            try:
                is_correct, pred, raw = evaluate_subset(
                    client, sample, subset, descs, settings
                )
            except Exception as exc:
                is_correct, pred, raw = False, "", f"ERROR: {exc}"

            results[sid][subset] = [is_correct]
            detail_rows.append({
                "sample_id": sid,
                "task_code": sample.task_code,
                "subset": "+".join(sorted(subset)),
                "gold": sample.answer,
                "pred": pred,
                "correct": is_correct,
            })
            pbar.update(1)

    pbar.close()

    mis_scores = compute_mis_scores(results)
    weights = softmax([mis_scores[m] for m in MODALITIES])
    token_alloc = mis_to_token_allocation(mis_scores, total_budget=768)

    output = {
        "n_samples": len(samples),
        "n_subsets": len(ALL_NONEMPTY_SUBSETS),
        "total_evaluations": total_evals,
        "mis_sample_ids": [str(s.sample_id) for s in samples],
        "raw_mis_scores": mis_scores,
        "softmax_weights": {MODALITIES[i]: weights[i] for i in range(len(MODALITIES))},
        "token_allocation_budget": 768,
        "token_allocation": token_alloc,
        "per_subset_accuracy": {},
    }

    for subset in ALL_NONEMPTY_SUBSETS:
        key = "+".join(sorted(subset))
        all_correct = [
            results[sid][subset][0]
            for sid in results
            if subset in results[sid]
        ]
        output["per_subset_accuracy"][key] = (
            sum(all_correct) / len(all_correct) if all_correct else 0.0
        )

    results_path = output_dir / "mis_results.json"
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    detail_path = output_dir / "mis_detail.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output
