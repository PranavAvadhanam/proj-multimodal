"""Modality Importance Score (MIS) computation module.

Empirically measures per-modality contribution by evaluating model performance
across all non-empty subsets of {text, audio, visual}, then normalizes via softmax
to produce token-budget weights for downstream description allocation.
"""
