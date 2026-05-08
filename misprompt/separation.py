"""Enforce sample separation between MIS calibration and idea2/vanilla evaluation.

Provides utilities to load MIS exclusion lists and filter sample sets to prevent
data leakage between the MIS calibration split and the evaluation split.
"""

from __future__ import annotations

import json
from pathlib import Path

MIS_EXCLUSION_FILE = "mis_excluded_sample_ids.json"


def load_mis_exclusion_ids(output_dir: str | Path) -> set[str]:
    """Load sample IDs used by MIS calibration that must be excluded from evaluation.

    Checks all known MIS output subdirectories (mis/, mis_cot/) and the parent
    output directory. Returns the union of all exclusion IDs found, ensuring
    samples from *any* MIS calibration run (CoT or non-CoT) are excluded.
    Returns empty set if no exclusion file exists (MIS hasn't been run yet).
    """
    ids: set[str] = set()
    base = Path(output_dir)

    candidates = [
        base / "mis" / MIS_EXCLUSION_FILE,
        base / "mis_cot" / MIS_EXCLUSION_FILE,
        base / MIS_EXCLUSION_FILE,
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                ids.update(str(x) for x in data)
            except (json.JSONDecodeError, TypeError):
                pass

    return ids


def filter_excluded_samples(samples: list, excluded_ids: set[str]) -> list:
    """Remove samples whose sample_id is in the exclusion set."""
    if not excluded_ids:
        return samples
    filtered = [s for s in samples if str(s.sample_id) not in excluded_ids]
    n_removed = len(samples) - len(filtered)
    if n_removed > 0:
        print(f"  [Separation] Excluded {n_removed} MIS-calibration samples from evaluation set.")
    return filtered
