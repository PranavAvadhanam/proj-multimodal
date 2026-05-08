"""Load MIS-derived token budgets for use by the modality describer.

If MIS has been run, returns per-modality token allocations from the saved results.
Allocations are scaled so text+audio+visual sum to ``3 * fallback_per_modality`` (matching
``run_mis.py`` when ``--total-token-budget`` is omitted). Uniform fallback uses
``fallback_per_modality`` per modality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModalityTokenBudget:
    text: int
    audio: int
    visual: int

    @property
    def total(self) -> int:
        return self.text + self.audio + self.visual


def _rescale_modality_budget(b: ModalityTokenBudget, target_total: int) -> ModalityTokenBudget:
    """Integer per-modality totals that sum exactly to ``target_total`` (preserve ratios)."""
    cur = b.total
    if cur == target_total:
        return b
    if cur <= 0:
        if target_total <= 0:
            return ModalityTokenBudget(text=0, audio=0, visual=0)
        base = target_total // 3
        rem = target_total % 3
        t = base + (1 if rem >= 1 else 0)
        au = base + (1 if rem >= 2 else 0)
        vi = base
        return ModalityTokenBudget(text=t, audio=au, visual=vi)
    raw = [b.text * target_total / cur, b.audio * target_total / cur, b.visual * target_total / cur]
    alloc = [int(round(x)) for x in raw]
    diff = target_total - sum(alloc)
    if diff != 0:
        idx = max(range(3), key=lambda i: raw[i])
        alloc[idx] += diff
    return ModalityTokenBudget(text=alloc[0], audio=alloc[1], visual=alloc[2])


def load_token_budget(
    output_dir: str | Path,
    fallback_per_modality: int = 256,
    mis_subdir: str = "mis",
) -> ModalityTokenBudget:
    """Load MIS-computed token allocation if available, else uniform fallback.

    Args:
        output_dir: Base output directory (e.g. ``outputs``).
        fallback_per_modality: Uniform per-modality describe cap when no MIS file exists;
            total target for MIS path is ``3 * fallback_per_modality``, and saved files
            from older runs are rescaled to match that total while preserving ratios.
        mis_subdir: Subdirectory under ``output_dir`` containing MIS results.
            ``"mis"`` for no-CoT, ``"mis_cot"`` for CoT.
    """
    base = Path(output_dir)
    alloc_path = base / mis_subdir / "token_allocation.json"
    target_total = 3 * fallback_per_modality

    if alloc_path.exists():
        try:
            data = json.loads(alloc_path.read_text(encoding="utf-8"))
            alloc = data.get("allocation", {})
            bud = ModalityTokenBudget(
                text=int(alloc.get("text", fallback_per_modality)),
                audio=int(alloc.get("audio", fallback_per_modality)),
                visual=int(alloc.get("visual", fallback_per_modality)),
            )
            if bud.total != target_total:
                return _rescale_modality_budget(bud, target_total)
            return bud
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return ModalityTokenBudget(
        text=fallback_per_modality,
        audio=fallback_per_modality,
        visual=fallback_per_modality,
    )
