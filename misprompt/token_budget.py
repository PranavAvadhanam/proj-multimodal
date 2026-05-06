"""Load MIS-derived token budgets for use by the modality describer.

If MIS has been run, returns per-modality token allocations from the saved results.
Otherwise falls back to uniform allocation from settings.max_output_tokens_describe.
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


def load_token_budget(output_dir: str | Path, fallback_per_modality: int = 256) -> ModalityTokenBudget:
    """Load MIS-computed token allocation if available, else uniform fallback.

    Searches for outputs/mis/token_allocation.json.
    """
    base = Path(output_dir)
    alloc_path = base / "mis" / "token_allocation.json"

    if alloc_path.exists():
        try:
            data = json.loads(alloc_path.read_text(encoding="utf-8"))
            alloc = data.get("allocation", {})
            return ModalityTokenBudget(
                text=int(alloc.get("text", fallback_per_modality)),
                audio=int(alloc.get("audio", fallback_per_modality)),
                visual=int(alloc.get("visual", fallback_per_modality)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return ModalityTokenBudget(
        text=fallback_per_modality,
        audio=fallback_per_modality,
        visual=fallback_per_modality,
    )
