"""Run MIS calibration with 768 thinking tokens (CoT).

Writes to outputs/mis_cot/ so CoT and no-CoT calibrations are stored separately.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv()

# CoT variant: enable 768 thinking tokens.
os.environ["GEMINI_THINKING_BUDGET"] = "768"
os.environ["GEMINI_THINKING_BUDGET_IDEA2"] = "768"

# Store CoT calibration in a separate subdirectory.
os.environ.setdefault("MIS_SUBDIR", "mis_cot")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misprompt.run_mis import main

if __name__ == "__main__":
    main()
