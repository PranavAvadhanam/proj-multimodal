"""Run MIS calibration without thinking tokens (no CoT).

Writes to outputs/mis/ by default.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv()

# No-CoT variant: disable thinking tokens.
os.environ["GEMINI_THINKING_BUDGET"] = "0"
os.environ["GEMINI_THINKING_BUDGET_IDEA2"] = "0"

os.environ.setdefault("MIS_SUBDIR", "mis")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misprompt.run_mis import main

if __name__ == "__main__":
    main()
