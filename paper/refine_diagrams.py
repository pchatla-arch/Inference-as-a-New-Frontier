"""Compatibility entry point for source-grounded systems diagrams.

Run after ``generate_authentic_figures.py``.  The general generator creates the
full figure set; this script replaces manuscript Figures 7--9 with the refined,
publication-ready versions in ``recreate_figures_7_9.py``.
"""
from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("recreate_figures_7_9.py")
if not SCRIPT.exists():
    raise FileNotFoundError(f"Missing figure recreation script: {SCRIPT}")
runpy.run_path(str(SCRIPT), run_name="__main__")
