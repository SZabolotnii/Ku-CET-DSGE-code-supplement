#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cetspace.experiments import run_all_experiments


if __name__ == "__main__":
    index = run_all_experiments(ROOT)
    print(f"Wrote experiment index: {ROOT / 'results' / 'experiment_index.json'}")
    print(f"Experiments: {', '.join(index['summary_metrics'].keys())}")
