"""Run the Phase A core-hardening experiments (additive to E0-E10).

Usage:  python experiments/run_phase_a.py
Writes: results/phase_a/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cetspace.phase_a import run_phase_a  # noqa: E402


def main() -> None:
    index = run_phase_a(ROOT)
    print(json.dumps(index, ensure_ascii=False, indent=2))
    print("\n" + (ROOT / "results" / "phase_a" / "phase_a_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
