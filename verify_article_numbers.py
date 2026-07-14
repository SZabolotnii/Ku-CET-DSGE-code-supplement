#!/usr/bin/env python3
"""Verify that the headline numbers in the paper match the stored result logs.

Run:  python verify_article_numbers.py
Exits non-zero if any reported number drifts from the results/ artifacts.
The checks below mirror the manuscript exactly (Section 4).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

TOL = 0.005  # tolerance on AUROC / macro-F1 (results are 30-seed means)

checks: list[tuple[str, bool, str]] = []


def near(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def add(name: str, ok: bool, detail: str) -> None:
    checks.append((name, ok, detail))


# --- Section 4.1: leakage audit ------------------------------------------------
audit = json.loads((RES / "phase_a" / "a1_leakage_audit" / "metrics.json").read_text())
add(
    "4.1 rule baseline is the label generator (exact match = 1.0)",
    audit["rule_vs_label_exact_match_fraction"] == 1.0,
    f"exact_match={audit['rule_vs_label_exact_match_fraction']}",
)
add(
    "4.1 decisive-feature partition = 34 columns",
    audit["n_decision_columns"] == 34,
    f"n_decision_columns={audit['n_decision_columns']}",
)

# --- Section 4.2: de-leaked closed-set classification --------------------------
deleaked = json.loads((RES / "phase_a" / "a1_deleaked_multiclass" / "metrics.json").read_text())
add(
    "4.2 de-leaked GE macro-F1 = 0.666",
    near(deleaked["generating_element_macro_f1"], 0.6659),
    f"GE={deleaked['generating_element_macro_f1']:.4f}",
)
add(
    "4.2 de-leaked nearest-centroid macro-F1 = 0.522",
    near(deleaked["nearest_centroid_macro_f1"], 0.5223),
    f"NC={deleaked['nearest_centroid_macro_f1']:.4f}",
)
add(
    "4.2 de-leaked alpha_opt = 0.70 and degenerate 0.5 excluded",
    deleaked["best_alpha"] == 0.7 and deleaked["degenerate_alpha_is_candidate"] is False,
    f"best_alpha={deleaked['best_alpha']}, deg_candidate={deleaked['degenerate_alpha_is_candidate']}",
)

# --- Section 4.3: open-set detection (corrected v2, 30 seeds) -------------------
osv = json.loads((RES / "open_set_v2" / "open_set_v2.json").read_text())
lenses = osv["lenses"]
add("4.3 protocol = 30 seeds", osv["n_seeds"] == 30, f"n_seeds={osv['n_seeds']}")

l1 = lenses["L1_all_classes"]
add(
    "4.3 L1 full corpus GE = 0.832",
    near(l1["mean_auroc_ge"], 0.832),
    f"GE={l1['mean_auroc_ge']:.4f}",
)
add(
    "4.3 L1 full corpus NC = 0.658",
    near(l1["mean_auroc_nc"], 0.658),
    f"NC={l1['mean_auroc_nc']:.4f}",
)

l3 = lenses["L3_provenance_clean"]
add(
    "4.3 L3 provenance-clean GE = 0.795 (KEY)",
    near(l3["mean_auroc_ge"], 0.795),
    f"GE={l3['mean_auroc_ge']:.4f}",
)
add(
    "4.3 L3 provenance-clean NC = 0.611 (KEY)",
    near(l3["mean_auroc_nc"], 0.611),
    f"NC={l3['mean_auroc_nc']:.4f}",
)
ci = l3["diff_ci_classcluster"]
add(
    "4.3 L3 diff class-cluster 95% CI = [0.124, 0.246] and excludes 0",
    near(ci[0], 0.124, 0.01) and near(ci[1], 0.246, 0.01) and ci[0] > 0,
    f"CI={[round(c, 3) for c in ci]}",
)

# --- report --------------------------------------------------------------------
print("=" * 74)
print("Ku-CET-DSGE — article-number verification")
print("=" * 74)
all_ok = True
for name, ok, detail in checks:
    flag = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"[{flag}] {name}\n        {detail}")
print("=" * 74)
print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
