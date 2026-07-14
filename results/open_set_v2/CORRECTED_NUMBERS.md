# B1 fix — corrected open-set numbers (single source of truth for the manuscript revision)

**Experiment:** `experiments/run_open_set_v2.py` → `results/open_set_v2/open_set_v2.json`.
Corrects `src/cetspace/phase_a.py::run_open_set` (B1: α tuned on the ROC negatives; centroid untuned).
**Corrected protocol:** 3-way KNOWN split train(60%)/val(20%)/test(20%); `select_alpha` on **val only**
(disjoint from the test negatives); GE fit on train; nearest-centroid fit on train (symmetric — neither
detector sees test/novel in fitting); ROC eval = test (neg) + held-out class (pos); **30 seeds**;
deterministic per-(seed,class) seeding; **DeLong** (validated against paired bootstrap) + bootstrap CIs.

## Bottom line
The open-set advantage is **REAL and survives the B1 fix**: corrected GE 0.832 vs centroid 0.658 is
essentially identical to the buggy 0.833 vs 0.665 → the α-peeking did **not** inflate it. It is robust
across 30 seeds with a class-cluster 95% CI that excludes 0 on three of four lenses.

## Four honest lenses (mean over 30 seeds)

| Lens | GE AUROC [fold-boot CI] | NC AUROC [CI] | mean diff | diff class-cluster 95% CI | GE-better classes | DeLong p<.05 |
|---|---|---|---|---|---|---|
| L1 all classes (mixed corpus) | 0.832 [0.817, 0.849] | 0.658 [0.628, 0.690] | **+0.174** | **[0.075, 0.263]** | 5/5 | 75% of folds |
| L2 minus trivial non_bijective_weak (M4) | 0.790 [0.780, 0.801] | 0.573 [0.555, 0.591] | +0.218 | [0.153, 0.284] | 4/4 | 93% |
| L3 provenance-clean: minus 2 fully-synthetic novel classes (M5) | 0.795 [0.782, 0.809] | 0.611 [0.593, 0.629] | **+0.184** | **[0.124, 0.246]** | 3/3 | 91% |
| L4 real-only corpus (M5 strict) | 0.793 [0.744, 0.842] | 0.710 [0.666, 0.753] | +0.084 | [−0.049, 0.297] (incl. 0) | 2/3 | 34% |

**Recommended headline for the manuscript = L3 (provenance-clean): GE 0.795 vs centroid 0.611,
diff +0.184, class-cluster 95% CI [0.124, 0.246], significant (DeLong) in 91% of 90 folds.**
Report L1 as the full-corpus number and L4 as the honest robustness caveat.

## Per-class (L1, 30 seeds, mean AUROC GE vs NC, diff [seed-bootstrap CI], real fraction of positives)
- borderline_avalanche: 0.777 vs 0.459, **+0.318** [0.301, 0.334] — realpos 0.00 (synthetic-only, M5)
- derived_good: 0.756 vs 0.509, **+0.246** [0.234, 0.259] — realpos 0.10
- high_complexity_good: 0.786 vs 0.605, **+0.182** [0.165, 0.197] — realpos 0.44
- linear_weak: 0.843 vs 0.719, **+0.124** [0.099, 0.148] — realpos 0.95
- non_bijective_weak: 1.000 vs 0.999, +0.001 [0.000, 0.002] — **trivial tie** (both detectors saturate; M4)

## Real-only (L4) per-class — the honest caveat
- high_complexity_good: GE 0.920 vs 0.623, **+0.297** (GE wins decisively within real data)
- linear_weak: 1.000 vs 1.000 (tie, both saturate)
- derived_good: GE 0.467 vs 0.517, **−0.049** (GE loses — but only n=15 real ops, very thin)
→ Within real-only the advantage is carried by high_complexity_good; the class-cluster CI includes 0
because there are only 3 real classes and derived_good is tiny. State this explicitly.

## MI residual-leakage diagnostic (M4) — MI(feature; multiclass label)
mean_cycle_length 0.785 · cycle_count 0.770 · max_cycle_length 0.683 · hamming_dist_1 0.603 ·
control_bits 0.587 · hamming_dist_0 0.510 · hamming_dist_3 0.448 · hamming_dist_2 0.326 · is_involution 0.060.
→ The representational ("de-leaked") features are **informative about class (MI up to 0.785), not
independent of it**. "De-leaked" means "not a deterministic rule scalar," NOT "label-independent."
The open-set capability rests on this informativeness; the manuscript must state this precisely and must
NOT claim the representational features are independent of the label. Numerical health: max condition
number 6.6e8 (full corpus) / 7.8e6 (real-only); selected α spans the grid (degenerate α=0.5 excluded).

## Figures
- `figures/fig_openset_v2_lenses.png` — GE vs NC across L1–L4 with CIs (headline figure).
- `figures/fig_openset_v2_perclass.png` — per-class GE vs NC (L1).

## Manuscript edits this implies (B1 + M4 + M5)
1. Abstract / §6.12 / §7.1 / §10: replace the bare "AUROC 0.833 vs 0.665 (4/5 folds)" with the
   corrected CI-bearing statement (L3 headline + L1 full-corpus), keep the "wins in N/N classes" framing,
   add the DeLong significance and the 30-seed protocol; soften any "істотно краще" to match the CI.
2. §6.12: report all four lenses (the L4 real-only caveat is mandatory — it is the honest M5 control).
3. Methods (§4/§5) + §8: document the corrected 3-way-split protocol and state that the representational
   features are informative-not-independent (MI diagnostic), replacing any "de-leaked = independent" wording.
4. Note the non_bijective_weak trivial tie (drop it from the headline mean; it inflates L1).
