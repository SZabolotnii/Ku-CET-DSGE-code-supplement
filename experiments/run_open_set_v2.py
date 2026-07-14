"""B1 FIX — corrected open-set / novelty experiment (3-way split, symmetric, multi-seed, CI).

The original `cetspace.phase_a.run_open_set` selected the PATP `alpha` on `known_test_idx`
and then used those same points as the ROC negatives, while the nearest-centroid baseline
received no tuning -> an optimistically-biased, asymmetric head-to-head (review blocker B1).

This corrected protocol:
  * 3-way split of the KNOWN operations: train (60%) / val (20%) / test (20%).
  * `select_alpha` chooses alpha on VAL only (disjoint from the test negatives).
  * GE model fit on train; nearest-centroid fit on train -> SYMMETRIC (neither sees test/novel).
  * ROC eval set = test (negatives, is_novel=0) + held-out novel class (positives, is_novel=1).
  * >= 30 seeds; deterministic per-(seed,class) seeding (no hash()).
  * Per-fold DeLong test for the paired AUROC difference (GE vs centroid, same eval sample),
    plus per-fold paired bootstrap, plus seed-level per-class CIs.
  * Four honest lenses: (1) all classes, (2) minus trivially-separable non_bijective_weak (M4),
    (3) provenance-clean = minus the two fully-synthetic novel classes (M5),
    (4) real-only corpus (M5 strict).
  * MI leakage diagnostic of the representational features vs the multiclass label (M4).

Does NOT modify any shipped module. Writes results/open_set_v2/.
Run:  /opt/homebrew/bin/python3 experiments/run_open_set_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cetspace.labels import multiclass_labels  # noqa: E402
from cetspace.models import (  # noqa: E402
    StandardScaler,
    fit_generating_element_model,
    select_alpha,
)
from cetspace.phase_a import representational_columns  # noqa: E402
from cetspace.pipeline import build_feature_tables  # noqa: E402

MASTER = 20260619
N_SEEDS = 30
MIN_NOVEL = 10
OUT = ROOT / "results" / "open_set_v2"


# --------------------------------------------------------------------------- DeLong
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_paired(y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """Fast DeLong (Sun & Xu 2014) for two correlated AUCs on the SAME sample.

    Returns AUC1 (s1), AUC2 (s2), the difference, its SE, a z-stat and a one-sided
    p-value for H1: AUC1 > AUC2. s1 = GE novelty score, s2 = centroid; high = novel.
    """
    pos = y_true == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        return {"auc1": float("nan"), "auc2": float("nan"), "diff": float("nan"),
                "se": float("nan"), "z": float("nan"), "p_one_sided": float("nan")}
    # fast-DeLong requires positives-first ordering; our eval arrays are negatives-first.
    order = np.concatenate([np.where(pos)[0], np.where(neg)[0]])
    scores = np.vstack([s1[order], s2[order]])  # (2, N), first m cols = positives
    pos_s = scores[:, :m]
    neg_s = scores[:, m:]
    k = 2
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos_s[r])
        ty[r] = _compute_midrank(neg_s[r])
        tz[r] = _compute_midrank(scores[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    s = np.atleast_2d(s)
    diff = float(aucs[0] - aucs[1])
    var = float(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    se = float(np.sqrt(var)) if var > 0 else 0.0
    z = diff / se if se > 0 else (np.inf if diff > 0 else (-np.inf if diff < 0 else 0.0))
    # one-sided p for AUC1 > AUC2
    from math import erf, sqrt
    p = 1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))) if np.isfinite(z) else (0.0 if diff > 0 else 1.0)
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": diff,
            "se": se, "z": float(z), "p_one_sided": float(p)}


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, scores))


# --------------------------------------------------------------------------- one fold
def run_fold(seed: int, novel_cls: str, X: np.ndarray, labels: np.ndarray,
             is_real: np.ndarray, restrict_real: bool, class_index: int) -> dict | None:
    base_mask = is_real if restrict_real else np.ones(len(labels), bool)
    sub = np.where(base_mask)[0]
    Xs, ys, reals = X[sub], labels[sub], is_real[sub]

    novel_mask = ys == novel_cls
    if int(novel_mask.sum()) < MIN_NOVEL:
        return None
    known_mask = ~novel_mask
    if len(set(ys[known_mask].tolist())) < 2:
        return None

    known_idx = np.where(known_mask)[0]
    rng = np.random.default_rng([MASTER, seed, class_index, int(restrict_real)])
    rng.shuffle(known_idx)
    n = len(known_idx)
    n_tr = int(round(0.6 * n))
    n_va = int(round(0.2 * n))
    tr = known_idx[:n_tr]
    va = known_idx[n_tr:n_tr + n_va]
    te = known_idx[n_tr + n_va:]
    if len(te) == 0 or len(va) == 0:
        return None
    x_tr, y_tr = Xs[tr], ys[tr]
    x_va, y_va = Xs[va], ys[va]
    # GE needs >= 2 classes in train; alpha selection needs >= 2 in val.
    if len(set(y_tr.tolist())) < 2 or len(set(y_va.tolist())) < 2:
        return None

    novel_idx = np.where(novel_mask)[0]

    # --- GE: alpha on VAL (disjoint from test negatives), fit on train.
    best_alpha, _ = select_alpha(x_tr, y_tr, x_va, y_va)
    model = fit_generating_element_model(x_tr, y_tr, alpha=best_alpha)

    eval_idx = np.concatenate([te, novel_idx])
    is_novel = np.concatenate([np.zeros(len(te), int), np.ones(len(novel_idx), int)])
    x_eval = Xs[eval_idx]
    ge_score = np.min(model.log_msed_features(x_eval), axis=1)

    # --- Centroid: fit on train only (symmetric), same train-fit scaler.
    sc = StandardScaler().fit(x_tr)
    cents = {c: np.mean(sc.transform(x_tr)[y_tr == c], axis=0) for c in sorted(set(y_tr.tolist()))}
    xe = sc.transform(x_eval)
    nc_score = np.array([min(float(np.mean((r - c) ** 2)) for c in cents.values()) for r in xe])

    dl = delong_paired(is_novel, ge_score, nc_score)
    real_neg = float(reals[te].mean())
    real_pos = float(reals[novel_idx].mean())
    return {
        "seed": seed, "held_out_class": novel_cls,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test_neg": int(len(te)),
        "n_novel_pos": int(len(novel_idx)), "best_alpha": float(best_alpha),
        "max_condition_number": float(max(model.condition_numbers.values())),
        "real_frac_negatives": real_neg, "real_frac_positives": real_pos,
        "auroc_ge": _auc(is_novel, ge_score), "auroc_nc": _auc(is_novel, nc_score),
        "delong_diff": dl["diff"], "delong_se": dl["se"], "delong_p_one_sided": dl["p_one_sided"],
    }


# --------------------------------------------------------------------------- aggregation
def _boot_ci(vals: np.ndarray, b: int, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(vals), size=(b, len(vals)))
    means = vals[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(folds: list[dict], label: str) -> dict:
    if not folds:
        return {"lens": label, "n_folds": 0}
    rng = np.random.default_rng(MASTER)
    ge = np.array([f["auroc_ge"] for f in folds])
    nc = np.array([f["auroc_nc"] for f in folds])
    diff = ge - nc
    classes = sorted(set(f["held_out_class"] for f in folds))
    per_class = {}
    class_mean_diffs = []
    for c in classes:
        g = np.array([f["auroc_ge"] for f in folds if f["held_out_class"] == c])
        d = np.array([f["auroc_nc"] for f in folds if f["held_out_class"] == c])
        per_class[c] = {
            "n_seeds": int(len(g)),
            "mean_auroc_ge": float(g.mean()), "ge_ci": _boot_ci(g, 5000, rng),
            "mean_auroc_nc": float(d.mean()), "nc_ci": _boot_ci(d, 5000, rng),
            "mean_diff": float((g - d).mean()), "diff_ci": _boot_ci(g - d, 5000, rng),
            "real_frac_positives": float(np.mean([f["real_frac_positives"] for f in folds if f["held_out_class"] == c])),
        }
        class_mean_diffs.append(float((g - d).mean()))
    # class-cluster bootstrap (resample classes -> conservative, accounts for seed correlation)
    cmd = np.array(class_mean_diffs)
    cluster_idx = rng.integers(0, len(cmd), size=(10000, len(cmd)))
    cluster_means = cmd[cluster_idx].mean(axis=1)
    return {
        "lens": label, "n_folds": int(len(folds)), "classes": classes,
        "mean_auroc_ge": float(ge.mean()), "ge_ci_foldboot": _boot_ci(ge, 10000, rng),
        "mean_auroc_nc": float(nc.mean()), "nc_ci_foldboot": _boot_ci(nc, 10000, rng),
        "mean_diff": float(diff.mean()), "diff_ci_foldboot": _boot_ci(diff, 10000, rng),
        "diff_ci_classcluster": [float(np.quantile(cluster_means, 0.025)), float(np.quantile(cluster_means, 0.975))],
        "ge_win_rate": float(np.mean(diff > 0)),
        "delong_frac_p_lt_05": float(np.mean([f["delong_p_one_sided"] < 0.05 for f in folds])),
        "classes_with_ge_better": int(sum(1 for c in classes if per_class[c]["mean_diff"] > 0)),
        "n_classes": len(classes),
        "max_condition_number": float(max(f["max_condition_number"] for f in folds)),
        "alpha_values": sorted(set(round(f["best_alpha"], 2) for f in folds)),
        "per_class": per_class,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, feature_df, feature_cols = build_feature_tables(seed=20260617)
    rep_cols = representational_columns(feature_cols)
    labels = multiclass_labels(feature_df)
    X = feature_df[rep_cols].astype(float).to_numpy()
    is_real = feature_df["is_real_corpus"].astype(bool).to_numpy()
    class_list = sorted(set(labels.tolist()))
    cidx = {c: i for i, c in enumerate(class_list)}

    # MI leakage diagnostic (M4): MI of each representational feature with the multiclass label.
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, labels, discrete_features="auto", random_state=MASTER)
    mi_table = sorted(zip(rep_cols, [float(v) for v in mi]), key=lambda t: -t[1])

    SYNTH_ONLY = ["borderline_avalanche", "non_bijective_weak"]  # 0 real ops (M5)
    TRIVIAL = ["non_bijective_weak"]  # AUROC 1.0/1.0 both detectors (M4)

    all_folds = []
    for seed in range(N_SEEDS):
        for c in class_list:
            r = run_fold(seed, c, X, labels, is_real, restrict_real=False, class_index=cidx[c])
            if r:
                all_folds.append(r)
    real_folds = []
    for seed in range(N_SEEDS):
        for c in class_list:
            r = run_fold(seed, c, X, labels, is_real, restrict_real=True, class_index=cidx[c])
            if r:
                real_folds.append(r)

    lenses = {
        "L1_all_classes": aggregate(all_folds, "all classes (mixed corpus)"),
        "L2_minus_trivial": aggregate([f for f in all_folds if f["held_out_class"] not in TRIVIAL],
                                      "minus trivially-separable non_bijective_weak (M4)"),
        "L3_provenance_clean": aggregate([f for f in all_folds if f["held_out_class"] not in SYNTH_ONLY],
                                         "provenance-clean: minus the 2 fully-synthetic novel classes (M5)"),
        "L4_real_only": aggregate(real_folds, "real-corpus only (M5 strict)"),
    }

    out = {
        "protocol": "corrected open-set: 3-way known split (train/val/test), alpha on VAL, "
                    "symmetric GE-vs-centroid, >=30 seeds, DeLong + bootstrap CIs",
        "fix_of": "src/cetspace/phase_a.py::run_open_set (B1: alpha tuned on ROC negatives)",
        "n_seeds": N_SEEDS, "min_novel": MIN_NOVEL, "rep_cols": rep_cols,
        "mi_diagnostic_label_vs_features": mi_table,
        "lenses": lenses,
        "all_folds": all_folds, "real_only_folds": real_folds,
    }
    (OUT / "open_set_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # console summary
    print("MI(feature; multiclass label) — residual-leakage diagnostic (M4):")
    for name, v in mi_table:
        print(f"  {name:18} {v:.3f}")
    print()
    for key, L in lenses.items():
        if not L.get("n_folds"):
            print(f"{key}: no folds"); continue
        print(f"== {key}: {L['lens']} ==")
        print(f"   folds={L['n_folds']}  GE={L['mean_auroc_ge']:.3f} {L['ge_ci_foldboot']}  "
              f"NC={L['mean_auroc_nc']:.3f} {L['nc_ci_foldboot']}")
        print(f"   diff={L['mean_diff']:.3f}  fold-boot CI {L['diff_ci_foldboot']}  "
              f"class-cluster CI {L['diff_ci_classcluster']}")
        print(f"   GE-better classes {L['classes_with_ge_better']}/{L['n_classes']}  "
              f"win-rate {L['ge_win_rate']:.2f}  DeLong p<.05 in {L['delong_frac_p_lt_05']:.2f} of folds  "
              f"maxcond={L['max_condition_number']:.1e}  alphas={L['alpha_values']}")
        for c, pc in L["per_class"].items():
            print(f"     {c:22} GE {pc['mean_auroc_ge']:.3f} vs NC {pc['mean_auroc_nc']:.3f}  "
                  f"diff {pc['mean_diff']:+.3f} CI {pc['diff_ci']}  realpos={pc['real_frac_positives']:.2f}")
        print()
    print(f"written: {OUT/'open_set_v2.json'}")


if __name__ == "__main__":
    main()
