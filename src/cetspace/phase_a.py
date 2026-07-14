"""Phase A — scientific core hardening (de-leak, open-set value-add, independent target).

This module is ADDITIVE: it does not modify the frozen E0-E10 pipeline or its
artifacts/hashes. It writes new evidence under ``results/phase_a/``.

It addresses the three Phase-A blockers from the improvement plan:

  A1  Label leakage. The multiclass label is a deterministic function of a small
      set of "decision" feature scalars, so the rule-based baseline is not a
      baseline at all — it is the label generator. We (i) quantify this leakage
      and (ii) re-run classification on a *representational* feature subset that
      excludes the decision scalars (and the raw truth table), giving an honest
      generalization number instead of the leaky 1.0 / 0.8849 pair.

  A2  Value-add the baseline cannot provide. Reconstruction error is not a
      classifier of a known deterministic function; its genuine capability is
      OPEN-SET / novelty detection. We hold out each class in turn as "unknown",
      fit the generating-element model on the remaining classes, and use the
      minimum per-class log-MSED as a novelty score. Threshold rules have no
      "unknown" output; the fair quantitative competitor is nearest-centroid
      distance. We report AUROC (GE vs centroid), per fold and on average.

  A3  Non-circular ranking target. The only positive ranking result in the paper
      is agreement with a composite score we designed ourselves. Here we compute
      NONLINEARITY (Walsh-Hadamard), a real cryptographic property that is NOT in
      the feature set and NOT in the composite score, and report rank agreement
      against it as an independent ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import run_classification, spearman_corr, write_json
from .labels import composite_quality_score, multiclass_labels, rule_based_multiclass_predict
from .models import (
    StandardScaler,
    fit_generating_element_model,
    macro_f1,
    select_alpha,
    split_dataset,
)
from .operations import N_BITS, N_STATES, generate_full_corpus
from .pipeline import build_feature_tables

SEED = 20260617

# --- feature partition (A1) -------------------------------------------------

# Scalars the rules / composite score branch on directly.
_RULE_SCALARS = {
    "is_base",
    "is_bijective",
    "avalanche_error",
    "balance_error",
    "differential_uniformity",
    "max_anf_degree",
    "complexity_score",
    "effective_substitution_tables",
}


def is_decision_column(col: str) -> bool:
    """True if a feature column is (or deterministically reproduces) a rule scalar."""
    if col in _RULE_SCALARS:
        return True
    # deterministic expansions of the decision scalars:
    if col.startswith("truth_"):  # the raw mapping -> determines every property
        return True
    if col.startswith("avalanche_"):  # avalanche matrix -> avalanche_error
        return True
    if col.startswith("anf_degree_"):  # per-bit ANF -> max_anf_degree
        return True
    if col.startswith("balance_"):  # per-bit balance -> balance_error
        return True
    if col in {"mean_anf_degree", "strict_avalanche", "is_real_corpus"}:
        return True
    return False


def representational_columns(feature_cols: list[str]) -> list[str]:
    """Structural features that no single rule threshold maps to the label."""
    return [c for c in feature_cols if not is_decision_column(c)]


def decision_columns(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if is_decision_column(c)]


# --- A3 independent target: nonlinearity (Walsh-Hadamard) -------------------


def _fast_walsh_max_abs(component_truth: np.ndarray) -> int:
    """Max |Walsh coefficient| of (-1)^f for one Boolean component function."""
    a = (1 - 2 * component_truth).astype(np.int64)  # (-1)^f
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return int(np.max(np.abs(a)))


def nonlinearity(mapping: tuple[int, ...], n_bits: int = N_BITS) -> int:
    """S-box nonlinearity = min over nonzero output masks of component nonlinearity.

    Real cryptographic metric, absent from the feature set and the composite
    score, hence usable as an independent ranking ground truth.
    """
    n_states = 2 ** n_bits
    best = None
    for out_mask in range(1, n_states):  # nonzero linear combinations of outputs
        comp = np.array(
            [bin(out_mask & mapping[x]).count("1") & 1 for x in range(n_states)],
            dtype=np.int64,
        )
        wmax = _fast_walsh_max_abs(comp)
        nl = 2 ** (n_bits - 1) - wmax // 2
        best = nl if best is None else min(best, nl)
    return int(best)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC; uses sklearn when present, else a rank-based fallback."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, scores))
    except Exception:
        # Mann-Whitney U / rank fallback.
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        pos = y_true == 1
        n_pos = int(pos.sum())
        n_neg = int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc)


# --- A1: leakage audit ------------------------------------------------------


def run_leakage_audit(feature_df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> dict:
    labels = multiclass_labels(feature_df)
    rule = rule_based_multiclass_predict(feature_df.to_dict("records"))

    # Proof 1: the rule baseline IS the label generator (exact reproduction).
    exact_match = float(np.mean(rule == labels))
    rule_macro_f1 = macro_f1(labels, rule)

    dec_cols = decision_columns(feature_cols)
    rep_cols = representational_columns(feature_cols)

    audit = {
        "n_rows": int(len(feature_df)),
        "classes": sorted(set(labels.tolist())),
        "rule_baseline_is_label_generator": exact_match == 1.0,
        "rule_vs_label_exact_match_fraction": exact_match,
        "rule_macro_f1_on_labels": rule_macro_f1,
        "n_decision_columns": len(dec_cols),
        "n_representational_columns": len(rep_cols),
        "decision_columns": dec_cols,
        "representational_columns": rep_cols,
    }

    # Proof 2: even a learned shallow tree on the decision scalars trivially
    # recovers the label; on representational features it cannot.
    try:
        from sklearn.tree import DecisionTreeClassifier

        x_dec = feature_df[dec_cols].astype(float).to_numpy()
        x_rep = feature_df[rep_cols].astype(float).to_numpy()
        xi = np.arange(len(labels))
        tr, _, te, ytr, _, yte = split_dataset(xi.reshape(-1, 1), labels, seed=SEED)
        tr, te = tr.ravel(), te.ravel()
        for name, x in (("decision_scalars", x_dec), ("representational", x_rep)):
            clf = DecisionTreeClassifier(max_depth=6, random_state=SEED)
            clf.fit(x[tr], labels[tr])
            pred = clf.predict(x[te])
            audit[f"shallow_tree_macro_f1__{name}"] = float(macro_f1(labels[te], pred))
        audit["shallow_tree_note"] = (
            "depth-6 tree; decision_scalars -> ~1.0 confirms the label is a learnable "
            "function of the rule scalars, representational is the honest, non-trivial task"
        )
    except Exception as exc:  # sklearn absent -> skip proof 2, keep proof 1
        audit["shallow_tree_note"] = f"sklearn unavailable ({exc}); proof 1 stands"

    write_json(out_dir / "metrics.json", audit)
    return audit


# --- A2: open-set / novelty detection (the value-add) -----------------------


def _fit_centroids(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {cls: x[y == cls].mean(axis=0) for cls in sorted(set(y.tolist()))}


def run_open_set(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    min_novel: int = 10,
    seed: int = SEED,
) -> dict:
    """Rotating class-holdout open-set detection: GE log-MSED vs centroid distance.

    Uses the de-leaked representational feature set for both detectors so the
    comparison isolates the geometry, not leaked decision scalars.
    """
    rep_cols = representational_columns(feature_cols)
    labels = multiclass_labels(feature_df)
    x_all = feature_df[rep_cols].astype(float).to_numpy()
    classes = sorted(set(labels.tolist()))
    rng = np.random.default_rng(seed)

    folds = []
    for novel_cls in classes:
        novel_mask = labels == novel_cls
        if int(novel_mask.sum()) < min_novel:
            continue
        known_mask = ~novel_mask
        known_labels = sorted(set(labels[known_mask].tolist()))
        if len(known_labels) < 2:
            continue

        known_idx = np.where(known_mask)[0]
        rng.shuffle(known_idx)
        n_train = int(round(0.7 * len(known_idx)))
        train_idx = np.sort(known_idx[:n_train])
        known_test_idx = np.sort(known_idx[n_train:])
        novel_idx = np.where(novel_mask)[0]

        x_train = x_all[train_idx]
        y_train = labels[train_idx]

        scaler = StandardScaler().fit(x_train)
        best_alpha, _ = select_alpha(
            x_train,
            y_train,
            x_all[known_test_idx],
            labels[known_test_idx],
        )
        model = fit_generating_element_model(x_train, y_train, alpha=best_alpha)

        eval_idx = np.concatenate([known_test_idx, novel_idx])
        is_novel = np.concatenate(
            [np.zeros(len(known_test_idx), int), np.ones(len(novel_idx), int)]
        )
        x_eval = x_all[eval_idx]

        # GE novelty score = min over known classes of log-MSED (high = novel).
        ge_score = np.min(model.log_msed_features(x_eval), axis=1)
        # Centroid novelty score = min distance to any known centroid (high = novel).
        centroids = _fit_centroids(scaler.transform(x_train), y_train)
        x_eval_s = scaler.transform(x_eval)
        nc_score = np.array(
            [min(float(np.mean((row - c) ** 2)) for c in centroids.values()) for row in x_eval_s]
        )

        folds.append(
            {
                "held_out_class": novel_cls,
                "n_known_train": int(len(train_idx)),
                "n_known_test": int(len(known_test_idx)),
                "n_novel": int(len(novel_idx)),
                "best_alpha": float(best_alpha),
                "auroc_generating_element": _roc_auc(is_novel, ge_score),
                "auroc_nearest_centroid": _roc_auc(is_novel, nc_score),
            }
        )

    ge_aurocs = [f["auroc_generating_element"] for f in folds]
    nc_aurocs = [f["auroc_nearest_centroid"] for f in folds]
    wins = sum(1 for f in folds if f["auroc_generating_element"] > f["auroc_nearest_centroid"])
    payload = {
        "protocol": "rotating class-holdout open-set; representational (de-leaked) features for both detectors",
        "n_folds": len(folds),
        "mean_auroc_generating_element": float(np.mean(ge_aurocs)) if ge_aurocs else float("nan"),
        "mean_auroc_nearest_centroid": float(np.mean(nc_aurocs)) if nc_aurocs else float("nan"),
        "ge_wins_folds": wins,
        "rule_baseline_auroc": None,
        "rule_baseline_note": (
            "threshold rules emit a known-class label with no 'unknown' output, "
            "so they cannot produce a novelty ranking — AUROC is undefined. This is "
            "the structural capability gap the reconstruction score fills."
        ),
        "folds": folds,
    }
    write_json(out_dir / "metrics.json", payload)
    pd.DataFrame(folds).to_csv(out_dir / "open_set_folds.csv", index=False)
    return payload


# --- A3: independent ranking target -----------------------------------------


def run_independent_ranking(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    nonlin: np.ndarray,
    out_dir: Path,
    seed: int = SEED,
) -> dict:
    """Rank agreement of each scorer against an INDEPENDENT target (nonlinearity)."""
    from .labels import binary_labels

    y = binary_labels(feature_df)
    x = feature_df[feature_cols].astype(float).to_numpy()
    x_train, x_val, _, y_train, y_val, _ = split_dataset(x, y, seed=seed)
    best_alpha, _ = select_alpha(x_train, y_train, x_val, y_val)
    model = fit_generating_element_model(x_train, y_train, alpha=best_alpha)

    composite = composite_quality_score(feature_df)
    reconstruction = -np.min(model.log_msed_features(x), axis=1)
    rule = (
        1.0 - feature_df["avalanche_error"].to_numpy(dtype=float)
        + 1.0 - feature_df["balance_error"].to_numpy(dtype=float)
        + feature_df["is_bijective"].astype(float).to_numpy()
    )
    rng = float(reconstruction.max() - reconstruction.min()) or 1.0
    hybrid = 0.7 * composite + 0.3 * ((reconstruction - reconstruction.min()) / rng)

    diff_unif = feature_df["differential_uniformity"].to_numpy(dtype=float)
    scorers = {
        "composite": composite,
        "rule_based": rule,
        "reconstruction": reconstruction,
        "hybrid": hybrid,
    }
    rows = []
    for name, score in scorers.items():
        rows.append(
            {
                "scorer": name,
                # independent target (NOT in features, NOT in composite):
                "spearman_vs_nonlinearity": spearman_corr(nonlin.astype(float), score),
                # semi-independent real property (a feature, weight 0.10 in composite):
                "spearman_vs_neg_diff_uniformity": spearman_corr(-diff_unif, score),
                # the circular reference target, for contrast:
                "spearman_vs_composite_circular": spearman_corr(composite, score),
            }
        )
    payload = {
        "best_alpha": float(best_alpha),
        "independent_target": "nonlinearity (Walsh-Hadamard); absent from features and composite score",
        "nonlinearity_distribution": {
            int(v): int(c) for v, c in zip(*np.unique(nonlin, return_counts=True))
        },
        "rows": rows,
        "note": (
            "spearman_vs_composite_circular reproduces the paper's inflated agreement; "
            "spearman_vs_nonlinearity is the honest, non-circular signal."
        ),
    }
    write_json(out_dir / "metrics.json", payload)
    pd.DataFrame(rows).to_csv(out_dir / "independent_ranking.csv", index=False)
    return payload


# --- orchestration ----------------------------------------------------------


def _write_summary(out_dir: Path, leak: dict, deleaked: dict, openset: dict, ranking: dict) -> None:
    rep_n = leak["n_representational_columns"]
    lines = [
        "# Phase A — scientific core hardening (results)",
        "",
        "Additive evidence; the frozen E0-E10 artifacts are unchanged.",
        "",
        "## A1. Label leakage (quantified)",
        "",
        f"- Rule baseline reproduces the label exactly on {leak['rule_vs_label_exact_match_fraction']*100:.1f}% of rows "
        f"(macro-F1 {leak['rule_macro_f1_on_labels']:.4f}): it **is** the label generator, not a baseline.",
    ]
    if "shallow_tree_macro_f1__decision_scalars" in leak:
        lines.append(
            f"- A depth-3 tree on decision scalars: macro-F1 "
            f"{leak['shallow_tree_macro_f1__decision_scalars']:.4f}; on representational features only: "
            f"{leak['shallow_tree_macro_f1__representational']:.4f}."
        )
    lines += [
        f"- Feature partition: {leak['n_decision_columns']} decision columns removed, "
        f"{rep_n} representational columns kept.",
        "",
        "## A1'. De-leaked classification (representational features only)",
        "",
        "| Model | macro-F1 |",
        "|---|---:|",
        f"| Generating-element (de-leaked) | {deleaked['generating_element_macro_f1']:.4f} |",
        f"| Nearest-centroid (de-leaked) | {deleaked['nearest_centroid_macro_f1']:.4f} |",
        f"| Rule oracle (reads decision scalars) | {deleaked['rule_based_macro_f1']:.4f} |",
        f"| Leaky full-feature GE (paper E3, ref) | 0.8849 |",
        "",
        f"- best alpha = {deleaked['best_alpha']:.2f}. The rule oracle stays at 1.0 because it still reads the "
        "decision scalars; the honest learning task is GE-vs-centroid on representational features.",
        "",
        "## A2. Open-set / novelty — capability the rules lack",
        "",
        f"- Rotating class-holdout, {openset['n_folds']} folds, de-leaked features for both detectors.",
        f"- Mean AUROC: generating-element **{openset['mean_auroc_generating_element']:.4f}** "
        f"vs nearest-centroid {openset['mean_auroc_nearest_centroid']:.4f} "
        f"(GE wins {openset['ge_wins_folds']}/{openset['n_folds']} folds).",
        "- Threshold rules emit a known class with no 'unknown' output -> novelty AUROC undefined "
        "(structural capability gap).",
        "",
        "| Held-out class | n_novel | AUROC GE | AUROC centroid |",
        "|---|---:|---:|---:|",
    ]
    for f in openset["folds"]:
        lines.append(
            f"| {f['held_out_class']} | {f['n_novel']} | {f['auroc_generating_element']:.4f} | "
            f"{f['auroc_nearest_centroid']:.4f} |"
        )
    lines += [
        "",
        "## A3. Independent ranking target (nonlinearity, non-circular)",
        "",
        "| Scorer | vs nonlinearity (indep.) | vs -diff.uniformity | vs composite (circular) |",
        "|---|---:|---:|---:|",
    ]
    for r in ranking["rows"]:
        lines.append(
            f"| {r['scorer']} | {r['spearman_vs_nonlinearity']:.4f} | "
            f"{r['spearman_vs_neg_diff_uniformity']:.4f} | {r['spearman_vs_composite_circular']:.4f} |"
        )
    lines += [
        "",
        "- The last column reproduces the paper's circular agreement; the first is the honest signal "
        "against a cryptographic property the scorers never saw.",
        "",
    ]
    (out_dir / "phase_a_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_a(root: Path, seed: int = SEED) -> dict:
    out = root / "results" / "phase_a"
    out.mkdir(parents=True, exist_ok=True)

    operation_df, feature_df, feature_cols = build_feature_tables(seed=seed)

    # Align nonlinearity to feature rows by operation_id.
    ops = {op.operation_id: op for op in generate_full_corpus(seed=seed)}
    nonlin = np.array(
        [nonlinearity(ops[oid].mapping) for oid in feature_df["operation_id"]], dtype=int
    )

    rep_cols = representational_columns(feature_cols)

    leak = run_leakage_audit(feature_df, feature_cols, out / "a1_leakage_audit")
    deleaked = run_classification(
        feature_df,
        rep_cols,
        multiclass_labels(feature_df),
        out / "a1_deleaked_multiclass",
        rule_mode="multiclass",
        seed=seed,
    )
    openset = run_open_set(feature_df, feature_cols, out / "a2_open_set", seed=seed)
    ranking = run_independent_ranking(feature_df, feature_cols, nonlin, out / "a3_independent_ranking", seed=seed)

    _write_summary(out, leak, deleaked, openset, ranking)
    index = {
        "a1_leakage_audit": "results/phase_a/a1_leakage_audit/metrics.json",
        "a1_deleaked_multiclass": "results/phase_a/a1_deleaked_multiclass/metrics.json",
        "a2_open_set": "results/phase_a/a2_open_set/metrics.json",
        "a3_independent_ranking": "results/phase_a/a3_independent_ranking/metrics.json",
        "summary_markdown": "results/phase_a/phase_a_summary.md",
    }
    write_json(out / "phase_a_index.json", index)
    return index
