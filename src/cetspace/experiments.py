from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnostics import feature_diagnostics, write_corpus_manifest
from .labels import binary_labels, composite_quality_score, multiclass_labels, rule_based_multiclass_predict
from .models import (
    bootstrap_macro_f1,
    classification_report_dict,
    fit_classical_baselines,
    fit_generating_element_model,
    macro_f1,
    rule_based_predict,
    select_alpha,
    split_dataset,
)
from .patp import alpha_grid, is_degenerate_alpha
from .pipeline import build_feature_tables


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        true: {pred: int(np.sum((y_true == true) & (y_pred == pred))) for pred in labels}
        for true in labels
    }


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in sorted(set(y_true.tolist())):
        denom = np.sum(y_true == cls)
        recalls.append(float(np.sum((y_true == cls) & (y_pred == cls)) / denom) if denom else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_overlap(score_a: np.ndarray, score_b: np.ndarray, k: int) -> float:
    a = set(np.argsort(score_a)[-k:].tolist())
    b = set(np.argsort(score_b)[-k:].tolist())
    return float(len(a & b) / k)


def run_classification(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
    out_dir: Path,
    rule_mode: str,
    seed: int = 20260617,
) -> dict:
    x = feature_df[feature_cols].astype(float).to_numpy()
    y = labels
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, y, seed=seed)
    best_alpha, alpha_rows = select_alpha(x_train, y_train, x_val, y_val)
    model = fit_generating_element_model(x_train, y_train, alpha=best_alpha)
    y_pred = model.predict(x_test)
    _, _, test_indices, _, _, _ = split_dataset(np.arange(len(y)).reshape(-1, 1), y, seed=seed)
    test_indices = test_indices.ravel().astype(int)
    test_records = feature_df.iloc[test_indices].to_dict("records")
    if rule_mode == "binary":
        y_rule = rule_based_predict(test_records)
    else:
        y_rule = rule_based_multiclass_predict(test_records)
    classical = fit_classical_baselines(x_train, y_train, x_test)
    metrics = {
        "best_alpha": best_alpha,
        "degenerate_alpha": 0.5,
        "degenerate_alpha_is_candidate": False,
        "generating_element_macro_f1": macro_f1(y_test, y_pred),
        "rule_based_macro_f1": macro_f1(y_test, y_rule),
        "nearest_centroid_macro_f1": macro_f1(y_test, classical["nearest_centroid"]),
        "balanced_accuracy": balanced_accuracy(y_test, y_pred),
        "classification_report": classification_report_dict(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "rule_confusion_matrix": confusion_matrix(y_test, y_rule),
        "bootstrap_macro_f1": bootstrap_macro_f1(y_test, y_pred, n_bootstrap=1000, seed=seed),
        "condition_numbers": model.condition_numbers,
        "alpha_sweep": alpha_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(alpha_rows).to_csv(out_dir / "alpha_sweep.csv", index=False)
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def run_ranking(feature_df: pd.DataFrame, feature_cols: list[str], out_dir: Path, seed: int = 20260617) -> dict:
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
    reconstruction_range = float(np.ptp(reconstruction)) or 1.0
    hybrid = 0.7 * composite + 0.3 * ((reconstruction - reconstruction.min()) / reconstruction_range)
    rows = []
    for name, score in [
        ("rule_based_rank", rule),
        ("generating_element_rank", reconstruction),
        ("hybrid_rank", hybrid),
    ]:
        rows.append(
            {
                "method": name,
                "spearman_vs_composite": spearman_corr(composite, score),
                "top10_overlap": topk_overlap(composite, score, 10),
                "top25_overlap": topk_overlap(composite, score, 25),
                "top50_overlap": topk_overlap(composite, score, 50),
                "real_top50_share": float(feature_df.iloc[np.argsort(score)[-50:]]["is_real_corpus"].mean()),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = feature_df[["operation_id", "source", "is_real_corpus", "quality_label"]].copy()
    ranked["composite_quality_score"] = composite
    ranked["generating_element_rank_score"] = reconstruction
    ranked["hybrid_score"] = hybrid
    ranked.sort_values("hybrid_score", ascending=False).to_csv(out_dir / "ranked_operations.csv", index=False)
    metrics = {"best_alpha": best_alpha, "ranking_metrics": rows}
    pd.DataFrame(rows).to_csv(out_dir / "ranking_metrics.csv", index=False)
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def run_alpha_sweep(feature_df: pd.DataFrame, feature_cols: list[str], out_dir: Path, seed: int = 20260617) -> dict:
    y = multiclass_labels(feature_df)
    x = feature_df[feature_cols].astype(float).to_numpy()
    x_train, x_val, _, y_train, y_val, _ = split_dataset(x, y, seed=seed)
    rows = []
    for alpha in alpha_grid():
        model = fit_generating_element_model(x_train, y_train, alpha=float(alpha))
        pred = model.predict(x_val)
        rows.append(
            {
                "alpha": float(alpha),
                "macro_f1": macro_f1(y_val, pred),
                "is_degenerate_alpha": is_degenerate_alpha(float(alpha)),
                "is_boundary": float(alpha) in (0.0, 1.0),
                "max_condition_number": max(model.condition_numbers.values()),
            }
        )
    candidates = [r for r in rows if not r["is_degenerate_alpha"]]
    best = max(candidates, key=lambda r: r["macro_f1"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "alpha_sweep.csv", index=False)
    metrics = {"best_alpha": best["alpha"], "best_macro_f1": best["macro_f1"], "rows": rows}
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def run_ablation(feature_df: pd.DataFrame, feature_cols: list[str], out_dir: Path, seed: int = 20260617) -> dict:
    groups = {
        "truth": [c for c in feature_cols if c.startswith("truth_")],
        "avalanche_hamming": [c for c in feature_cols if c.startswith("avalanche_") or c.startswith("hamming_dist_")],
        "algebraic": [c for c in feature_cols if c.startswith("anf_degree_") or c in ["max_anf_degree", "mean_anf_degree"]],
        "structural": [c for c in feature_cols if c in ["cycle_count", "max_cycle_length", "mean_cycle_length", "is_bijective", "is_involution"]],
        "complexity": [c for c in feature_cols if c in ["complexity_score", "effective_substitution_tables", "control_bits"]],
    }
    configs = {
        "all_features": feature_cols,
        "truth_table_only": groups["truth"],
        "avalanche_hamming_only": groups["avalanche_hamming"],
        "algebraic_only": groups["algebraic"],
        "structural_cycles_only": groups["structural"],
        "complexity_removed": [c for c in feature_cols if c not in groups["complexity"]],
        "truth_table_removed": [c for c in feature_cols if c not in groups["truth"]],
        "avalanche_removed": [c for c in feature_cols if c not in groups["avalanche_hamming"]],
        "minimal_cryptographic_metrics": [
            c for c in ["avalanche_error", "balance_error", "differential_uniformity", "max_anf_degree", "is_bijective"] if c in feature_cols
        ],
    }
    labels = multiclass_labels(feature_df)
    rows = []
    for name, cols in configs.items():
        if not cols:
            continue
        metrics = run_classification(feature_df, cols, labels, out_dir / name, rule_mode="multiclass", seed=seed)
        rank = run_ranking(feature_df, cols, out_dir / name / "ranking", seed=seed)
        ge_rank = next(r for r in rank["ranking_metrics"] if r["method"] == "generating_element_rank")
        rows.append(
            {
                "config": name,
                "n_features": len(cols),
                "macro_f1": metrics["generating_element_macro_f1"],
                "alpha_opt": metrics["best_alpha"],
                "max_condition_number": max(metrics["condition_numbers"].values()),
                "spearman_rank": ge_rank["spearman_vs_composite"],
                "top25_overlap": ge_rank["top25_overlap"],
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "ablation_summary.csv", index=False)
    payload = {"rows": rows}
    write_json(out_dir / "metrics.json", payload)
    return payload


def add_noise(feature_df: pd.DataFrame, feature_cols: list[str], seed: int, feature_noise: str, label_noise: float) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    noisy = feature_df.copy()
    noisy[feature_cols] = noisy[feature_cols].astype(float)
    x = noisy[feature_cols].astype(float).to_numpy()
    if feature_noise == "gaussian":
        x = x + rng.normal(0, 0.05, size=x.shape)
    elif feature_noise == "heavy_tailed":
        x = x + rng.standard_t(df=3, size=x.shape) * 0.03
    noisy.loc[:, feature_cols] = x
    labels = binary_labels(noisy).copy()
    if label_noise > 0:
        flip = rng.random(len(labels)) < label_noise
        labels[flip] = np.where(labels[flip] == "acceptable", "weak", "acceptable")
    return noisy, labels


def run_monte_carlo(feature_df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> dict:
    rows = []
    sample_sizes = [50, 100, 200, 500, 1000]
    model_repetitions = 20
    metric_bootstrap_repetitions = 1000
    rng = np.random.default_rng(20260617)
    for n in sample_sizes:
        scores = []
        for rep in range(model_repetitions):
            sample = feature_df.sample(n=min(n, len(feature_df)), replace=True, random_state=20260617 + rep)
            metrics = run_classification(sample, feature_cols, binary_labels(sample), out_dir / f"n{n}_rep{rep}", "binary", seed=20260617 + rep)
            scores.append(metrics["generating_element_macro_f1"])
        arr = np.array(scores)
        boot_means = []
        for _ in range(metric_bootstrap_repetitions):
            boot_means.append(float(rng.choice(arr, size=len(arr), replace=True).mean()))
        rows.append(
            {
                "n": n,
                "model_repetitions": model_repetitions,
                "metric_bootstrap_repetitions": metric_bootstrap_repetitions,
                "mean_macro_f1": float(arr.mean()),
                "std_macro_f1": float(arr.std()),
                "ci95_low": float(np.quantile(boot_means, 0.025)),
                "ci95_high": float(np.quantile(boot_means, 0.975)),
                "bias_from_one": float(1.0 - arr.mean()),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "sample_size_sweep.csv", index=False)
    scenario = feature_df.groupby("source").agg(
        count=("operation_id", "count"),
        acceptable_share=("quality_label", lambda s: float((s == "acceptable").mean())),
        mean_avalanche_error=("avalanche_error", "mean"),
        mean_balance_error=("balance_error", "mean"),
        mean_complexity=("complexity_score", "mean"),
    ).reset_index()
    scenario.to_csv(out_dir / "scenario_summary.csv", index=False)
    payload = {
        "sample_size_sweep": rows,
        "scenario_summary": scenario.to_dict("records"),
        "note": "Each sample-size cell uses 20 full model refits and 1000 bootstrap replicates over the fitted metrics for confidence bounds.",
    }
    write_json(out_dir / "metrics.json", payload)
    return payload


def run_robustness(feature_df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> dict:
    rows = []
    for seed in range(20260617, 20260637):
        for label_noise in [0.0, 0.05, 0.10, 0.20]:
            for feature_noise in ["none", "gaussian", "heavy_tailed"]:
                noisy, labels = add_noise(feature_df, feature_cols, seed, feature_noise, label_noise)
                metrics = run_classification(noisy, feature_cols, labels, out_dir / f"s{seed}_ln{label_noise}_{feature_noise}", "binary", seed=seed)
                rows.append(
                    {
                        "seed": seed,
                        "label_noise": label_noise,
                        "feature_noise": feature_noise,
                        "macro_f1": metrics["generating_element_macro_f1"],
                        "alpha_opt": metrics["best_alpha"],
                    }
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "robustness_runs.csv", index=False)
    summary = df.groupby(["label_noise", "feature_noise"]).agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
    ).reset_index()
    summary.to_csv(out_dir / "robustness_summary.csv", index=False)
    alpha_freq = df["alpha_opt"].value_counts().sort_index().reset_index()
    alpha_freq.columns = ["alpha_opt", "count"]
    alpha_freq.to_csv(out_dir / "alpha_opt_frequency.csv", index=False)
    payload = {"summary": summary.to_dict("records"), "alpha_frequency": alpha_freq.to_dict("records")}
    write_json(out_dir / "metrics.json", payload)
    return payload


def run_real_data_validation(feature_df: pd.DataFrame, feature_cols: list[str], out_dir: Path, seed: int = 20260617) -> dict:
    real = feature_df[feature_df["is_real_corpus"] == True].copy()
    labels = multiclass_labels(feature_df)
    x = feature_df[feature_cols].astype(float).to_numpy()
    x_train, x_val, _, y_train, y_val, _ = split_dataset(x, labels, seed=seed)
    best_alpha, _ = select_alpha(x_train, y_train, x_val, y_val)
    model = fit_generating_element_model(x_train, y_train, alpha=best_alpha)
    pred = model.predict(real[feature_cols].astype(float).to_numpy())
    real["multiclass_prediction"] = pred
    out_dir.mkdir(parents=True, exist_ok=True)
    real.to_csv(out_dir / "real_reconstructed_predictions.csv", index=False)
    by_tables = real.groupby("effective_substitution_tables").size().reset_index(name="count")
    by_pred = real["multiclass_prediction"].value_counts().reset_index()
    by_tables.to_csv(out_dir / "substitution_table_regimes.csv", index=False)
    by_pred.to_csv(out_dir / "real_prediction_distribution.csv", index=False)
    payload = {
        "r1_complete": True,
        "r2_public_sources_status": "not_extracted_in_v1; local PDFs used for reconstruction",
        "r3_author_data_status": "not_provided",
        "real_rows": int(len(real)),
        "base_rows": int((real["is_base"] == True).sum()),
        "prediction_distribution": dict(zip(by_pred["multiclass_prediction"], by_pred["count"])),
    }
    write_json(out_dir / "metrics.json", payload)
    return payload


def write_negative_results(out_dir: Path, summaries: dict) -> dict:
    lines = [
        "# Negative and Boundary Result Analysis",
        "",
        "This document records boundaries of applicability and prevents overclaiming.",
        "",
        "## Observed cases",
        "",
    ]
    e2 = summaries.get("e2_sanity", {})
    if e2 and e2.get("generating_element_macro_f1") == e2.get("rule_based_macro_f1"):
        lines.append("- Rule-based baseline equals the generating-element model in the binary sanity task; this task is treated only as pipeline verification.")
    e5 = summaries.get("e5_alpha_sweep", {})
    if e5 and e5.get("best_alpha") in (0.0, 1.0):
        lines.append("- Best alpha is a boundary value; this is interpreted as evidence that a fixed generating element suffices for the current task.")
    lines.append("- alpha = 0.5 is a degenerate control point and is not used as an adaptation candidate.")
    lines.append("- No result is interpreted as proof of cryptographic security.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "negative_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"negative_result_notes": lines}
    write_json(out_dir / "metrics.json", payload)
    return payload


def write_experiment_summary(results: Path, summaries: dict) -> None:
    e2 = summaries["e2_sanity"]
    e3 = summaries["e3_multiclass"]
    e4 = summaries["e4_ranking"]["ranking_metrics"]
    e5 = summaries["e5_alpha_sweep"]
    e7 = summaries["e7_monte_carlo"]["sample_size_sweep"]
    e9 = summaries["e9_real_data"]
    lines = [
        "# Experiment Summary",
        "",
        "## Corpus",
        "",
        f"- Operation rows: {summaries['e0_data_audit']['operation_rows']}",
        f"- Reconstructed real CET rows: {summaries['e0_data_audit']['real_reconstructed_rows']}",
        f"- Base operation rows: {summaries['e0_data_audit']['base_rows']}",
        "",
        "## Classification",
        "",
        "| Experiment | GE macro-F1 | Rule macro-F1 | Centroid macro-F1 | alpha_opt |",
        "|---|---:|---:|---:|---:|",
        f"| E2 sanity | {e2['generating_element_macro_f1']:.4f} | {e2['rule_based_macro_f1']:.4f} | {e2['nearest_centroid_macro_f1']:.4f} | {e2['best_alpha']:.2f} |",
        f"| E3 multiclass | {e3['generating_element_macro_f1']:.4f} | {e3['rule_based_macro_f1']:.4f} | {e3['nearest_centroid_macro_f1']:.4f} | {e3['best_alpha']:.2f} |",
        "",
        "## Ranking",
        "",
        "| Method | Spearman | Top-25 overlap | Top-50 overlap |",
        "|---|---:|---:|---:|",
    ]
    for row in e4:
        lines.append(f"| {row['method']} | {row['spearman_vs_composite']:.4f} | {row['top25_overlap']:.2f} | {row['top50_overlap']:.2f} |")
    lines.extend(
        [
            "",
            "## Alpha Sweep",
            "",
            f"- Best non-degenerate alpha: {e5['best_alpha']:.2f}",
            f"- Best validation macro-F1: {e5['best_macro_f1']:.4f}",
            "- alpha = 0.50 is a degenerate control point, not a candidate.",
            "",
            "## Monte Carlo",
            "",
            "| n | Mean macro-F1 | Std |",
            "|---:|---:|---:|",
        ]
    )
    for row in e7:
        lines.append(f"| {row['n']} | {row['mean_macro_f1']:.4f} | {row['std_macro_f1']:.4f} |")
    lines.extend(
        [
            "",
            "## Real-Data Validation",
            "",
            f"- R1 real reconstructed rows: {e9['real_rows']}",
            f"- Base rows: {e9['base_rows']}",
            f"- R2 status: {e9['r2_public_sources_status']}",
            f"- R3 status: {e9['r3_author_data_status']}",
            "",
            "## Boundary Interpretation",
            "",
            "- Binary sanity is pipeline verification because rule-based baseline equals the GE model.",
            "- Pure reconstruction rank is not a standalone cryptographic ranking.",
            "- No result is a proof of cryptographic security.",
        ]
    )
    (results / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_experiments(root: Path, seed: int = 20260617) -> dict:
    results = root / "results"
    operation_df, feature_df, feature_cols = build_feature_tables(seed=seed)
    corpus_dir = root / "data" / "corpus_v1"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    operation_df.to_csv(corpus_dir / "operation_corpus.csv", index=False)
    feature_df.to_csv(corpus_dir / "feature_matrix.csv", index=False)
    manifest = write_corpus_manifest(operation_df, feature_df, results / "corpus_manifest.json")
    write_json(results / "e0_data_audit" / "metrics.json", manifest)
    feature_diag = feature_diagnostics(feature_df, feature_cols, results / "e1_feature_validation")
    summaries = {
        "e0_data_audit": manifest,
        "e1_feature_validation": feature_diag,
    }
    summaries["e2_sanity"] = run_classification(feature_df, feature_cols, binary_labels(feature_df), results / "e2_sanity", "binary", seed)
    summaries["e3_multiclass"] = run_classification(feature_df, feature_cols, multiclass_labels(feature_df), results / "e3_multiclass", "multiclass", seed)
    summaries["e4_ranking"] = run_ranking(feature_df, feature_cols, results / "e4_ranking", seed)
    summaries["e5_alpha_sweep"] = run_alpha_sweep(feature_df, feature_cols, results / "e5_alpha_sweep", seed)
    summaries["e6_ablation"] = run_ablation(feature_df, feature_cols, results / "e6_ablation", seed)
    summaries["e7_monte_carlo"] = run_monte_carlo(feature_df, feature_cols, results / "e7_monte_carlo")
    summaries["e8_robustness"] = run_robustness(feature_df, feature_cols, results / "e8_robustness")
    summaries["e9_real_data"] = run_real_data_validation(feature_df, feature_cols, results / "e9_real_data", seed)
    summaries["e10_negative_results"] = write_negative_results(results / "e10_negative_results", summaries)
    write_experiment_summary(results, summaries)
    index = {
        "experiments": {
            key: str((results / key).relative_to(root))
            for key in summaries
        },
        "summary_markdown": "results/experiment_summary.md",
        "corpus_manifest": "results/corpus_manifest.json",
        "corpus_v1": "data/corpus_v1",
        "summary_metrics": summaries,
    }
    write_json(results / "experiment_index.json", index)
    return index
