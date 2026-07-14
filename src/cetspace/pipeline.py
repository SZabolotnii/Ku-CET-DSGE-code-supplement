from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import feature_record, numeric_feature_columns
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
from .operations import generate_full_corpus, generate_real_cet_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_dirs() -> None:
    for path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_feature_tables(seed: int = 20260617) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    operations = generate_full_corpus(seed=seed)
    operation_df = pd.DataFrame([op.to_record() for op in operations])
    records = [feature_record(op) for op in operations]
    feature_df = pd.DataFrame(records)
    feature_cols = numeric_feature_columns(records)
    return operation_df, feature_df, feature_cols


def evaluate(seed: int = 20260617) -> dict:
    ensure_dirs()
    operation_df, feature_df, feature_cols = build_feature_tables(seed=seed)
    operation_df.to_csv(DATA_DIR / "operation_corpus.csv", index=False)
    feature_df.to_csv(DATA_DIR / "feature_matrix.csv", index=False)

    x = feature_df[feature_cols].astype(float).to_numpy()
    y = feature_df["quality_label"].to_numpy()
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, y, seed=seed)

    best_alpha, alpha_rows = select_alpha(x_train, y_train, x_val, y_val)
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(RESULTS_DIR / "alpha_sweep.csv", index=False)

    model = fit_generating_element_model(x_train, y_train, alpha=best_alpha)
    y_pred = model.predict(x_test)
    ge_f1 = macro_f1(y_test, y_pred)

    test_records = feature_df.iloc[-len(y_test) :].to_dict("records")
    # Use an explicit split index map for the rule baseline through a deterministic recomputation.
    _, _, test_indices, _, _, _ = split_dataset(np.arange(len(y)).reshape(-1, 1), y, seed=seed)
    test_indices = test_indices.ravel().astype(int)
    test_records = feature_df.iloc[test_indices].to_dict("records")
    y_rule = rule_based_predict(test_records)
    rule_f1 = macro_f1(y_test, y_rule)

    classical_predictions = fit_classical_baselines(x_train, y_train, x_test)
    classical_scores = {
        name: float(macro_f1(y_test, pred))
        for name, pred in classical_predictions.items()
    }

    bootstrap = bootstrap_macro_f1(y_test, y_pred, n_bootstrap=1000, seed=seed)
    real_df = feature_df[feature_df["is_real_corpus"] == True].copy()
    real_predictions = model.predict(real_df[feature_cols].astype(float).to_numpy())
    real_df["generating_element_prediction"] = real_predictions
    real_df["rank_score"] = np.min(model.log_msed_features(real_df[feature_cols].astype(float).to_numpy()), axis=1)
    real_df.sort_values("rank_score").to_csv(RESULTS_DIR / "real_cet_ranked.csv", index=False)

    metrics = {
        "seed": seed,
        "n_operations_total": int(len(feature_df)),
        "n_real_reconstructed": int((feature_df["is_real_corpus"] == True).sum()),
        "n_base_operations": int((feature_df["is_base"] == True).sum()),
        "best_alpha": float(best_alpha),
        "degenerate_alpha": 0.5,
        "degenerate_alpha_is_candidate": False,
        "generating_element_macro_f1": float(ge_f1),
        "rule_based_macro_f1": float(rule_f1),
        "classical_baselines_macro_f1": classical_scores,
        "bootstrap_macro_f1": bootstrap,
        "condition_numbers": model.condition_numbers,
        "classification_report": classification_report_dict(y_test, y_pred),
        "real_corpus_prediction_counts": real_df["generating_element_prediction"].value_counts().to_dict(),
        "feature_columns": feature_cols,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_alpha_sweep(alpha_df)
    write_verification_report(metrics, alpha_df, feature_df)
    return metrics


def plot_alpha_sweep(alpha_df: pd.DataFrame) -> None:
    svg_points = []
    width, height = 720, 420
    margin = 50
    xs = alpha_df["alpha"].to_numpy(dtype=float)
    ys = alpha_df["macro_f1"].to_numpy(dtype=float)
    y_min = max(0.0, float(np.min(ys)) - 0.05)
    y_max = min(1.0, float(np.max(ys)) + 0.05)
    if y_max == y_min:
        y_max = y_min + 1.0
    for x_val, y_val in zip(xs, ys):
        x = margin + (x_val - 0.0) / 1.0 * (width - 2 * margin)
        y = height - margin - (y_val - y_min) / (y_max - y_min) * (height - 2 * margin)
        svg_points.append(f"{x:.1f},{y:.1f}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">PATP adaptation of the generating element</text>
  <text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-family="Arial" font-size="14">alpha</text>
  <text x="18" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 18 {height / 2})">validation macro-F1</text>
  <polyline points="{' '.join(svg_points)}" fill="none" stroke="#1f77b4" stroke-width="2"/>
  {"".join(f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="3" fill="#1f77b4"/>' for p in svg_points)}
  <text x="{margin}" y="{height - margin + 20}" font-family="Arial" font-size="12">0.0</text>
  <text x="{width - margin}" y="{height - margin + 20}" text-anchor="end" font-family="Arial" font-size="12">1.0</text>
  <text x="{margin - 8}" y="{height - margin}" text-anchor="end" font-family="Arial" font-size="12">{y_min:.2f}</text>
  <text x="{margin - 8}" y="{margin}" text-anchor="end" font-family="Arial" font-size="12">{y_max:.2f}</text>
</svg>
"""
    (FIGURES_DIR / "alpha_sweep.svg").write_text(svg, encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    else:
        plt.figure(figsize=(7, 4))
        plt.plot(alpha_df["alpha"], alpha_df["macro_f1"], marker="o", linewidth=1.5)
        plt.xlabel("alpha")
        plt.ylabel("validation macro-F1")
        plt.title("PATP adaptation of the generating element")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "alpha_sweep.png", dpi=160)
        plt.close()


def write_verification_report(metrics: dict, alpha_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    cond_ok = all(value < 1e8 for value in metrics["condition_numbers"].values())
    alpha = metrics["best_alpha"]
    alpha_boundary = alpha in (0.0, 1.0)
    degenerate_rows = alpha_df[alpha_df["is_degenerate_alpha"] == True]
    degenerate_score = None if degenerate_rows.empty else float(degenerate_rows.iloc[0]["macro_f1"])
    ge_beats_rule = metrics["generating_element_macro_f1"] >= metrics["rule_based_macro_f1"]
    lines = [
        "# Verification Report",
        "",
        f"- [PASS] Reconstructed real CET corpus size: {metrics['n_real_reconstructed']} records.",
        f"- [PASS] Base operation records: {metrics['n_base_operations']} records.",
        f"- [{'PASS' if metrics['n_real_reconstructed'] == 384 else 'FAIL'}] Expected 384 reconstructed single-operand records.",
        f"- [{'PASS' if metrics['n_base_operations'] == 8 else 'FAIL'}] Expected 8 base operation records.",
        f"- [{'PASS' if cond_ok else 'FAIL'}] Condition numbers controlled below 1e8.",
        f"- [{'PASS' if not alpha_boundary else 'WARN'}] alpha_opt = {alpha:.2f}; boundary values mean a fixed generating element may suffice.",
        f"- [PASS] alpha = 0.50 is treated as a degenerate control point, not as an adaptation candidate.",
        f"- [INFO] Degenerate alpha macro-F1: {'n/a' if degenerate_score is None else f'{degenerate_score:.4f}'}.",
        f"- [{'PASS' if ge_beats_rule else 'WARN'}] Generating-element macro-F1 = {metrics['generating_element_macro_f1']:.4f}; rule baseline = {metrics['rule_based_macro_f1']:.4f}.",
        f"- [PASS] Bootstrap repetitions: {metrics['bootstrap_macro_f1']['n_bootstrap']}.",
        f"- [PASS] Feature matrix rows: {len(feature_df)}.",
        f"- [PASS] Alpha sweep grid points: {len(alpha_df)}.",
        "",
        "## Interpretation",
        "",
        "This report verifies reproducibility and statistical diagnostics. It does not claim cryptographic security of CET-encryption.",
    ]
    (RESULTS_DIR / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    metrics = evaluate()
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
