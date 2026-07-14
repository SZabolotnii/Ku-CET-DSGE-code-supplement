from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PLOT_FEATURES = [
    "avalanche_error",
    "balance_error",
    "differential_uniformity",
    "max_anf_degree",
    "complexity_score",
    "effective_substitution_tables",
]


def dataframe_hash(df: pd.DataFrame) -> str:
    payload = df.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_corpus_manifest(operation_df: pd.DataFrame, feature_df: pd.DataFrame, out: Path) -> dict:
    manifest = {
        "operation_rows": int(len(operation_df)),
        "feature_rows": int(len(feature_df)),
        "real_reconstructed_rows": int((feature_df["is_real_corpus"] == True).sum()),
        "base_rows": int((feature_df["is_base"] == True).sum()),
        "operation_hash": dataframe_hash(operation_df),
        "feature_hash": dataframe_hash(feature_df),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def feature_diagnostics(feature_df: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> dict:
    x = feature_df[feature_cols].astype(float)
    missing = x.isna().sum()
    inf_counts = np.isinf(x.to_numpy()).sum(axis=0)
    nunique = x.nunique()
    constant_cols = sorted(nunique[nunique <= 1].index.tolist())
    corr = x.corr(numeric_only=True).fillna(0.0)
    cond_before = float(np.linalg.cond(x.to_numpy() + 1e-12))
    pruned_cols = [c for c in feature_cols if c not in constant_cols]
    cond_after = float(np.linalg.cond(x[pruned_cols].to_numpy() + 1e-12)) if pruned_cols else float("inf")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature": feature_cols,
            "missing": [int(missing[c]) for c in feature_cols],
            "inf": [int(inf_counts[i]) for i, _ in enumerate(feature_cols)],
            "nunique": [int(nunique[c]) for c in feature_cols],
            "is_constant": [c in constant_cols for c in feature_cols],
        }
    ).to_csv(out_dir / "feature_diagnostics.csv", index=False)
    corr.to_csv(out_dir / "feature_correlation.csv")
    distribution_rows = []
    if "quality_label" in feature_df.columns:
        plotted = [c for c in PLOT_FEATURES if c in feature_df.columns]
        grouped = feature_df.groupby("quality_label")
        for label, group in grouped:
            for col in plotted:
                values = group[col].astype(float)
                distribution_rows.append(
                    {
                        "class": label,
                        "feature": col,
                        "count": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "min": float(values.min()),
                        "q25": float(values.quantile(0.25)),
                        "median": float(values.median()),
                        "q75": float(values.quantile(0.75)),
                        "max": float(values.max()),
                    }
                )
        pd.DataFrame(distribution_rows).to_csv(out_dir / "feature_distribution_by_class.csv", index=False)
        _write_distribution_svg(pd.DataFrame(distribution_rows), out_dir / "feature_distribution_by_class.svg")
    summary = {
        "n_features": len(feature_cols),
        "constant_columns": constant_cols,
        "has_nan": bool(missing.sum() > 0),
        "has_inf": bool(np.sum(inf_counts) > 0),
        "condition_before_pruning": cond_before,
        "condition_after_pruning": cond_after,
        "feature_distribution_by_class": bool(distribution_rows),
    }
    (out_dir / "feature_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _write_distribution_svg(distribution_df: pd.DataFrame, out: Path) -> None:
    if distribution_df.empty:
        return
    features = distribution_df["feature"].drop_duplicates().tolist()
    classes = distribution_df["class"].drop_duplicates().tolist()
    width = 980
    row_h = 34
    height = 80 + row_h * len(distribution_df)
    label_w = 260
    plot_w = width - label_w - 60
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="Arial" font-size="18" font-weight="700">Feature distributions by class</text>',
        '<text x="20" y="55" font-family="Arial" font-size="12" fill="#555">Bars show normalized class-wise means for selected diagnostic features.</text>',
    ]
    y = 84
    palette = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]
    for feature_i, feature in enumerate(features):
        feature_rows = distribution_df[distribution_df["feature"] == feature]
        min_mean = float(feature_rows["mean"].min())
        max_mean = float(feature_rows["mean"].max())
        span = max(max_mean - min_mean, 1e-12)
        lines.append(f'<text x="20" y="{y - 8}" font-family="Arial" font-size="13" font-weight="700">{feature}</text>')
        for class_i, cls in enumerate(classes):
            row = feature_rows[feature_rows["class"] == cls]
            if row.empty:
                continue
            mean = float(row.iloc[0]["mean"])
            norm = (mean - min_mean) / span
            bar_w = max(2.0, norm * plot_w)
            color = palette[class_i % len(palette)]
            lines.append(f'<text x="38" y="{y + 14}" font-family="Arial" font-size="11" fill="#333">{cls}</text>')
            lines.append(f'<rect x="{label_w}" y="{y}" width="{bar_w:.2f}" height="18" fill="{color}" opacity="0.78"/>')
            lines.append(f'<text x="{label_w + bar_w + 6:.2f}" y="{y + 14}" font-family="Arial" font-size="11" fill="#333">{mean:.4g}</text>')
            y += row_h
        y += 10
    lines.append("</svg>")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
