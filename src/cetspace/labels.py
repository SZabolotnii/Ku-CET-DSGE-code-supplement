from __future__ import annotations

import numpy as np
import pandas as pd


def binary_labels(df: pd.DataFrame) -> np.ndarray:
    return df["quality_label"].to_numpy()


def multiclass_labels(df: pd.DataFrame) -> np.ndarray:
    labels: list[str] = []
    for _, row in df.iterrows():
        if bool(row["is_base"]) and bool(row["is_bijective"]) and row["avalanche_error"] <= 0.25:
            labels.append("base_good")
        elif not bool(row["is_bijective"]):
            labels.append("non_bijective_weak")
        elif row["max_anf_degree"] <= 1 and row["avalanche_error"] >= 0.45:
            labels.append("linear_weak")
        elif 0.24 < row["avalanche_error"] < 0.40:
            labels.append("borderline_avalanche")
        elif row["complexity_score"] >= 12.0 and row["avalanche_error"] <= 0.25:
            labels.append("high_complexity_good")
        elif row["complexity_score"] >= 12.0:
            labels.append("high_complexity_weak")
        elif row["avalanche_error"] <= 0.25 and row["balance_error"] <= 0.05:
            labels.append("derived_good")
        else:
            labels.append("borderline_avalanche")
    return np.array(labels)


def composite_quality_score(df: pd.DataFrame) -> np.ndarray:
    avalanche = 1.0 - _normalize(df["avalanche_error"].to_numpy(dtype=float))
    balance = 1.0 - _normalize(df["balance_error"].to_numpy(dtype=float))
    bijective = df["is_bijective"].astype(float).to_numpy()
    anf = _normalize(df["max_anf_degree"].to_numpy(dtype=float))
    differential = 1.0 - _normalize(df["differential_uniformity"].to_numpy(dtype=float))
    complexity = _normalize(df["complexity_score"].to_numpy(dtype=float))
    return (
        0.30 * avalanche
        + 0.15 * balance
        + 0.20 * bijective
        + 0.15 * anf
        + 0.10 * differential
        - 0.10 * complexity
    )


def rule_based_multiclass_predict(records: list[dict]) -> np.ndarray:
    return multiclass_labels(pd.DataFrame(records))


def _normalize(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi == lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)
