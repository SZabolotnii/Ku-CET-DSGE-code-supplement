from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cetspace.diagnostics import dataframe_hash, feature_diagnostics
from cetspace.experiments import run_classification, run_ranking
from cetspace.labels import composite_quality_score, multiclass_labels
from cetspace.models import select_alpha
from cetspace.pipeline import build_feature_tables


def test_corpus_hash_is_stable():
    ops_a, features_a, _ = build_feature_tables(seed=20260617)
    ops_b, features_b, _ = build_feature_tables(seed=20260617)
    assert dataframe_hash(ops_a) == dataframe_hash(ops_b)
    assert dataframe_hash(features_a) == dataframe_hash(features_b)


def test_multiclass_labels_have_expected_classes():
    _, features, _ = build_feature_tables(seed=20260617)
    labels = set(multiclass_labels(features).tolist())
    assert "base_good" in labels
    assert "non_bijective_weak" in labels
    assert "derived_good" in labels


def test_alpha_half_is_excluded_from_optimum_when_tied():
    _, features, cols = build_feature_tables(seed=20260617)
    x = features[cols].astype(float).to_numpy()
    y = features["quality_label"].to_numpy()
    best_alpha, rows = select_alpha(x[:600], y[:600], x[600:800], y[600:800])
    assert best_alpha != 0.5
    assert any(row["alpha"] == 0.5 and row["is_degenerate_alpha"] for row in rows)


def test_feature_diagnostics_outputs(tmp_path):
    _, features, cols = build_feature_tables(seed=20260617)
    summary = feature_diagnostics(features, cols, tmp_path)
    assert not summary["has_nan"]
    assert not summary["has_inf"]
    assert summary["feature_distribution_by_class"]
    assert (tmp_path / "feature_diagnostics.csv").exists()
    assert (tmp_path / "feature_distribution_by_class.csv").exists()
    assert (tmp_path / "feature_distribution_by_class.svg").exists()


def test_ranking_score_is_deterministic(tmp_path):
    _, features, cols = build_feature_tables(seed=20260617)
    small = features.sample(n=120, random_state=1).reset_index(drop=True)
    first = run_ranking(small, cols, tmp_path / "first", seed=20260617)
    second = run_ranking(small, cols, tmp_path / "second", seed=20260617)
    assert first["ranking_metrics"] == second["ranking_metrics"]
    scores = composite_quality_score(small)
    assert np.isfinite(scores).all()


def test_mini_classification_run(tmp_path):
    _, features, cols = build_feature_tables(seed=20260617)
    small = features.groupby("quality_label", group_keys=False).head(30).reset_index(drop=True)
    metrics = run_classification(small, cols, small["quality_label"].to_numpy(), tmp_path, "binary")
    assert "generating_element_macro_f1" in metrics
    assert (tmp_path / "metrics.json").exists()
