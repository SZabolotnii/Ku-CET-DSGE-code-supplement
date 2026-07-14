from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cetspace.pipeline import build_feature_tables


def test_feature_table_reproducibility():
    _, features_a, cols_a = build_feature_tables(seed=123)
    _, features_b, cols_b = build_feature_tables(seed=123)
    assert cols_a == cols_b
    assert features_a.equals(features_b)
    assert {"operation_id", "quality_label", "avalanche_error"}.issubset(features_a.columns)
