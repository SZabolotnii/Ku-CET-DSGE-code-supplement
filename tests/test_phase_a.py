import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cetspace.labels import multiclass_labels, rule_based_multiclass_predict
from cetspace.operations import mapping_from_rule
from cetspace.phase_a import (
    is_decision_column,
    nonlinearity,
    representational_columns,
)
from cetspace.pipeline import build_feature_tables


def test_nonlinearity_linear_map_is_zero():
    # Identity and any purely linear/affine map has nonlinearity 0.
    identity = tuple(range(8))
    assert nonlinearity(identity) == 0
    linear = mapping_from_rule(lambda b: (b[0] ^ b[1], b[1] ^ b[2], b[2]))
    assert nonlinearity(linear) == 0


def test_nonlinearity_is_in_valid_range_for_n3():
    # For 3-bit functions the achievable nonlinearity is 0..2.
    _, feature_df, _ = build_feature_tables()
    # a non-linear bent-ish 3-bit S-box should reach the max of 2 somewhere
    vals = set()
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = tuple(int(v) for v in rng.permutation(8))
        nl = nonlinearity(m)
        assert 0 <= nl <= 2
        vals.add(nl)
    assert max(vals) == 2  # the corpus space contains maximally-nonlinear maps


def test_leakage_rule_is_label_generator():
    # The rule-based multiclass baseline must reproduce the labels exactly:
    # that is the leakage the Phase-A audit quantifies.
    _, feature_df, _ = build_feature_tables()
    labels = multiclass_labels(feature_df)
    rule = rule_based_multiclass_predict(feature_df.to_dict("records"))
    assert np.mean(rule == labels) == 1.0


def test_representational_excludes_decision_scalars():
    _, _, feature_cols = build_feature_tables()
    rep = representational_columns(feature_cols)
    for leaked in ["avalanche_error", "max_anf_degree", "complexity_score", "truth_0", "balance_error"]:
        assert leaked not in rep
    assert all(not is_decision_column(c) for c in rep)
    assert len(rep) >= 5  # a non-empty structural representation survives
