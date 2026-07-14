from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cetspace.patp import alpha_grid, is_degenerate_alpha, patp_power, patp_transform


def test_patp_special_cases():
    for i in range(2, 7):
        assert np.isclose(patp_power(i, 0.0), 1.0 / i)
        assert np.isclose(patp_power(i, 0.5), 1.0)
        assert np.isclose(patp_power(i, 1.0), float(i))


def test_patp_powers_are_positive_on_grid():
    for alpha in alpha_grid():
        for i in range(2, 5):
            assert patp_power(i, float(alpha)) > 0


def test_patp_transform_shape():
    x = np.ones((5, 4))
    transformed = patp_transform(x, alpha=0.5, order=3)
    assert transformed.shape == (5, 16)


def test_alpha_half_is_degenerate_control_point():
    assert is_degenerate_alpha(0.5)
    assert not is_degenerate_alpha(0.45)
