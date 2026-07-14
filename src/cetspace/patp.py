from __future__ import annotations

import numpy as np


def patp_power(i: int, alpha: float) -> float:
    """Return the PATP exponent p_i(alpha)."""
    if i < 2:
        raise ValueError("PATP basis index i must be >= 2")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    a = 1.0 / i
    b = 4.0 - i - 3.0 / i
    c = 2.0 * i - 4.0 + 2.0 / i
    return a + b * alpha + c * alpha**2


def patp_transform(x: np.ndarray, alpha: float, order: int = 3) -> np.ndarray:
    """Expand normalized features with a sign-preserving PATP basis."""
    if order < 1:
        raise ValueError("order must be positive")
    x = np.asarray(x, dtype=float)
    parts = [x]
    for i in range(2, order + 2):
        p = patp_power(i, alpha)
        parts.append(np.sign(x) * np.abs(x) ** p)
    return np.concatenate(parts, axis=1)


def is_degenerate_alpha(alpha: float, tol: float = 1e-12) -> bool:
    """Return True when PATP collapses to repeated linear powers."""
    return abs(alpha - 0.5) <= tol


def alpha_grid(step: float = 0.05) -> np.ndarray:
    """Return the standard alpha grid used in the article experiments."""
    if step <= 0:
        raise ValueError("step must be positive")
    values = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    return values[(values >= 0.0) & (values <= 1.0)]
