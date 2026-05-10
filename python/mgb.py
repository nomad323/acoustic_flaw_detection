from __future__ import annotations

import numpy as np

# Fast default basis for first-pass MGB correction.
# These are practical starting coefficients and can be tuned later.
DEFAULT_MGB_A = np.array([0.42, 0.33, 0.18, 0.07], dtype=np.float64)
DEFAULT_MGB_B = np.array([2200.0, 7800.0, 18000.0, 36000.0], dtype=np.float64)*0.000000000001


def default_mgb_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Return copies of default MGB basis coefficients."""
    return DEFAULT_MGB_A.copy(), DEFAULT_MGB_B.copy()


def validate_mgb_coefficients(mgb_a: np.ndarray, mgb_b: np.ndarray) -> None:
    if mgb_a.ndim != 1 or mgb_b.ndim != 1:
        raise ValueError("mgb_a and mgb_b must be 1D arrays.")
    if mgb_a.size == 0 or mgb_b.size == 0:
        raise ValueError("mgb_a and mgb_b must not be empty.")
    if mgb_a.size != mgb_b.size:
        raise ValueError("mgb_a and mgb_b must have the same length.")
    if np.any(mgb_b <= 0):
        raise ValueError("mgb_b values must be positive.")
