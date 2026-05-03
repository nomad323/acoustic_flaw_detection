from __future__ import annotations

import numpy as np
from scipy.special import j1


def pcalculate(x: float, y: float, x1: float, lam: float, r: float) -> float:
    # x, y are focus point; x1 is probe center; lam is wavelength; r is radius.
    r1 = np.linalg.norm(np.array([x, y]) - np.array([x1, 0.0]))
    delta = abs(x1 - x)
    if r1 == 0:
        return 1.0

    m = 2.0 * np.pi * r * delta / (r1 * lam)
    if m != 0:
        return float(2.0 * j1(m) / m)
    return 1.0
