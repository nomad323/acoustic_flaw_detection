from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, lfilter

from hcalculate import accumulate_shifted_scaled
from pcalculate import pcalculate


@lru_cache(maxsize=8)
def _bandpass_coeffs(
    fs: float = 1e9,
    fc: float = 2.5e6,
    bw: float = 0.5e6,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    wn = [(fc - bw / 2) / (fs / 2), (fc + bw / 2) / (fs / 2)]
    b, a_filter = butter(order, wn, btype="bandpass")
    return b, a_filter


def my_stack(
    x: float,
    y: float,
    x1: float,
    x2: float,
    a: float,
    lam: float,
    c: float,
    t0: float,
    data: np.ndarray,
    n: int,
    d: float,
    apply_filter: bool = False,
    filter_coeffs: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    length = data.shape[2]
    result = np.zeros(length, dtype=np.float64)

    x1_array = x1 + np.arange(n) * d
    x2_array = x2 + np.arange(n) * d

    for i in range(n):
        xi = float(x1_array[i])
        for j in range(n):
            xj = float(x2_array[j])
            p1 = pcalculate(x, y, xi, lam, a)
            p2 = pcalculate(x, y, xj, lam, a)
            d_x1 = np.linalg.norm(np.array([xi, 0.0]) - np.array([x, y]))
            d_x2 = np.linalg.norm(np.array([xj, 0.0]) - np.array([x, y]))
            k = (p1 * p2 * a * a) / (d_x1 * d_x2)
            if k == 0:
                continue
            x0 = int(np.round(((d_x1 + d_x2) / c) / t0))
            accumulate_shifted_scaled(result, data[i, j, :], x0, 1.0 / k)

    if apply_filter:
        if filter_coeffs is None:
            b, a_filter = _bandpass_coeffs()
        else:
            b, a_filter = filter_coeffs
        result = lfilter(b, a_filter, result)
    return result
