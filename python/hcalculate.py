from __future__ import annotations

import numpy as np

from pcalculate import pcalculate

try:
    from numba import njit  # pyright: ignore[reportMissingImports]
except ImportError:
    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def accumulate_shifted_scaled(result: np.ndarray, signal: np.ndarray, x0: int, scale: float) -> None:
    length = signal.size
    if x0 >= length:
        return
    out_len = length - x0
    for idx in range(out_len):
        result[idx] += signal[idx + x0] * scale


def hcalculate(
    x: float,
    y: float,
    x1: float,
    x2: float,
    a: float,
    lam: float,
    m: np.ndarray,
    c: float,
    t0: float,
) -> np.ndarray:
    p1 = pcalculate(x, y, x1, lam, a)
    p2 = pcalculate(x, y, x2, lam, a)
    d_x1 = np.linalg.norm(np.array([x1, 0.0]) - np.array([x, y]))
    d_x2 = np.linalg.norm(np.array([x2, 0.0]) - np.array([x, y]))
    k = (p1 * p2 * a * a) / (d_x1 * d_x2)

    x0 = int(np.round(((d_x1 + d_x2) / c) / t0))
    m = np.asarray(m).reshape(-1)
    length = m.size
    h = np.zeros(length, dtype=np.float64)

    if x0 < length and k != 0:
        h[: length - x0] = m[x0:] / k
    return h
