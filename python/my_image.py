from __future__ import annotations

import math
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from mgb import default_mgb_coefficients, validate_mgb_coefficients
from my_stack import _bandpass_coeffs, my_stack

try:
    from numba import njit, prange  # pyright: ignore[reportMissingImports]

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

    def prange(*args):  # type: ignore[misc]
        return range(*args)


@njit(cache=True, fastmath=True)
def _j1_approx(x: float) -> float:
    ax = abs(x)
    if ax < 8.0:
        y = x * x
        ans1 = x * (
            72362614232.0
            + y
            * (
                -7895059235.0
                + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))
            )
        )
        ans2 = 144725228442.0 + y * (
            2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))
        )
        return ans1 / ans2

    z = 8.0 / ax
    y = z * z
    xx = ax - 2.356194491
    ans1 = 1.0 + y * (-0.001098628627 + y * (2.734510407e-05 + y * (-2.073370639e-06 + y * 2.093887211e-07)))
    ans2 = -0.01562499995 + y * (
        0.0001430488765 + y * (-6.911147651e-06 + y * (7.621095161e-07 + y * (-9.34945152e-08)))
    )
    ans = math.sqrt(0.636619772 / ax) * (math.cos(xx) * ans1 - z * math.sin(xx) * ans2)
    if x < 0.0:
        return -ans
    return ans


@njit(cache=True, fastmath=True)
def _pcalculate_numba(x: float, y: float, x1: float, lam: float, r: float) -> float:
    dx = x - x1
    r1 = math.sqrt(dx * dx + y * y)
    if r1 == 0.0:
        return 1.0

    delta = abs(dx)
    m = 2.0 * math.pi * r * delta / (r1 * lam)
    if m == 0.0:
        return 1.0
    return 2.0 * _j1_approx(m) / m


@njit(cache=True, fastmath=True)
def _mgb_gain_numba(r2: float, mgb_a: np.ndarray, mgb_b: np.ndarray) -> float:
    gain = 0.0
    for k in range(mgb_a.size):
        gain += mgb_a[k] * math.exp(-mgb_b[k] * r2)
    if gain <= 1e-12:
        return 1e-12
    return gain


@njit(cache=True, fastmath=True, parallel=True)
def _compute_z_numba(
    data: np.ndarray,
    x1: float,
    x2: float,
    a: float,
    lam: float,
    c: float,
    t0: float,
    n: int,
    l0: float,
    delta: float,
    nx: int,
    ny: int,
    subset_len: int,
    use_fan_mask: bool,
    fan_origin_x: float,
    fan_origin_y: float,
    fan_half_angle_rad: float,
    use_mgb: bool,
    mgb_a: np.ndarray,
    mgb_b: np.ndarray,
) -> np.ndarray:
    z = np.full((nx, ny), -1.0, dtype=np.float64)
    total_pixels = nx * ny
    data_len = data.shape[2]

    for idx in prange(total_pixels):
        l = idx // ny
        m = idx - l * ny
        px = (l + 1) * delta
        py = (m + 1) * delta

        if use_fan_mask:
            dy = py - fan_origin_y
            if dy <= 0.0:
                continue
            angle = abs(math.atan2(px - fan_origin_x, dy))
            if angle > fan_half_angle_rad:
                continue
        result = np.zeros(subset_len, dtype=np.float64)
        for i in range(n):
            xi = x1 + i * l0
            for j in range(n):
                xj = x2 + j * l0
                d_x1 = math.sqrt((xi - px) * (xi - px) + py * py)
                d_x2 = math.sqrt((xj - px) * (xj - px) + py * py)

                if use_mgb:
                    p1 = _mgb_gain_numba((xi - px) * (xi - px) + py * py, mgb_a, mgb_b)
                    p2 = _mgb_gain_numba((xj - px) * (xj - px) + py * py, mgb_a, mgb_b)
                else:
                    p1 = _pcalculate_numba(px, py, xi, lam, a)
                    p2 = _pcalculate_numba(px, py, xj, lam, a)
                k = (p1 * p2 * a * a) / (d_x1 * d_x2)
                if k == 0.0:
                    continue

                x0 = int(round(((d_x1 + d_x2) / c) / t0))
                inv_k = 1.0 / k
                for t in range(subset_len):
                    src_idx = t + x0
                    if src_idx < data_len:
                        result[t] += data[i, j, src_idx] * inv_k

        max_abs = 0.0
        for t in range(subset_len):
            val = abs(result[t])
            if val > max_abs:
                max_abs = val
        z[l, m] = max_abs
    return z


def my_image(
    x1: float,
    x2: float,
    a: float,
    lam: float,
    c: float,
    t0: float,
    data: np.ndarray,
    n: int,
    l0: float,
    delta: float = 1e-3,
    length: float = 0.1,
    width: float = 0.06,
    subset_len: int = 600,
    show: bool = True,
    log_progress: bool = True,
    progress_every: int = 50,
    apply_filter: bool = False,
    use_numba: bool = True,
    use_fan_mask: bool = True,
    fan_half_angle_deg: float = 35.0,
    fan_origin_x: float | None = None,
    fan_origin_y: float = 0.0,
    beam_model: str = "mgb",
    mgb_a: np.ndarray | None = None,
    mgb_b: np.ndarray | None = None,
) -> np.ndarray:
    t_start = perf_counter()
    data = np.asarray(data, dtype=np.float64, order="C")
    subset_len = min(subset_len, data.shape[2])
    nx = max(1, int(round(length / delta)))
    ny = max(1, int(round(width / delta)))

    if fan_origin_x is None:
        tx_center = x1 + 0.5 * (n - 1) * l0
        rx_center = x2 + 0.5 * (n - 1) * l0
        fan_origin_x = 0.5 * (tx_center + rx_center)
    fan_half_angle_rad = math.radians(fan_half_angle_deg)
    beam_model = beam_model.lower().strip()
    if beam_model not in {"mgb", "legacy"}:
        raise ValueError("beam_model must be 'mgb' or 'legacy'.")
    use_mgb = beam_model == "mgb"

    if mgb_a is None or mgb_b is None:
        default_a, default_b = default_mgb_coefficients()
        if mgb_a is None:
            mgb_a = default_a
        if mgb_b is None:
            mgb_b = default_b
    mgb_a = np.asarray(mgb_a, dtype=np.float64)
    mgb_b = np.asarray(mgb_b, dtype=np.float64)
    validate_mgb_coefficients(mgb_a, mgb_b)

    print(f"[my_image] beam_model={beam_model}, mgb_terms={mgb_a.size if use_mgb else 0}")

    if use_numba and (not apply_filter) and NUMBA_AVAILABLE:
        print("[my_image] using numba kernel")
        z = _compute_z_numba(
            data,
            x1,
            x2,
            a,
            lam,
            c,
            t0,
            n,
            l0,
            delta,
            nx,
            ny,
            subset_len,
            use_fan_mask,
            fan_origin_x,
            fan_origin_y,
            fan_half_angle_rad,
            use_mgb,
            mgb_a,
            mgb_b,
        )
    else:
        if use_numba and not NUMBA_AVAILABLE:
            print("[my_image] numba not installed, using python path")
        if apply_filter:
            print("[my_image] filter enabled; using python path")
        if use_mgb and apply_filter:
            print("[my_image] MGB + filter in python path can be very slow.")

        z = np.full((nx, ny), -1.0, dtype=np.float64)
        filter_coeffs = _bandpass_coeffs() if apply_filter else None

        for l in range(nx):
            xi = (l + 1) * delta
            for m in range(ny):
                yi = (m + 1) * delta
                if use_fan_mask:
                    dy = yi - fan_origin_y
                    if dy <= 0:
                        continue
                    angle = abs(math.atan2(xi - fan_origin_x, dy))
                    if angle > fan_half_angle_rad:
                        continue

                if use_mgb:
                    result = np.zeros(subset_len, dtype=np.float64)
                    for i in range(n):
                        tx = x1 + i * l0
                        for j in range(n):
                            rx = x2 + j * l0
                            d_x1 = math.sqrt((tx - xi) * (tx - xi) + yi * yi)
                            d_x2 = math.sqrt((rx - xi) * (rx - xi) + yi * yi)
                            p1 = float(np.sum(mgb_a * np.exp(-mgb_b * ((tx - xi) * (tx - xi) + yi * yi))))
                            p2 = float(np.sum(mgb_a * np.exp(-mgb_b * ((rx - xi) * (rx - xi) + yi * yi))))
                            k_gain = (max(p1, 1e-12) * max(p2, 1e-12) * a * a) / (d_x1 * d_x2)
                            if k_gain == 0:
                                continue
                            x0 = int(round(((d_x1 + d_x2) / c) / t0))
                            inv_k = 1.0 / k_gain
                            for t in range(subset_len):
                                src_idx = t + x0
                                if src_idx < data.shape[2]:
                                    result[t] += data[i, j, src_idx] * inv_k
                    z[l, m] = np.max(np.abs(result))
                else:
                    k = my_stack(
                        xi,
                        yi,
                        x1,
                        x2,
                        a,
                        lam,
                        c,
                        t0,
                        data,
                        n,
                        l0,
                        apply_filter=apply_filter,
                        filter_coeffs=filter_coeffs,
                    )
                    z[l, m] = np.max(np.abs(k[:subset_len]))
            if log_progress and ((l + 1) % progress_every == 0 or (l + 1) == nx):
                elapsed = perf_counter() - t_start
                rows_done = l + 1
                avg_per_row = elapsed / rows_done
                eta = avg_per_row * (nx - rows_done)
                print(f"[my_image] rows {rows_done}/{nx}, elapsed {elapsed:.1f}s, eta {eta:.1f}s")
    valid = z >= 0
    # print(z)
    if np.any(valid):
        a_log = np.zeros_like(z)
        a_log[valid] = np.log10(z[valid] + 1.0)
        low = np.min(a_log[valid])
        high = np.max(a_log[valid])
        denom = high - low
        a_norm = np.zeros_like(z)
        if denom != 0:
            a_norm[valid] = (a_log[valid] - low) / denom
    else:
        a_norm = np.zeros_like(z)

    if show:
        plt.figure()
        plt.imshow(np.flipud(a_norm.T), origin="lower", aspect="equal")
        plt.colorbar()
        plt.title("Normalized Imaging Result")
        plt.show()

    total = perf_counter() - t_start
    print(f"[my_image] total runtime: {total:.2f}s")
    return a_norm
