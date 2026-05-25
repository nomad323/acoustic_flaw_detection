from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, lfilter

from mgb import default_mgb_coefficients, validate_mgb_coefficients

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
def _legacy_gain_numba(raw_gain: float, use_abs: bool, min_gain: float) -> float:
    gain = abs(raw_gain) if use_abs else raw_gain
    if gain < min_gain:
        gain = min_gain
    return gain


@njit(cache=True, fastmath=True)
def _mgb_gain_numba(r2: float, mgb_a: np.ndarray, mgb_b: np.ndarray) -> float:
    gain = 0.0
    for k in range(mgb_a.size):
        gain += mgb_a[k] * math.exp(-mgb_b[k] * r2)
    if gain <= 1e-12:
        return 1e-12
    return gain


@njit(cache=True, fastmath=True)
def _mgb_angular_factor_numba(theta: float, mgb_angle_c: float) -> float:
    if mgb_angle_c <= 0.0:
        return 1.0
    return math.exp(-mgb_angle_c * theta * theta)


@njit(cache=True, fastmath=True)
def _piston_directivity_numba(theta: float, lam: float, radius: float) -> float:
    sin_theta = math.sin(theta)
    arg = (2.0 * math.pi / lam) * radius * sin_theta
    if abs(arg) < 1e-12:
        return 1.0
    return 2.0 * _j1_approx(arg) / arg


@njit(cache=True, fastmath=True)
def _piston_gain_numba(raw_gain: float, use_abs: bool, min_gain: float) -> float:
    gain = abs(raw_gain) if use_abs else raw_gain
    if gain < min_gain:
        gain = min_gain
    return gain


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


@njit(cache=True, fastmath=True)
def _trace_metric_numba(
    abs_trace: np.ndarray,
    reduction_mode: int,
    gate_center_idx: int,
    gate_half_width: int,
) -> float:
    length = abs_trace.size
    if length == 0:
        return 0.0

    if reduction_mode == 1:  # rms
        s = 0.0
        for t in range(length):
            v = abs_trace[t]
            s += v * v
        return math.sqrt(s / length)

    if reduction_mode == 2:  # gated_max
        start = gate_center_idx - gate_half_width
        end = gate_center_idx + gate_half_width + 1
        if start < 0:
            start = 0
        if end > length:
            end = length
        if start >= end:
            return 0.0
        m = 0.0
        for t in range(start, end):
            if abs_trace[t] > m:
                m = abs_trace[t]
        return m

    # max
    m = 0.0
    for t in range(length):
        if abs_trace[t] > m:
            m = abs_trace[t]
    return m


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
    correction_mode: int,
    piston_radius: float,
    attenuation_np_per_m: float,
    piston_use_abs: bool,
    piston_min_gain: float,
    tx_weights: np.ndarray,
    rx_weights: np.ndarray,
    legacy_use_abs: bool,
    legacy_min_gain: float,
    mgb_a: np.ndarray,
    mgb_b: np.ndarray,
    mgb_angle_c: float,
    use_cf: bool,
    coherence_gamma: float,
    reduction_mode: int,
    gate_center_idx: int,
    gate_half_width: int,
    use_sensitivity_comp: bool,
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
        result = np.zeros(subset_len, dtype=data.dtype)
        energy = np.zeros(subset_len, dtype=np.float64)
        counts = np.zeros(subset_len, dtype=np.float64)
        sens_denom = 0.0
        for i in range(n):
            xi = x1 + i * l0
            tx_w = tx_weights[i]
            for j in range(n):
                xj = x2 + j * l0
                pair_w = tx_w * rx_weights[j]
                d_x1 = math.sqrt((xi - px) * (xi - px) + py * py)
                d_x2 = math.sqrt((xj - px) * (xj - px) + py * py)
                theta1 = abs(math.atan2(xi - px, py))
                theta2 = abs(math.atan2(xj - px, py))

                if correction_mode == 2:
                    p1 = _mgb_gain_numba((xi - px) * (xi - px) + py * py, mgb_a, mgb_b) * _mgb_angular_factor_numba(
                        theta1, mgb_angle_c
                    )
                    p2 = _mgb_gain_numba((xj - px) * (xj - px) + py * py, mgb_a, mgb_b) * _mgb_angular_factor_numba(
                        theta2, mgb_angle_c
                    )
                    k = (p1 * p2 * a * a) / (d_x1 * d_x2)
                elif correction_mode == 1:
                    p1_raw = _pcalculate_numba(px, py, xi, lam, a)
                    p2_raw = _pcalculate_numba(px, py, xj, lam, a)
                    p1 = _legacy_gain_numba(p1_raw, legacy_use_abs, legacy_min_gain)
                    p2 = _legacy_gain_numba(p2_raw, legacy_use_abs, legacy_min_gain)
                    k = (p1 * p2 * a * a) / (d_x1 * d_x2)
                elif correction_mode == 3:
                    p1_raw = _piston_directivity_numba(theta1, lam, piston_radius)
                    p2_raw = _piston_directivity_numba(theta2, lam, piston_radius)
                    p1 = _piston_gain_numba(p1_raw, piston_use_abs, piston_min_gain)
                    p2 = _piston_gain_numba(p2_raw, piston_use_abs, piston_min_gain)
                    k = (p1 * p2 * piston_radius * piston_radius) / (d_x1 * d_x2)
                    if attenuation_np_per_m > 0.0:
                        k *= math.exp(-attenuation_np_per_m * (d_x1 + d_x2))
                else:
                    k = 1.0
                if k == 0.0:
                    continue
                if use_sensitivity_comp:
                    inv_k_pair = pair_w / k
                    sens_denom += inv_k_pair * inv_k_pair

                x0 = int(round(((d_x1 + d_x2) / c) / t0))
                inv_k = 1.0 / k
                for t in range(subset_len):
                    src_idx = t + x0
                    if src_idx >= 0 and src_idx < data_len:
                        contrib = data[i, j, src_idx] * inv_k * pair_w
                        result[t] += contrib
                        if use_cf:
                            energy[t] += contrib.real * contrib.real + contrib.imag * contrib.imag
                            counts[t] += 1.0

        abs_trace = np.zeros(subset_len, dtype=np.float64)
        sens_scale = 1.0
        if use_sensitivity_comp and sens_denom > 0.0:
            sens_scale = 1.0 / math.sqrt(sens_denom)
        for t in range(subset_len):
            val = abs(result[t])
            val *= sens_scale
            if use_cf:
                denom = counts[t] * energy[t] + 1e-12
                num = result[t].real * result[t].real + result[t].imag * result[t].imag
                cf = num / denom if denom > 0.0 else 0.0
                if cf < 0.0:
                    cf = 0.0
                val *= math.pow(cf, coherence_gamma)
            abs_trace[t] = val
        z[l, m] = _trace_metric_numba(abs_trace, reduction_mode, gate_center_idx, gate_half_width)
    return z


@dataclass(slots=True)
class ImagingConfig:
    x1: float
    x2: float
    a: float
    lam: float
    c: float
    t0: float
    n: int
    l0: float
    delta: float = 1e-3
    length: float = 0.1
    width: float = 0.06
    subset_len: int = 600
    show: bool = True
    log_progress: bool = True
    progress_every: int = 50
    apply_filter: bool = False
    use_numba: bool = True
    use_fan_mask: bool = True
    fan_half_angle_deg: float = 90.0
    fan_origin_x: float | None = None
    fan_origin_y: float = 0.0
    beam_model: str = "mgb"  # mgb | legacy | piston | none
    piston_diameter: float | None = None
    attenuation_db_per_m: float = 0.0
    piston_use_abs: bool = True
    piston_min_gain: float = 0.2
    aperture_apodization: str = "none"  # none | hann
    legacy_use_abs: bool = True
    legacy_min_gain: float = 0.8
    mgb_angle_c: float = 0.0
    coherence_mode: str = "none"  # none | cf | cf_hilbert
    coherence_gamma: float = 1.0
    reduction_mode: str = "max"  # max | rms | gated_max
    gate_center_idx: int = 0
    gate_half_width: int = 40
    sensitivity_comp: bool = False
    mgb_a: np.ndarray | None = None
    mgb_b: np.ndarray | None = None


class AcousticImager:
    def __init__(self, config: ImagingConfig) -> None:
        self.config = config

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        cfg = self.config
        return my_image(
            cfg.x1,
            cfg.x2,
            cfg.a,
            cfg.lam,
            cfg.c,
            cfg.t0,
            data,
            cfg.n,
            cfg.l0,
            delta=cfg.delta,
            length=cfg.length,
            width=cfg.width,
            subset_len=cfg.subset_len,
            show=cfg.show,
            log_progress=cfg.log_progress,
            progress_every=cfg.progress_every,
            apply_filter=cfg.apply_filter,
            use_numba=cfg.use_numba,
            use_fan_mask=cfg.use_fan_mask,
            fan_half_angle_deg=cfg.fan_half_angle_deg,
            fan_origin_x=cfg.fan_origin_x,
            fan_origin_y=cfg.fan_origin_y,
            beam_model=cfg.beam_model,
            piston_diameter=cfg.piston_diameter,
            attenuation_db_per_m=cfg.attenuation_db_per_m,
            piston_use_abs=cfg.piston_use_abs,
            piston_min_gain=cfg.piston_min_gain,
            aperture_apodization=cfg.aperture_apodization,
            legacy_use_abs=cfg.legacy_use_abs,
            legacy_min_gain=cfg.legacy_min_gain,
            mgb_angle_c=cfg.mgb_angle_c,
            coherence_mode=cfg.coherence_mode,
            coherence_gamma=cfg.coherence_gamma,
            reduction_mode=cfg.reduction_mode,
            gate_center_idx=cfg.gate_center_idx,
            gate_half_width=cfg.gate_half_width,
            sensitivity_comp=cfg.sensitivity_comp,
            mgb_a=cfg.mgb_a,
            mgb_b=cfg.mgb_b,
        )


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
    fan_half_angle_deg: float = 90.0,
    fan_origin_x: float | None = None,
    fan_origin_y: float = 0.0,
    beam_model: str = "mgb",
    piston_diameter: float | None = None,
    attenuation_db_per_m: float = 0.0,
    piston_use_abs: bool = True,
    piston_min_gain: float = 0.2,
    aperture_apodization: str = "none",
    legacy_use_abs: bool = True,
    legacy_min_gain: float = 0.8,
    mgb_angle_c: float = 0.0,
    coherence_mode: str = "none",
    coherence_gamma: float = 1.0,
    reduction_mode: str = "max",
    gate_center_idx: int = 0,
    gate_half_width: int = 40,
    sensitivity_comp: bool = False,
    mgb_a: np.ndarray | None = None,
    mgb_b: np.ndarray | None = None,
) -> np.ndarray:
    t_start = perf_counter()
    data = np.asarray(data, dtype=np.float64, order="C")
    if mgb_angle_c < 0:
        raise ValueError("mgb_angle_c must be >= 0.")
    if attenuation_db_per_m < 0:
        raise ValueError("attenuation_db_per_m must be >= 0.")
    # Convert amplitude attenuation from dB/m to Np/m:
    # A_out = A_in * exp(-alpha_np * d) = A_in * 10^(-alpha_db * d / 20)
    attenuation_np_per_m = attenuation_db_per_m * (math.log(10.0) / 20.0)

    if piston_min_gain < 0:
        raise ValueError("piston_min_gain must be >= 0.")
    if legacy_min_gain < 0:
        raise ValueError("legacy_min_gain must be >= 0.")
    if coherence_gamma < 0:
        raise ValueError("coherence_gamma must be >= 0.")
    if gate_half_width < 0:
        raise ValueError("gate_half_width must be >= 0.")
    if gate_center_idx < 0:
        raise ValueError("gate_center_idx must be >= 0.")

    coherence_mode = coherence_mode.lower().strip()
    if coherence_mode not in {"none", "cf", "cf_hilbert"}:
        raise ValueError("coherence_mode must be 'none', 'cf', or 'cf_hilbert'.")
    use_cf = coherence_mode == "cf"
    use_cf_hilbert = coherence_mode == "cf_hilbert"
    if use_cf_hilbert:
        data = hilbert(data, axis=2)
    reduction_mode = reduction_mode.lower().strip()
    if reduction_mode not in {"max", "rms", "gated_max"}:
        raise ValueError("reduction_mode must be 'max', 'rms', or 'gated_max'.")
    reduction_mode_code = 1 if reduction_mode == "rms" else (2 if reduction_mode == "gated_max" else 0)

    subset_len = min(subset_len, data.shape[2])
    nx = max(1, int(round(length / delta)))
    ny = max(1, int(round(width / delta)))

    if fan_origin_x is None:
        tx_center = x1 + 0.5 * (n - 1) * l0
        rx_center = x2 + 0.5 * (n - 1) * l0
        fan_origin_x = 0.5 * (tx_center + rx_center)
    fan_half_angle_rad = math.radians(fan_half_angle_deg)
    beam_model = beam_model.lower().strip()
    if beam_model not in {"mgb", "legacy", "piston", "none"}:
        raise ValueError("beam_model must be 'mgb', 'legacy', 'piston', or 'none'.")
    correction_mode = 2 if beam_model == "mgb" else (1 if beam_model == "legacy" else (3 if beam_model == "piston" else 0))
    if piston_diameter is not None and piston_diameter <= 0:
        raise ValueError("piston_diameter must be > 0 when provided.")
    piston_radius = 0.5 * piston_diameter if piston_diameter is not None else a
    aperture_apodization = aperture_apodization.lower().strip()
    if aperture_apodization not in {"none", "hann"}:
        raise ValueError("aperture_apodization must be 'none' or 'hann'.")
    if aperture_apodization == "hann" and n >= 3:
        win = np.hanning(n).astype(np.float64)
        mean_win = float(np.mean(win))
        if mean_win > 0:
            win = win / mean_win
            tx_weights = win
            rx_weights = win
        else:
            tx_weights = np.ones(n, dtype=np.float64)
            rx_weights = np.ones(n, dtype=np.float64)
    else:
        tx_weights = np.ones(n, dtype=np.float64)
        rx_weights = np.ones(n, dtype=np.float64)

    if correction_mode == 2:
        if mgb_a is None or mgb_b is None:
            default_a, default_b = default_mgb_coefficients()
            if mgb_a is None:
                mgb_a = default_a
            if mgb_b is None:
                mgb_b = default_b
        mgb_a = np.asarray(mgb_a, dtype=np.float64)
        mgb_b = np.asarray(mgb_b, dtype=np.float64)
        validate_mgb_coefficients(mgb_a, mgb_b)
    else:
        mgb_a = np.array([1.0], dtype=np.float64)
        mgb_b = np.array([1.0], dtype=np.float64)

    print(
        f"[my_image] beam_model={beam_model}, "
        f"subset_len={subset_len}, mgb_terms={mgb_a.size if correction_mode == 2 else 0}, "
        f"mgb_angle_c={mgb_angle_c:.4g}, piston_radius={piston_radius:.4g}, alpha={attenuation_db_per_m:.4g}dB/m, "
        f"piston_abs={piston_use_abs}, piston_min={piston_min_gain:.4g}, coherence={coherence_mode}, "
        f"coh_gamma={coherence_gamma:.3g}, reduction={reduction_mode}, "
        f"gate=({gate_center_idx},{gate_half_width}), sens_comp={sensitivity_comp}, "
        f"apodization={aperture_apodization}, legacy_abs={legacy_use_abs}, legacy_min={legacy_min_gain:.4g}"
    )

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
            correction_mode,
            piston_radius,
            attenuation_np_per_m,
            piston_use_abs,
            piston_min_gain,
            tx_weights,
            rx_weights,
            legacy_use_abs,
            legacy_min_gain,
            mgb_a,
            mgb_b,
            mgb_angle_c,
            use_cf or use_cf_hilbert,
            coherence_gamma,
            reduction_mode_code,
            gate_center_idx,
            gate_half_width,
            sensitivity_comp,
        )
    else:
        if use_numba and not NUMBA_AVAILABLE:
            print("[my_image] numba not installed, using python path")
        if apply_filter:
            print("[my_image] filter enabled; using python path")
        if correction_mode == 2 and apply_filter:
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

                result = np.zeros(subset_len, dtype=data.dtype)
                energy = np.zeros(subset_len, dtype=np.float64)
                counts = np.zeros(subset_len, dtype=np.float64)
                sens_denom = 0.0
                for i in range(n):
                    tx = x1 + i * l0
                    tx_w = tx_weights[i]
                    for j in range(n):
                        rx = x2 + j * l0
                        pair_w = tx_w * rx_weights[j]
                        d_x1 = math.sqrt((tx - xi) * (tx - xi) + yi * yi)
                        d_x2 = math.sqrt((rx - xi) * (rx - xi) + yi * yi)
                        if correction_mode == 2:
                            p1 = float(np.sum(mgb_a * np.exp(-mgb_b * ((tx - xi) * (tx - xi) + yi * yi))))
                            p2 = float(np.sum(mgb_a * np.exp(-mgb_b * ((rx - xi) * (rx - xi) + yi * yi))))
                            if mgb_angle_c > 0:
                                theta1 = abs(math.atan2(tx - xi, yi))
                                theta2 = abs(math.atan2(rx - xi, yi))
                                p1 *= math.exp(-mgb_angle_c * theta1 * theta1)
                                p2 *= math.exp(-mgb_angle_c * theta2 * theta2)
                            k_gain = (max(p1, 1e-12) * max(p2, 1e-12) * a * a) / (d_x1 * d_x2)
                        elif correction_mode == 1:
                            p1_raw = _pcalculate_numba(xi, yi, tx, lam, a)
                            p2_raw = _pcalculate_numba(xi, yi, rx, lam, a)
                            p1 = max(abs(p1_raw), legacy_min_gain) if legacy_use_abs else max(
                                p1_raw, legacy_min_gain
                            )
                            p2 = max(abs(p2_raw), legacy_min_gain) if legacy_use_abs else max(
                                p2_raw, legacy_min_gain
                            )
                            k_gain = (p1 * p2 * a * a) / (d_x1 * d_x2)
                        elif correction_mode == 3:
                            theta1 = abs(math.atan2(tx - xi, yi))
                            theta2 = abs(math.atan2(rx - xi, yi))
                            p1_raw = _piston_directivity_numba(theta1, lam, piston_radius)
                            p2_raw = _piston_directivity_numba(theta2, lam, piston_radius)
                            p1 = max(abs(p1_raw), piston_min_gain) if piston_use_abs else max(
                                p1_raw, piston_min_gain
                            )
                            p2 = max(abs(p2_raw), piston_min_gain) if piston_use_abs else max(
                                p2_raw, piston_min_gain
                            )
                            k_gain = (p1 * p2 * piston_radius * piston_radius) / (d_x1 * d_x2)
                            if attenuation_np_per_m > 0:
                                k_gain *= math.exp(-attenuation_np_per_m * (d_x1 + d_x2))
                        else:
                            k_gain = 1.0
                        if k_gain == 0:
                            continue
                        if sensitivity_comp:
                            inv_k_pair = pair_w / k_gain
                            sens_denom += inv_k_pair * inv_k_pair
                        x0 = int(round(((d_x1 + d_x2) / c) / t0))
                        inv_k = 1.0 / k_gain
                        for t in range(subset_len):
                            src_idx = t + x0
                            if 0 <= src_idx < data.shape[2]:
                                contrib = data[i, j, src_idx] * inv_k * pair_w
                                result[t] += contrib
                                if use_cf or use_cf_hilbert:
                                    energy[t] += contrib.real * contrib.real + contrib.imag * contrib.imag
                                    counts[t] += 1.0
                if apply_filter and filter_coeffs is not None:
                    b, a_filter = filter_coeffs
                    result = lfilter(b, a_filter, result)
                abs_eval = np.abs(result)
                if sensitivity_comp and sens_denom > 0.0:
                    abs_eval = abs_eval / np.sqrt(sens_denom)
                if use_cf or use_cf_hilbert:
                    denom = counts * energy + 1e-12
                    result_sq_mag = result.real * result.real + result.imag * result.imag
                    cf = np.where(denom > 0.0, result_sq_mag / denom, 0.0)
                    cf = np.clip(cf, 0.0, 1.0)
                    abs_eval = abs_eval * np.power(cf, coherence_gamma)

                if reduction_mode_code == 1:
                    z[l, m] = float(np.sqrt(np.mean(abs_eval * abs_eval)))
                elif reduction_mode_code == 2:
                    start = max(0, gate_center_idx - gate_half_width)
                    end = min(abs_eval.size, gate_center_idx + gate_half_width + 1)
                    z[l, m] = float(np.max(abs_eval[start:end])) if start < end else 0.0
                else:
                    z[l, m] = float(np.max(abs_eval))
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
        a_log[valid] = np.log2(z[valid] + 1.0)
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
        plt.imshow(
            np.flipud(a_norm.T),
            origin="lower",
            aspect="equal",
            cmap="turbo",
            interpolation="nearest",
        )
        plt.colorbar()
        plt.title("Normalized Imaging Result")
        plt.show()

    total = perf_counter() - t_start
    print(f"[my_image] total runtime: {total:.2f}s")
    return a_norm
