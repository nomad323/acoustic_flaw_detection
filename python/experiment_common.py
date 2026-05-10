from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from my_image import AcousticImager, ImagingConfig


def preprocess_channels(
    data: np.ndarray,
    fs: float,
    center_freq_hz: float = 5e6,
    bandwidth_hz: float = 2e6,
    remove_dc: bool = True,
    mute_front_samples: int = 80,
    subtract_common_mode: bool = True,
    common_mode_method: str = "median",
) -> np.ndarray:
    out = np.asarray(data, dtype=np.float64).copy()
    if remove_dc:
        out -= np.mean(out, axis=2, keepdims=True)

    low = max(1.0, center_freq_hz - bandwidth_hz / 2.0)
    high = min(0.49 * fs, center_freq_hz + bandwidth_hz / 2.0)
    if high > low:
        sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
        out = sosfiltfilt(sos, out, axis=2)

    if mute_front_samples > 0:
        front = min(mute_front_samples, out.shape[2])
        out[:, :, :front] = 0.0

    if subtract_common_mode:
        method = common_mode_method.lower().strip()
        if method == "median":
            common_mode = np.median(out, axis=(0, 1), keepdims=True)
        elif method == "mean":
            common_mode = np.mean(out, axis=(0, 1), keepdims=True)
        else:
            raise ValueError("common_mode_method must be 'median' or 'mean'.")
        out -= common_mode

    return out


def maybe_preprocess_data(
    data: np.ndarray,
    fs: float,
    use_preprocess: bool,
    preprocess_kwargs: dict[str, Any],
) -> np.ndarray:
    if not use_preprocess:
        return data
    return preprocess_channels(data, fs=fs, **preprocess_kwargs)


def build_imaging_config(
    *,
    x1: float,
    x2: float,
    a: float,
    lam: float,
    c: float,
    t0: float,
    n: int,
    l0: float,
    imaging_kwargs: dict[str, Any],
) -> ImagingConfig:
    return ImagingConfig(
        x1=x1,
        x2=x2,
        a=a,
        lam=lam,
        c=c,
        t0=t0,
        n=n,
        l0=l0,
        delta=imaging_kwargs["delta"],
        length=imaging_kwargs["length"],
        width=imaging_kwargs["width"],
        subset_len=imaging_kwargs["subset_len"],
        show=imaging_kwargs["show"],
        log_progress=imaging_kwargs["log_progress"],
        progress_every=imaging_kwargs["progress_every"],
        apply_filter=imaging_kwargs["apply_filter"],
        use_numba=imaging_kwargs["use_numba"],
        use_fan_mask=imaging_kwargs["use_fan_mask"],
        fan_half_angle_deg=imaging_kwargs["fan_half_angle_deg"],
        fan_origin_x=imaging_kwargs["fan_origin_x"],
        fan_origin_y=imaging_kwargs["fan_origin_y"],
        beam_model=imaging_kwargs["beam_model"],
        piston_diameter=imaging_kwargs["piston_diameter"],
        attenuation_db_per_m=imaging_kwargs["attenuation_db_per_m"],
        piston_use_abs=imaging_kwargs["piston_use_abs"],
        piston_min_gain=imaging_kwargs["piston_min_gain"],
        aperture_apodization=imaging_kwargs["aperture_apodization"],
        legacy_use_abs=imaging_kwargs["legacy_use_abs"],
        legacy_min_gain=imaging_kwargs["legacy_min_gain"],
        mgb_angle_c=imaging_kwargs["mgb_angle_c"],
        coherence_mode=imaging_kwargs["coherence_mode"],
        coherence_gamma=imaging_kwargs["coherence_gamma"],
        reduction_mode=imaging_kwargs["reduction_mode"],
        gate_center_idx=imaging_kwargs["gate_center_idx"],
        gate_half_width=imaging_kwargs["gate_half_width"],
        sensitivity_comp=imaging_kwargs["sensitivity_comp"],
    )


def reconstruct_image(data: np.ndarray, config: ImagingConfig) -> np.ndarray:
    return AcousticImager(config).reconstruct(data)


def apply_shallow_mask(
    image: np.ndarray,
    delta: float,
    shallow_mask_mm: float,
) -> np.ndarray:
    display_img = np.flipud(image.T).copy()
    rows_to_mask = int(round(shallow_mask_mm / (delta * 1e3)))
    if rows_to_mask > 0:
        rows_to_mask = min(rows_to_mask, display_img.shape[0])
        display_img[:rows_to_mask, :] = 0.0
    return np.flipud(display_img).T


def show_shallow_masked_image(
    image: np.ndarray,
    delta: float,
    shallow_mask_mm: float,
    title: str,
) -> None:
    masked_image = apply_shallow_mask(image=image, delta=delta, shallow_mask_mm=shallow_mask_mm)
    display_img = np.flipud(masked_image.T)
    plt.figure()
    plt.imshow(display_img, origin="lower", aspect="equal")
    plt.colorbar()
    plt.title(title)
    plt.show()
