from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def _is_numeric_array(value: Any) -> bool:
    arr = np.asarray(value)
    return np.issubdtype(arr.dtype, np.number)


def _load_mat_dict(file_path: str) -> dict[str, Any]:
    try:
        return loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import h5py

        result: dict[str, Any] = {}
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                result[key] = np.array(f[key])
        return result


def my_read2_new(
    file: str | Path,
    t: int,
    t1: int,
    m: int,
    n: int,
    variable_keyword: str = "voltage",
) -> np.ndarray:
    """
    Python port of sbts_new/my_read2.m.

    Loads one MAT file, finds variables whose names contain `variable_keyword`,
    and packs them in order into shape (m, n, n, window_size).
    """
    file = str(file)
    if t <= 0 or t1 <= 0:
        raise ValueError("t and t1 must be positive.")
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive.")
    if not variable_keyword:
        raise ValueError("variable_keyword must be non-empty.")

    window_size = min(t, t1)
    result = np.zeros((m, n, n, window_size), dtype=np.float64)
    data = _load_mat_dict(file)

    var_names = sorted(k for k in data.keys() if not k.startswith("__"))
    voltage_vars = [name for name in var_names if variable_keyword in name]
    expected = m * n * n
    if len(voltage_vars) < expected:
        raise ValueError(
            f"数据不全: expected at least {expected} '*{variable_keyword}*' vars, got {len(voltage_vars)}."
        )

    idx = 0
    for i in range(m):
        for j in range(n):
            for k in range(n):
                var_name = voltage_vars[idx]
                voltage_data = data[var_name]
                if not _is_numeric_array(voltage_data):
                    raise TypeError(f"Variable '{var_name}' is not numeric.")
                signal = np.asarray(voltage_data).reshape(-1).astype(np.float64, copy=False)
                use_len = min(window_size, signal.size)
                result[i, j, k, :use_len] = signal[:use_len]
                idx += 1

    return result
