from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat


def _extract_first_numeric_vector(mat_dict: dict) -> np.ndarray:
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value).squeeze()
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            return arr.astype(np.float64, copy=False)
    raise KeyError("No numeric 1D signal found in MAT file.")


def load_mat_signal(file_path: str | Path, variable_name: str = "C1_data") -> np.ndarray:
    file_path = str(file_path)
    try:
        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        if variable_name in mat:
            data = np.asarray(mat[variable_name]).squeeze()
        else:
            data = _extract_first_numeric_vector(mat)
        return data.reshape(-1).astype(np.float64, copy=False)
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5-based.
        import h5py

        with h5py.File(file_path, "r") as f:
            if variable_name in f:
                data = np.array(f[variable_name]).squeeze()
            else:
                # Fall back to the first numeric dataset.
                data = None
                for key in f.keys():
                    arr = np.array(f[key]).squeeze()
                    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                        data = arr
                        break
                if data is None:
                    raise KeyError("No numeric 1D signal found in MAT file.")
        return data.reshape(-1).astype(np.float64, copy=False)


def my_read(
    files: Sequence[Sequence[str]],
    t: int,
    n: int | None = None,
    max_samples: int = 40000,
    variable_name: str = "C1_data",
) -> np.ndarray:
    """
    Read a matrix of MAT files and stack into shape (n, n, samples).

    This is the shared reader used by experiment scripts, including the
    multi-file load pattern from experiment2.
    """
    if n is None:
        n = len(files)
    samples = min(t, max_samples)
    data = np.zeros((n, n, samples), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            signal = load_mat_signal(files[i][j], variable_name=variable_name)
            use_len = min(samples, signal.size)
            data[i, j, :use_len] = signal[:use_len]
    return data
