# AGENTS.md

## Project overview

Acoustic (ultrasonic) flaw detection — A-scan → imaging reconstruction → defect identification with geometric parameter output. Python primary, MATLAB reference in `matlab/`.

## Entry points

- `python/experiment.py` — multi-file MAT A-scan reader (`my_read.py`), `n=8` array, sbts_new geometry
- `python/experiment_new.py` — single-MAT reader (`my_read2_new.py`), `n=8` array, sbts_new data.mat geometry
- `python/flaw_infer_image.py` — CLI for single-image inference (from pre-reconstructed image, not A-scan)

## Architecture

```
A-scan data (MAT → numpy 3D n×n×T)
  → experiment_common.preprocess_channels()     [bandpass + DC + common-mode]
  → experiment_common.build_imaging_config()    [geometry + imaging params]
  → my_image.AcousticImager.reconstruct()       [core beamforming, Numba-accelerated]
  → experiment_common.apply_shallow_mask()      [top-layer suppression]
  → flaw_detection.detect_flaws(mode="rule"|"ml")
  → flaw_visualize.draw_defect_overlays()
```

## Coordinate conventions

- `detect_flaws()` internally calls `flipud(image.T)` to convert reconstruction image → display coordinates before segmentation.
- `flaw_infer_image._load_image_gray()` reverses this: reads a display image, does `flipud(arr).T` back to reconstruction orientation so `detect_flaws()` can process it.
- When feeding a pre-existing image directly to `detect_flaws()`, ensure it is in reconstruction orientation (NOT display orientation).

## Detection modes

| Mode | Imports | Notes |
|------|---------|-------|
| `rule` | `scipy.ndimage` | Otsu or percentile threshold, no model needed |
| `ml` | `ultralytics` (SAM/FastSAM) | Falls back to rule if `fallback_to_rule=True` and `ultralytics` missing |

## Dependencies

Required: `numpy`, `scipy`, `matplotlib`  
Optional: `numba` (speedup), `ultralytics` (ML mode)

No virtual env, no pyproject.toml, no setup.py, no lockfile. `requirements.txt` is a binary file — do not rely on it.

## Two experiment files have **different geometry**

`experiment.py` and `experiment_new.py` use `build_geometry_settings()` with different `l0`, `c`, `t0`, `x2` values. Do not copy-paste geometry between them.

## Adding new test data

- Place `.mat` files under `test_data/` following the naming convention `NN.mat` (e.g., `00.mat`–`33.mat`).
- The data shape is `(n, n, T)` where `n` is the array size (default 8) and `T` is time samples.

## Tool routing

- **Document/text generation** (PPT, docs, reports with Chinese text) → use `bash` + `python3`, NOT `skilllite_execute_code`. skilllite's entropy scanner false-positives on Chinese characters.
- **Untrusted/external code execution** → use `skilllite_execute_code` with sandbox.

## Known issues

- `.gitignore` has `opencode.json` — opencode config is not tracked in git.
- No test framework, no lint/typecheck config. There is no CI.
- MATLAB files in `matlab/` are reference only, not called by Python code.
- `experiment.py` uses hardcoded Windows paths (`D:\tanshang\...`).

<!-- evolver-evolution-memory -->
## Evolution Memory (Evolver)

This project uses evolver for self-evolution. Hooks automatically:
1. Inject recent evolution memory at session start
2. Detect evolution signals during file edits
3. Record outcomes at session end

For substantive tasks, call `gep_recall` before work and `gep_record_outcome` after.
Signals: log_error, perf_bottleneck, user_feature_request, capability_gap, deployment_issue, test_failure.
