# Standalone Pipeline End-to-End Test with Metal GPU Verification

**Date:** 2026-07-08
**Status:** Approved (design)

## Goal

Provide an automated end-to-end test for the standalone WINMOL pipeline
(`standalone/WINMOL_Analyzer.py`) that runs on a real orthomosaic and verifies:

1. The pipeline completes and writes its expected outputs.
2. The U-Net inference actually executes on the Apple **Metal** GPU.

## Context / findings

- `standalone/WINMOL_Analyzer.py` is a legacy top-to-bottom `__main__` script with
  no callable functions, so it is not directly importable/testable.
- It is currently **broken**: line 80 calls `IO.stems_to_geojson(...)`, which no
  longer exists. The codebase moved from GeoJSON to GeoPackage output.
- No test orthomosaic existed in-repo. The test will use a real file supplied by
  the user:
  `/Users/christian/data/Winmol/Winmol Orthos/WINMOL_x_OeBF/Raster/proj/repr_20250327_171_4_Kraking_Windwurf_SELEKTION.tif`
  (10528×7252 px, 4-band uint8 RGBA, EPSG:25833, ~4 cm GSD, ~291 MB).
- `IO.load_orthomosaic` reads only bands `1..config.n_channels` (=3), so the alpha
  band is safely ignored.
- `tensorflow-metal` is **not** installed in the `WINMOL_Analyzer` conda env
  (Python 3.11.15, TensorFlow 2.21.0 CPU). The standalone path — unlike
  `winmol_run.py` — never forces CPU-only, so Metal will be used once installed.
- No test framework is installed; there is no existing `tests/` tree.

## Scope decisions (approved)

- **Test goal:** full end-to-end run + Metal verification.
- **Input:** a **cropped ~2048×2048 px window** of the real ortho, written to a
  temp GeoTIFF at test time. Fast (~1 min), real data, likely real detections.
- **Framework:** pytest (installed into the env).

## Changes

### 1. Refactor `standalone/WINMOL_Analyzer.py` into a callable

Extract the `__main__` body into:

```python
def run_pipeline(model_path, img_path, pred_dir, output_dir, config=None) -> dict:
    """Run the full standalone pipeline; return output paths + stem count."""
    ...
    return {
        "stem_map_path": <path to exported stem-map .tiff>,
        "gpkg_path": <path to exported GeoPackage>,
        "num_stems": len(stems),
    }
```

Keep a thin `if __name__ == "__main__":` block that parses `sys.argv` and calls
`run_pipeline(...)`. Behavior for the CLI path is preserved.

### 2. Fix the broken export

Replace the dead call:

```python
IO.stems_to_geojson(stems, output_dir + file_name)          # remove
```
with the current GeoPackage API:
```python
gpkg_path = IO.write_all_layers_to_gpkg(stems, profile, output_dir + file_name)
```
`export_stem_map(pred, profile, pred_dir, file_name)` is unchanged (writes
`<pred_dir>/<file_name>.tiff`).

### 3. Install `tensorflow-metal`

Install into the `WINMOL_Analyzer` env so the forward pass can run on Metal.
Document it as an optional macOS/Apple-Silicon extra (it must NOT be added to
`requirements/tensorflow.txt`, which is the Linux/Windows CUDA path).

### 4. Add the test: `tests/test_standalone_pipeline.py`

**Fixtures**
- `metal_gpu`: returns the Metal `PhysicalDevice`; `pytest.skip` if none present
  (keeps the test portable to non-Metal/CI environments).
- `cropped_ortho(tmp_path)`: reads a 2048×2048 window from the source ortho with
  rasterio, writes a temp 3-band GeoTIFF preserving CRS + windowed transform.
  `pytest.skip` if the source file is absent.
- `model_path`: the `General` `.hdf5` in `standalone/model/`; `pytest.skip` if
  absent.

**Test body**
1. Assert `tf.config.list_physical_devices('GPU')` contains a Metal device.
2. Call `run_pipeline(model_path, cropped_ortho, pred_dir, output_dir)`.
3. Assert the returned `stem_map_path` exists and opens as a valid raster.
4. Assert the returned `gpkg_path` exists.
5. Assert Metal was used for inference (see below).
6. If `num_stems > 0`: assert stems expose positive `length`/`volume`.

**Metal-usage assertion**
Presence of a Metal GPU + a successful Keras forward pass is the primary
evidence, since Keras auto-places on the GPU when one is visible. To make it
explicit, the test additionally runs one model prediction inside
`with tf.device('/GPU:0'):` and asserts it completes without a placement error.

### 5. Dev dependency

Add `pytest` to `requirements/development.txt` and install it into the env.

## Out of scope

- No changes to `winmol_run.py` / the QGIS plugin path.
- No CUDA/Linux behavior changes.
- No full-image inference (cropped window only).
- No assertion on specific detection counts (real data, but content not pinned).

## Success criteria

- `pytest tests/test_standalone_pipeline.py` passes on the Apple-Silicon env with
  `tensorflow-metal` installed.
- The test skips (does not fail) when Metal, the source ortho, or the model is
  unavailable.
- `standalone/WINMOL_Analyzer.py` runs end-to-end (no `stems_to_geojson` crash)
  both via the test and via its CLI `__main__`.
