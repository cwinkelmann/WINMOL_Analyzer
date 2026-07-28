# Nodata-aware Edge Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the U-Net from producing spurious "stem" activations along an orthomosaic's nodata/transparent boundary (the barnekow top edge) by replacing invalid pixels with their nearest valid neighbour before inference.

**Architecture:** A single pure helper, `fill_invalid_with_nearest(tile, valid_mask)`, lives in a new leaf module `utils/edge_fill.py`. Both prediction read paths — the single-GPU stream producer (`utils/Prediction.py::TileBatchProducer`) and the multi-GPU reader (`utils/PredictWorkers.py::_read_batch_jobs`) — call it right after they compute the validity mask and before they hand the tile downstream. The validity mask is emitted unchanged, so the existing post-inference masking still zeroes the filled regions. Gated by a new `Config.fill_invalid_before_prediction` flag (default `True`).

**Tech Stack:** Python 3.9–3.11, numpy, scipy (`scipy.ndimage.distance_transform_edt`), rasterio, onnxruntime. No new dependencies — scipy already ships via the geo stack.

## Global Constraints

- **Runtime is TensorFlow-free** — ONNX via onnxruntime only. Add no TF imports.
- **No new dependencies.** scipy is already a dependency (scikit-image / geo stack).
- **flake8 is the CI gate** (`setup.cfg`): `max-line-length=80`, `max-doc-length=130`. Every code line must satisfy this.
- **Helper lives in a leaf module** (`utils/edge_fill.py`) importing only numpy (+ lazy scipy). It must NOT import `utils.Prediction`: `PredictWorkers` runs its readers in separate `multiprocessing` processes and deliberately avoids the heavy `utils.Prediction` import graph.
- **Config-gated:** `fill_invalid_before_prediction = True`, overridable at runtime via `WINMOL_CONFIG_OVERRIDES_JSON`.
- **Branch:** `feat/segmentation-edge-fill`, based on the rc11 tip. Experimental; **not** part of any release candidate.
- **Test env:** run pytest with the project conda env
  `/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python` and
  `PYTHONHASHSEED=0` (else `tests/conftest.py` re-execs and its child's output
  goes to `/dev/tty`, appearing silent). `WINMOL_BATCH_AUTOTUNE=off` is set by
  conftest for determinism.

---

### Task 1: Shared edge-fill helper + unit tests

**Files:**
- Create: `utils/edge_fill.py`
- Test: `tests/test_edge_fill.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fill_invalid_with_nearest(tile, valid_mask) -> tile`. `tile` is `(H, W, C)` any dtype; `valid_mask` is `(H, W)` bool (True = real data). Returns a tile with invalid pixels replaced by the nearest valid pixel's value; returns the input unchanged when the mask is all-valid or all-invalid; never mutates the input.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_edge_fill.py`:

```python
import numpy as np

from utils.edge_fill import fill_invalid_with_nearest


def test_all_valid_is_noop():
    tile = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    mask = np.ones((2, 2), dtype=bool)
    out = fill_invalid_with_nearest(tile, mask)
    assert np.array_equal(out, tile)


def test_all_invalid_is_noop():
    tile = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=bool)
    out = fill_invalid_with_nearest(tile, mask)
    assert np.array_equal(out, tile)


def test_invalid_pixels_take_nearest_valid_value():
    # Row of 3: left pixel valid, middle+right invalid (black).
    tile = np.zeros((1, 3, 3), dtype=np.uint8)
    tile[0, 0] = (10, 20, 30)
    mask = np.array([[True, False, False]])
    out = fill_invalid_with_nearest(tile, mask)
    assert tuple(out[0, 0]) == (10, 20, 30)   # valid untouched
    assert tuple(out[0, 1]) == (10, 20, 30)   # nearest-valid replicate
    assert tuple(out[0, 2]) == (10, 20, 30)


def test_does_not_mutate_input():
    tile = np.zeros((1, 2, 3), dtype=np.uint8)
    tile[0, 0] = (5, 6, 7)
    mask = np.array([[True, False]])
    original = tile.copy()
    _ = fill_invalid_with_nearest(tile, mask)
    assert np.array_equal(tile, original)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONHASHSEED=0 /Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest tests/test_edge_fill.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.edge_fill'`.

- [ ] **Step 3: Write the helper**

Create `utils/edge_fill.py`:

```python
"""Nodata-aware edge fill for U-Net segmentation tiles.

Shared by the single-GPU stream producer (utils/Prediction.py) and the
multi-GPU reader (utils/PredictWorkers.py). Kept in its own leaf module so
the multiprocessing workers in PredictWorkers can import it without pulling
in the heavy utils.Prediction dependency graph.

See docs/superpowers/specs/2026-07-28-segmentation-edge-fill-design.md.
"""
import numpy as np


def fill_invalid_with_nearest(tile, valid_mask):
    """Replace invalid pixels with their nearest valid neighbour.

    The U-Net must not see the hard black cliff at a nodata / out-of-bounds
    boundary: its receptive field bleeds spurious stem activations onto the
    valid pixels just inside the edge. Replicating the nearest valid pixel
    into the invalid region removes the cliff before inference; the output is
    still masked afterwards, so filled regions produce no stems.

    tile:       (H, W, C) array as read from the source (any dtype).
    valid_mask: (H, W) bool -- True where the pixel is real data.

    Returns a NEW tile with invalid pixels replaced; valid pixels untouched.
    Returns the input unchanged when the mask is all-valid or all-invalid.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        return tile
    if valid.all() or not valid.any():
        return tile
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(
        ~valid, return_distances=False, return_indices=True)
    return tile[tuple(idx)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONHASHSEED=0 /Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest tests/test_edge_fill.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint**

Run: `/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m flake8 utils/edge_fill.py tests/test_edge_fill.py`
Expected: no output (clean).

- [ ] **Step 6: Commit**

```bash
git add utils/edge_fill.py tests/test_edge_fill.py
git commit -m "feat(prediction): nearest-valid edge fill helper"
```

---

### Task 2: Config flag + wire into the single-GPU stream producer

**Files:**
- Modify: `classes/Config.py` (after `num_classes = 1`)
- Modify: `utils/Prediction.py` (`TileBatchProducer.__init__`, `TileBatchProducer.run`, and the instantiation ~line 1110)
- Test: `tests/test_edge_fill_producer.py`

**Interfaces:**
- Consumes: `fill_invalid_with_nearest` from Task 1.
- Produces: `TileBatchProducer(..., fill_invalid=True)` — new keyword arg, stored as `self.fill_invalid`; the producer fills each tile before queuing when it is truthy. `Config.fill_invalid_before_prediction` (bool, default `True`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_edge_fill_producer.py`:

```python
import queue

import numpy as np
import rasterio
from rasterio.transform import from_origin

from utils.Prediction import TileBatchProducer


def _make_raster(path):
    # Uniform colour with a black (invalid) top-left corner.
    data = np.zeros((3, 300, 300), dtype=np.uint8)
    data[0] = 100
    data[1] = 150
    data[2] = 200
    data[:, :50, :50] = 0
    profile = dict(
        driver="GTiff", height=300, width=300, count=3, dtype="uint8",
        crs="EPSG:25833", transform=from_origin(0, 300, 1, 1))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _run_producer(path, fill_invalid):
    q = queue.Queue()
    job = {"src_col": 0, "src_row": 0, "src_width": 300, "src_height": 300}
    p = TileBatchProducer(
        uav_path=str(path), chunk_size=1, jobs=[job], n_channels=3,
        out_queue=q, out_size=None, fill_invalid=fill_invalid)
    p.start()
    p.join()
    assert p.error is None, p.error
    msg = q.get_nowait()
    _, tile, valid_mask = msg["items"][0]
    return tile, valid_mask


def test_producer_fills_black_corner_when_enabled(tmp_path):
    path = tmp_path / "r.tif"
    _make_raster(path)
    tile, valid_mask = _run_producer(path, fill_invalid=True)
    assert not valid_mask[10, 10]                    # corner is invalid
    assert tuple(tile[10, 10]) == (100, 150, 200)    # filled from nearest


def test_producer_leaves_black_corner_when_disabled(tmp_path):
    path = tmp_path / "r.tif"
    _make_raster(path)
    tile, valid_mask = _run_producer(path, fill_invalid=False)
    assert tuple(tile[10, 10]) == (0, 0, 0)          # untouched cliff
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONHASHSEED=0 /Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest tests/test_edge_fill_producer.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'fill_invalid'`.

- [ ] **Step 3: Add the Config flag**

In `classes/Config.py`, immediately after the line `num_classes = 1`, add:

```python
    # Replace nodata / out-of-bounds pixels with the nearest valid pixel
    # BEFORE inference, so the U-Net never sees a hard black boundary cliff
    # and cannot bleed spurious stems onto valid edge pixels. Output is still
    # masked afterwards. See
    # docs/superpowers/specs/2026-07-28-segmentation-edge-fill-design.md.
    fill_invalid_before_prediction = True
```

- [ ] **Step 4: Add the constructor arg and store it**

In `utils/Prediction.py`, `TileBatchProducer.__init__`, change the signature and add the assignment. The signature becomes:

```python
    def __init__(self, uav_path, chunk_size, jobs, n_channels,
                 out_queue, producer_id=0, out_size=None,
                 fill_invalid=True):
```

and after `self.out_size = tuple(out_size) if out_size else None` add:

```python
        self.fill_invalid = bool(fill_invalid)
```

- [ ] **Step 5: Import the helper and fill before queuing**

In `utils/Prediction.py`, add to the imports (with the other `from utils import ...` / local imports near the top):

```python
from utils.edge_fill import fill_invalid_with_nearest
```

In `TileBatchProducer.run`, the block currently reads:

```python
                    if np.all(gdal_mask):
                        valid_mask = pixel_mask
                    else:
                        valid_mask = gdal_mask & pixel_mask

                    batch_read_s += time.perf_counter() - t0
                    batch_items.append((job, tile, valid_mask))
```

Insert the fill between the mask decision and `batch_items.append`:

```python
                    if np.all(gdal_mask):
                        valid_mask = pixel_mask
                    else:
                        valid_mask = gdal_mask & pixel_mask

                    if self.fill_invalid:
                        tile = fill_invalid_with_nearest(tile, valid_mask)

                    batch_read_s += time.perf_counter() - t0
                    batch_items.append((job, tile, valid_mask))
```

- [ ] **Step 6: Pass the flag at the instantiation site**

In `utils/Prediction.py`, at the `TileBatchProducer(` instantiation (~line 1110), add the argument after `out_size=(config.img_height, config.img_width),`:

```python
            out_size=(config.img_height, config.img_width),
            fill_invalid=bool(getattr(
                config, "fill_invalid_before_prediction", True)),
```

- [ ] **Step 7: Run the producer test to verify it passes**

Run: `PYTHONHASHSEED=0 /Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest tests/test_edge_fill_producer.py -q`
Expected: PASS (2 passed).

- [ ] **Step 8: Golden-inference regression must stay green**

Run: `PYTHONHASHSEED=0 /Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest tests/test_inference.py -q`
Expected: PASS. The golden fixture `tests/fixtures/crop_input.tif` is 100% valid (no black, no nodata), so `fill_invalid_with_nearest` hits its all-valid no-op path on interior tiles; any effect is confined to the crop's own outer edge and is absorbed by the existing `>= 0.999` agreement / foreground-drift tolerance. If it FAILS on foreground drift, the crop-edge tiles went out-of-bounds and the fill changed edge predictions — inspect the diff; if it is the intended edge-artifact removal, regenerate the golden with fill on via `tests/generate_fixtures.py` and note it in the commit.

- [ ] **Step 9: Lint**

Run: `/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m flake8 classes/Config.py utils/Prediction.py tests/test_edge_fill_producer.py`
Expected: no output (clean).

- [ ] **Step 10: Commit**

```bash
git add classes/Config.py utils/Prediction.py tests/test_edge_fill_producer.py
git commit -m "feat(prediction): edge-fill in the single-GPU stream producer"
```

---

### Task 3: Wire into the multi-GPU reader

**Files:**
- Modify: `utils/PredictWorkers.py` (`_read_batch_jobs` signature + body; call sites ~line 228 and ~line 282)
- Test: `tests/test_edge_fill_multigpu.py`

**Interfaces:**
- Consumes: `fill_invalid_with_nearest` from Task 1; `Config.fill_invalid_before_prediction` from Task 2.
- Produces: `_read_batch_jobs(src, indexes, batch_jobs, fill_invalid=True) -> (raw_tiles, raw_masks, stats)` — new keyword arg; fills each tile before appending when truthy.

- [ ] **Step 1: Write the failing test**

Create `tests/test_edge_fill_multigpu.py`:

```python
import numpy as np
import rasterio
from rasterio.transform import from_origin

from utils.PredictWorkers import _read_batch_jobs


def _make_raster(path):
    data = np.zeros((3, 300, 300), dtype=np.uint8)
    data[0] = 100
    data[1] = 150
    data[2] = 200
    data[:, :50, :50] = 0
    profile = dict(
        driver="GTiff", height=300, width=300, count=3, dtype="uint8",
        crs="EPSG:25833", transform=from_origin(0, 300, 1, 1))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _read(path, fill_invalid):
    job = {"src_col": 0, "src_row": 0, "src_width": 300, "src_height": 300}
    with rasterio.open(path) as src:
        tiles, masks, _ = _read_batch_jobs(
            src, [1, 2, 3], [job], fill_invalid=fill_invalid)
    return tiles[0], masks[0]


def test_multigpu_fills_black_corner_when_enabled(tmp_path):
    path = tmp_path / "r.tif"
    _make_raster(path)
    tile, mask = _read(path, fill_invalid=True)
    assert not mask[10, 10]
    assert tuple(tile[10, 10]) == (100, 150, 200)


def test_multigpu_leaves_black_corner_when_disabled(tmp_path):
    path = tmp_path / "r.tif"
    _make_raster(path)
    tile, mask = _read(path, fill_invalid=False)
    assert tuple(tile[10, 10]) == (0, 0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONHASHSEED=0 /Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest tests/test_edge_fill_multigpu.py -q`
Expected: FAIL — `TypeError: _read_batch_jobs() got an unexpected keyword argument 'fill_invalid'`.

- [ ] **Step 3: Import the helper**

In `utils/PredictWorkers.py`, add to the imports near the top (after the existing `import numpy as np` / `import rasterio`):

```python
from utils.edge_fill import fill_invalid_with_nearest
```

- [ ] **Step 4: Add the parameter and the fill**

In `utils/PredictWorkers.py`, change the signature:

```python
def _read_batch_jobs(src, indexes, batch_jobs, fill_invalid=True):
```

and in the loop body, between `stats['prep_s'] += time.perf_counter() - t0` and `raw_tiles.append(tile)`, insert:

```python
        if fill_invalid:
            tile = fill_invalid_with_nearest(tile, valid_mask)

        raw_tiles.append(tile)
        raw_masks.append(valid_mask)
```

- [ ] **Step 5: Pass the flag at both call sites**

In `utils/PredictWorkers.py`, both `_read_batch_jobs(src, indexes, batch_jobs)` calls (~line 228 and ~line 282) sit below a `cfg = _config_from_dict(config_dict)`. Change each call to:

```python
                _read_batch_jobs(
                    src, indexes, batch_jobs,
                    fill_invalid=bool(getattr(
                        cfg, "fill_invalid_before_prediction", True)))
```

(Match the existing indentation at each site.)

- [ ] **Step 6: Run the multi-GPU test to verify it passes**

Run: `PYTHONHASHSEED=0 /Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest tests/test_edge_fill_multigpu.py -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Lint**

Run: `/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m flake8 utils/PredictWorkers.py tests/test_edge_fill_multigpu.py`
Expected: no output (clean).

- [ ] **Step 8: Commit**

```bash
git add utils/PredictWorkers.py tests/test_edge_fill_multigpu.py
git commit -m "feat(prediction): edge-fill in the multi-GPU reader"
```

---

### Task 4: Full-suite regression check

**Files:**
- (no code change — verification gate)

- [ ] **Step 1: Run the whole test suite**

Run: `PYTHONHASHSEED=0 /Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest -q`
Expected: PASS (all green, including `test_inference.py` and the three new edge-fill test files). If a golden test regresses, see Task 2 Step 8.

- [ ] **Step 2: Full-tree lint**

Run: `/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m flake8`
Expected: no output (clean).

---

### Task 5: Barnekow before/after validation (local acceptance evidence)

**Files:**
- Create: `benchmark/validate_edge_fill_barnekow.py`

**Interfaces:**
- Consumes: `standalone.WINMOL_Analyzer.run_pipeline` (or the stream predictor) with `WINMOL_CONFIG_OVERRIDES_JSON` toggling `fill_invalid_before_prediction`.

This task produces the acceptance figure and numbers. It needs the model and
the barnekow ortho on disk, so it SKIPS cleanly when they are absent (it is not
a CI test).

- [ ] **Step 1: Write the validation script**

Create `benchmark/validate_edge_fill_barnekow.py`:

```python
"""Before/after evidence for the nodata-aware edge fill on barnekow.

Runs the stem-map prediction on a crop at barnekow's top diagonal boundary
with the fill OFF then ON, and reports:
  * spurious foreground in a near-boundary band (should collapse with fill on),
  * interior foreground (should stay at parity).
Saves a side-by-side PNG. Skips if the model or ortho is missing.

Usage:
  python benchmark/validate_edge_fill_barnekow.py \
      --ortho /path/20220212_Barnekow_4.tiff \
      --model /path/General.onnx \
      --out   /tmp/barnekow_edge_fill.png
"""
import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.windows import Window


def _predict(win_path, model_path, fill_on, workdir):
    # Config overrides are read from this env var at Config load time.
    os.environ["WINMOL_CONFIG_OVERRIDES_JSON"] = json.dumps(
        {"fill_invalid_before_prediction": bool(fill_on)})
    os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"
    os.environ["WINMOL_BATCH_AUTOTUNE"] = "off"
    # Use the known-good standalone entry point (same call the e2e test in
    # tests/test_standalone_pipeline.py exercises): run_pipeline(model, input,
    # pred_dir, output_dir) -> {"stem_map_path", "gpkg_path", "num_stems"}.
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "standalone", "WINMOL_Analyzer.py")
    spec = importlib.util.spec_from_file_location("winmol_standalone", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pred_dir = os.path.join(workdir, "pred_fill%d" % int(fill_on)) + os.sep
    out_dir = os.path.join(workdir, "out_fill%d" % int(fill_on)) + os.sep
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    result = mod.run_pipeline(model_path, win_path, pred_dir, out_dir)
    with rasterio.open(result["stem_map_path"]) as ds:
        return ds.read(1) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/barnekow_edge_fill.png")
    ap.add_argument("--win", type=int, default=3000,
                    help="crop size at the top boundary")
    args = ap.parse_args()

    if not (os.path.exists(args.ortho) and os.path.exists(args.model)):
        print("SKIP: ortho or model missing")
        return 0

    # Crop a window spanning the top diagonal boundary.
    crop = args.ortho + ".topcrop.tif"
    with rasterio.open(args.ortho) as s:
        w = min(args.win, s.width)
        h = min(args.win, s.height)
        window = Window((s.width - w) // 2, 0, w, h)
        data = s.read(window=window)
        prof = s.profile.copy()
        prof.update(width=w, height=h,
                    transform=s.window_transform(window))
    with rasterio.open(crop, "w", **prof) as d:
        d.write(data)
        alpha = data[3] if data.shape[0] >= 4 else np.full((h, w), 255,
                                                            np.uint8)

    workdir = crop + ".work"
    off = _predict(crop, args.model, False, workdir)
    on = _predict(crop, args.model, True, workdir)

    valid = alpha > 0
    # Near-boundary band: valid pixels within 60 px of an invalid pixel.
    from scipy.ndimage import binary_dilation
    invalid = ~valid
    band = binary_dilation(invalid, iterations=60) & valid
    interior = valid & ~band

    def frac(mask_pred, region):
        n = int(region.sum())
        return 0.0 if n == 0 else float((mask_pred & region).sum()) / n

    print("near-boundary fg frac  off=%.5f on=%.5f" % (
        frac(off, band), frac(on, band)))
    print("interior      fg frac  off=%.5f on=%.5f" % (
        frac(off, interior), frac(on, interior)))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        ax[0].imshow(off, cmap="gray")
        ax[0].set_title("fill OFF (artifacts at boundary)")
        ax[1].imshow(on, cmap="gray")
        ax[1].set_title("fill ON")
        for a in ax:
            a.axis("off")
        fig.savefig(args.out, dpi=120, bbox_inches="tight")
        print("wrote", args.out)
    except Exception as exc:
        print("figure skipped:", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint the script**

Run: `/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m flake8 benchmark/validate_edge_fill_barnekow.py`
Expected: no output (clean).

- [ ] **Step 3: Run it on barnekow**

Run:
```bash
/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python \
  benchmark/validate_edge_fill_barnekow.py \
  --ortho "/Users/christian/data/Winmol/Winmol Orthos/orthomosaics_storm_CW/20220212_Barnekow_4.tiff" \
  --model standalone/model_onnx/General.onnx \
  --out /tmp/barnekow_edge_fill.png
```
Expected: prints `near-boundary fg frac off=<hi> on=<~0>` and `interior fg frac off≈on`, and writes the PNG. Acceptance: near-boundary fg collapses with fill on; interior stays at parity. (Adjust `--model` to whichever variant is on disk under `standalone/model_onnx/`.)

- [ ] **Step 4: Commit**

```bash
git add benchmark/validate_edge_fill_barnekow.py
git commit -m "bench: barnekow before/after validation for edge fill"
```

---

## Notes for the executor

- Task 5 uses the known-good `standalone.WINMOL_Analyzer.run_pipeline` entry
  point (the same call `tests/test_standalone_pipeline.py` exercises). The
  numeric acceptance (near-boundary fg → ~0, interior parity) and the figure
  are the deliverable.
- Do NOT push a release tag. This branch is validation-only until the user
  decides otherwise.
