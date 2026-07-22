# WINMOL golden-master test suite

Characterization tests that pin the **current** behavior of every pipeline
stage. One deterministic reference run on a small crop (41 m², 62 stems,
31 crossing pairs — deliberately chosen for overlapping trees) produced the
fixtures in `tests/fixtures/`; each test module replays **one** stage using
the upstream fixture as input and compares against the golden output.

Consequence: a code change that alters any stage's output makes exactly that
stage's test fail. Intentional behavior changes (bug fixes!) must regenerate
the fixtures **in the same commit**, making the change reviewable as a
fixture diff. Known-buggy behavior that the fixtures pin on purpose is
annotated with review IDs from `docs/CODE_REVIEW.md` / `docs/CODE_REVIEW_2.md`.

## Running

```bash
pytest                    # everything (~2-3 min with the model present)
pytest -m "not slow"      # fast loop: stage tests only, no TF (<30 s)
```

The tools come from `requirements/dev.txt` (flake8 + pytest), installed on top
of a normal `requirements/cpu.txt` environment. `requirements/ci.txt` is the
exactly-pinned alternative — see `requirements/README.md`.

`tests/test_standalone_pipeline.py` is the one module that imports TensorFlow at
module scope; CI runs with `--ignore` for it. To run it locally install
`requirements/convert.txt` (which pins TF 2.16.2), plus `tensorflow-metal==1.2.0`
on Apple Silicon — the test skips itself without a TF-visible GPU.

The conftest re-execs pytest with `PYTHONHASHSEED=0` if needed — pipeline
output depends on set-iteration order (review A-4), so the fixtures are only
reproducible under that seed.

## Test map

| Module | Stage under test | Input | Golden |
|---|---|---|---|
| `test_units.py` | small pure functions (incl. pinned quirks) | — | — |
| `test_inference.py` [slow] | `predict_with_resampling_per_tile` | `crop_input.tif` + UNet HDF5 | `stem_map.tif` (≥99.9 % agreement) |
| `test_skeletonization.py` | `find_segments` | `stem_map.tif` | `stage_find_segments` |
| `test_vectorization.py` | `restore_geoinformation`, `build_stem_parts`, `connect_stems` | previous stage fixture | corresponding fixture |
| `test_quantification.py` | `quantify_stems` (contour + EDT) | `stage_connect_stems` | `stage_quantified_*` |
| `test_io_exports.py` | `stems_to_gdf` / `nodes_to_gdf` / gpkg writer | `stage_quantified_contour` | `golden_stems.gpkg` |
| `test_tiled_pipeline.py` [slow] | `build_tile_grid`, per-tile vectorize + merge | `stem_map.tif` | `tile_grid.json`, `golden_merged.gpkg` |
| `test_e2e_crop.py` [slow] | standalone `run_pipeline` | `crop_input.tif` | `golden_stems.gpkg` (tolerant) |

Only the [slow] tests need TensorFlow and the 374 MB model
(`standalone/model/model_UNet_GenDS_512_2023-02-27_211141.hdf5`); they skip
cleanly when it is absent. Everything else runs from the committed fixtures.

## CI

`.github/workflows/tests.yml` runs the suite on every PR and on pushes to
main, inside the container built by `.github/workflows/ci-image.yml`
(`ghcr.io/<owner>/winmol-analyzer-ci`). The image
(`docker/ci/Dockerfile`) carries the pinned test environment
(`requirements/ci.txt`) **and the Zenodo GenDS model** (sha256-verified at
build) at `/opt/winmol/model/` — no model file in git. Two jobs:

- **fast-tests** — fixture-based stage tests, no model (~2 min);
- **full-tests** — inference vs the golden stem map, tiled merge, e2e, and
  conformance for the one model present in the image (local-only manifest
  entries skip).

**Bootstrap:** the image workflow triggers on changes to
`docker/ci/**` / `requirements/{core,ci}.txt`, or manually via
*Actions → CI test image → Run workflow*. On the very first push the tests
job may race the image build — run the image workflow first (or re-run the
failed tests job once the image exists). Rebuilding the image is only needed
when dependencies or the baked model change.

## Model conformance (verifying NEW models against the same pipeline)

`test_model_conformance.py` proves that different models "work the same
way": same loader entry point, same 512/NHWC/sigmoid contract, same
prediction grid (shape + georeferencing must equal the reference stem
map's), same vector chain, same GPKG structure. Detections may differ
(different weights); pipeline behavior must not.

1. Edit `tests/models_manifest.json` — point the `path` entries at the
   models to verify (mixed `.hdf5` / `.keras` / `.onnx` is fine). Missing
   files are skipped with a notice. Override the manifest location with
   `WINMOL_MODELS_MANIFEST=<path>`.
2. `pytest tests/test_model_conformance.py -v`
3. Read `tests/conformance_report.md` — a side-by-side table
   (loader, timings, stem px %, parts/stems, totals) across all models.

Each model runs in a fresh subprocess (`conformance_driver.py`): mixed
formats cannot share a process because a Keras-2 HDF5 needs
`TF_USE_LEGACY_KERAS=1` while a Keras-3 HDF5 breaks under it — the harness
sniffs the `keras_version` HDF5 attribute and sets the env per model.

## Regenerating fixtures

```bash
PYTHONHASHSEED=0 python tests/generate_fixtures.py
```

- The script runs the whole vector chain **twice** and refuses to write
  unless both passes are identical (determinism guard).
- Re-runs are stable: unchanged content is not rewritten (the stem map is
  kept when a fresh prediction agrees ≥ 99.9 %, since TF inference is not
  bit-deterministic across devices).
- The crop window is auto-selected for overlapping trees: candidates ranked
  by predicted stem density, then by crossings among **connected** stems.
- `config_snapshot.json` freezes every Config value the fixtures were built
  with (serial workers, autotune off); tests always build their Config from
  it, never from bare `Config()`.
- `manifest.json` records the model hash, window, versions, and counts.

GPKG fixtures are compared content-wise (via pyogrio, order-insensitive,
`stem_id` ignored); their raw bytes may differ between regenerations because
GeoPackage embeds timestamps.
