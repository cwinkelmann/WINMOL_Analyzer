# Full-orthomosaic benchmark — three working models

Input: `repr_20250327_171_4_Kraking_Windwurf_SELEKTION.tif`
(10528×7252 px, 4 cm GSD, ~425×293 m). Runs sequential (one GPU job at a
time), 2026-07-11, Apple Silicon (Metal for Keras, CoreML for ONNX),
`winmol_run.py … Trees`, defaults. Prediction = 580 tiles (512² model
input, resampled 15 m tiles); vector phase = 10–11 foreground tiles of
4096² + halo; merge with 12 m edge buffer.

## Wall time per phase (seconds)

| phase | zenodo U-Net (HDF5/Metal) | twostage U-Net (ONNX/CoreML) | twostage DeepLab (ONNX/CoreML) |
|---|---|---|---|
| prediction (580 tiles) | **308.0** | **220.7** | **53.6** |
| vector phase (10–11 tiles) | 182.2 | 177.9 | 173.3 |
| merge + overhead | 5.3 | 6.3 | 9.2 |
| **total** | **495.5** | **405.0** | **236.1** |
| script wall (incl. TF/ORT init) | 498 | 406 | 238 |

## Prediction detail (per 512² tile, averaged)

| metric | zenodo | twostage U-Net | twostage DeepLab |
|---|---|---|---|
| read | 0.019 s | 0.020 s | 0.023 s |
| prep | 0.005 s | 0.006 s | 0.007 s |
| **infer** | **0.290 s** | **0.253 s** | **0.036 s** |
| write | ~0 | ~0 | ~0 |
| autotuned batch | 10 | 4 | 4 |
| throughput | 113 tiles/min | 158 tiles/min | 654 tiles/min |

Inference dominates prediction; I/O is negligible. DeepLabV3+ on CoreML is
**7–8× faster per tile** than either U-Net (119/126 graph nodes offloaded to
the Apple Neural Engine; the U-Nets' transposed convolutions fall back more).

## Vector phase detail (per 4096² tile, averaged)

| stage | zenodo | twostage U-Net | twostage DeepLab |
|---|---|---|---|
| skeletonize + segments | 8.2 s | 8.7 s | 7.9 s |
| quantify (diameters/volumes) | 7.1 s | 7.8 s | 7.0 s |
| connect stems | 1.1 s | 1.1 s | 0.8 s |
| **total per tile** | **16.4 s** | **17.7 s** | **15.7 s** |

The vector phase is nearly model-independent (~175–180 s regardless of
detector) and is the fixed floor of the pipeline. Its two dominant costs are
skeletonization (~50 %) and quantification (~43 %) — both with confirmed
optimization headroom (CODE_REVIEW_2 §C: the ~800 px padding, per-pixel
`get_neighbors`, full-raster `features.shapes`).

## Results

| result | zenodo | twostage U-Net | twostage DeepLab |
|---|---|---|---|
| stems (pre-merge, all tiles) | 1074 | 1061 | 993 |
| **stems (final, merged)** | **749** | **735** | **671** |
| nodes | 12349 | 10815 | 10308 |
| total length | 7400 m | 6293 m | 5868 m |
| total volume | **934 m³** | **481 m³** | **413 m³** |
| merge edge candidates / joins | 150 / 0 | 130 / **1** | 125 / 0 |

Stem counts agree within ~10 %; **volume differs ~2×** — Zenodo predicts
systematically wider stem masks, hence larger diameters. Which is closer to
the truth needs field reference data; the twostage family is internally
consistent (as on the test crop: IoU 0.65–0.68 within family vs ~0.5 against
Zenodo).

## Notes

- The twostage U-Net run **validated the A-1 fix in production**: one
  merge-time seam join fired (`1 stem segments appended`) — the exact
  operation that crashed with IndexError (and destroyed the tile outputs)
  before the 2026-07-11 fix.
- Per-run logs with full timing lines: `standalone/output/full_*/run.log`.
- Side-by-side figure: `docs/three_models_comparison.png`.

---

# Original pipeline vs the full change stack

A different comparison from the model benchmark above: same model weights on
both sides, **the pipeline itself** is what changes.

Input: `20220212_Barnekow_4.tiff` (9610×8662 px, 2.1 cm GSD, ~198×179 m),
2026-07-19, Apple Silicon, `winmol_run.py … Trees`, headless (no QGIS),
**median of 3 runs per configuration** via
`benchmark/bench_orig_vs_changed.py` (see `benchmark/README.md`).

- **original** — `origin/main`, TensorFlow/Keras `.hdf5`, `PYTHONHASHSEED`
  unset, i.e. exactly as shipped
- **changed** — the full stack, ONNX/onnxruntime
- **+threadpool** — the same, with the quantification pool fix (PR #7)

Both sides run the *same weights*: `General.onnx` was converted from
`model_UNet_GenDS_512_2023-02-27_211141.hdf5` at 0.0000 % binary-mask
disagreement (`onnx-conversion-parity.md`), so output differences come from the
pipeline, not the model.

## Results

| | original | changed | changed + threadpool |
|---|---|---|---|
| wall time (median) | 793.0 s | 114.2 s | **76.5 s** |
| per-run wall | 651 / 903 / 793 | 97 / 114 / 115 | 74.7 / 76.5 / 77.3 |
| stems | 449 | 484 | **484** |
| per-run stems | **446 / 451 / 449** | 484 / 484 / 484 | 484 / 484 / 484 |
| volume | 400.661 m³ | 439.228 m³ | **439.228 m³** |
| identical geometry across runs | **no — 3 distinct** | yes | yes |

**10.4× faster end to end**, of which 1.5× comes from the threadpool fix alone.

## What the numbers mean

- **Reproducibility is the headline for anyone doing science with this.** The
  original returned three different stem counts and three different geometries
  from three identical runs of one file — a ±5-stem noise floor that any
  before/after comparison was being read against. That is now zero.
- **The +35 stems are recovered detections, not a different model.** Same
  weights on both sides, so the difference is the ortho-boundary edge fix
  recovering stems the tiled merge had been discarding. ~39 m³ of timber on a
  single 3.5 ha site.
- **The threadpool fix is provably output-preserving**: stem count *and* total
  volume are byte-identical to the run before it (484, 439.228 m³). It is
  purely a speedup.
- **Timing also steadied.** The original swung 651–903 s (32 %); the fixed
  stack runs 74.7–77.3 s (3 %).

## Caveats

- Wall time is an **end-to-end old-stack-vs-new-stack** figure: it includes the
  TensorFlow → onnxruntime change as well as the pipeline work, because that is
  what a user experiences after upgrading. It is not an isolated measurement of
  the pipeline edits.
- Apple Silicon, where the vector phase dominates. On a CUDA machine inference
  nearly vanishes and the profile changes *shape*, so the ratio will differ.
  The harness is written to run there — see `benchmark/README.md`.
- A tile-level extrapolation predicted ~125 s for the threadpool fix; the
  measured 76.5 s is better than predicted, and the reason has not been
  isolated. Treat the tile→ortho arithmetic as a lower bound, not a model.
