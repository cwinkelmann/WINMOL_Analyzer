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
