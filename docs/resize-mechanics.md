# The mechanics of resizing — v0.5 parity, kernel by kernel

Review date 2026-08-10. Sources: v0.5.0 tag, this branch (02a8488), the
0.6.1-rc12 handoff (Stefan Reder), `docs/resampling-accuracy.md`, and a
numerical experiment on `20220212_Barnekow_4.tiff` reproducible with
`benchmark/bench_resize_parity.py`.

## The target grid

`tile_size = 15` m across `img_width = 512` px → **2.9297 cm/px**. v0.5.0
resampled **per tile** (a native window of `ceil(15/GSD)-1` px → 512), not
per ortho — `load_orthomosaic_with_resampling` was dead code in its live
path. The model's accuracy collapses away from this grid (a 30 cm stem is
~10 px on it; at 2× the GSD it is ~5 px and detection degrades sharply),
which is why every pipeline must resample to it.

## v0.5.0's exact semantics (the parity definition)

`tf.image.resize(tile, [512, 512], method='bicubic', antialias=False)` on
float32 **after** /255. Mathematically: a fixed 4-tap Catmull-Rom kernel
(a=-0.5), half-pixel centers, no anti-aliasing, out-of-range taps excluded
and renormalized at borders. The validity mask went through
`tf.image.resize(..., 'nearest')` then `> 0.5`.

## Who computes what

| Site | Kernel | AA | dtype at resize | v0.5-equiv |
|---|---|---|---|---|
| v0.5 live tile path (tf bicubic) | Catmull-Rom a=-0.5 | no | float32 | reference |
| this branch, `onnx_gpu` (in-graph Resize) | Catmull-Rom a=-0.5 | no | float32 | **yes, to 4.2e-07** |
| rc12 CUDA RawKernel (CuPy) | Catmull-Rom a=-0.5 | no | float32 | **yes** (see below) |
| this branch, fast default (GDAL in-read cubic) | Catmull-Rom **scaled by decimation factor** | yes | uint8 (rounded) | no |
| + overviews | average, then scaled cubic (2-stage) | yes ×2 | uint8 | no |
| consumer fallback (skimage `order=3`, AA off) | cubic B-spline | no | float32 | no (close) |

"Cubic" names three different convolutions. GDAL's in-read cubic widens its
kernel with the decimation factor — that is anti-aliasing. Measured on
current code, the effect is strongly scale-dependent: +5.6% stems at
Barnekow's 1.42× (470 vs 498), and at R13's 2.29× (fresh A/B, H100,
2026-08-11) **+20.3% stems / +23.2% length / +26.0% volume** (overview
19,186 / 105,023 m / 8,104 m³ / 26.1 min vs graph 15,942 / 85,226 m /
6,433 m³ / 34.9 min). The era context's absolute numbers were phantoms,
but its scale-dependence claim was directionally right. Whether the fast
path's extra detections are real stems or aliasing artifacts is decided by
the training-recipe / ground-truth question — at these magnitudes that
question carries ~1,700 m³ per R13-scale scene. **Validate resampling changes at ≥ 2.3×, never on
Barnekow-class data alone — and only against re-measured baselines.**

## rc12 equivalence (measured 2026-08-10)

The rc12 CUDA kernel replicated bit-for-bit vs the TF-equivalent ONNX
Resize, real Barnekow windows, same fp32 Spruce.onnx:

| Regime | interior max Δ input | 2px border band | binarized-core IoU | total fg |
|---|---|---|---|---|
| 1.42× | 0.0001 DN | 1.9 DN | ≥ 0.99988 | ±0.000% |
| 2.29× (R13 regime) | 0.0135 DN | 4.6 DN | ≥ 0.99990 | +0.004% |

The only difference — rc12 clamps border taps where TF excludes+renormalizes
— dies in the 4 px core crop. **rc12's resize is v0.5-equivalent**; its
open "interpolation equivalence check" is closed by these numbers.

## R13 end-to-end (measured 2026-08-10, RTX 4080, fp32 Spruce 36e5397c)

| Pipeline | stems | total wall |
|---|---:|---:|
| our `graph` (this branch) | 15,930 | 96.7 min |
| rc12 CUDA pipeline | 15,969 | 117 min |
| rc12 with his fp16 Spruce (his report) | 15,935 / 15,954 | — |

Two independent implementations of the v0.5 kernel — different resize
engineering, different vector stages — agree to **0.24 %** in stems and
**IoU 0.99877** at the prediction-raster level (fg ±0.004 %). fp16 vs fp32
was 0.2 % — never the explanation for anything.

**The era A/B table (12,714 "v0.5" / 12,722 "onnx_gpu" / 15,556 fullres /
16,548 overview, and "455 on every variant" on Barnekow) did not survive
re-measurement** and must not be cited as a baseline: current re-runs give
~15.9k for v0.5 semantics on R13 and 470–498 on Barnekow depending on
kernel family. The honest kernel effect is a few percent (AA vs no-AA),
not 30 %. A reproduction run of the era measurement at its own commit
(9e7ff5e, `reimpl/docker-batch`) is in progress to pin down what that
context did differently; until it lands, treat era numbers as unattributed.

Also measured: **ONNX Runtime 1.19.2's CUDA EP computes the opset-18
antialias Resize incorrectly** (graph_aa: 82 stems on CUDA EP vs 478 on
CPU EP, same run, CPU-side kernel verified sane against GDAL-AA to
0.01 DN mean). graph_aa therefore pins CPUExecutionProvider (a8b6a33);
report upstream, retest on newer ORT.

## Portability matrix — which kernel actually runs where

| Solution | NVIDIA/CUDA | Apple Silicon | CPU-only |
|---|---|---|---|
| `onnx_gpu` (Resize in the graph) | CR no-AA ✅ | CR no-AA ✅ (node may partition to CPU EP — same bits) | CR no-AA ✅ |
| fast default (GDAL) | scaled-cubic AA ❌ | scaled-cubic AA ❌ | scaled-cubic AA ❌ (consistent, but not v0.5) |
| rc12 `gpu` backend (CuPy) | CR no-AA ✅ | **unavailable → falls back to GDAL-AA ❌** | **unavailable → GDAL-AA / skimage ❌** |

CuPy exists only for CUDA. rc12's `auto` mode therefore swaps interpolation
families silently on non-NVIDIA machines: a mixed fleet returns
systematically different stem counts from the same ortho. The in-graph
Resize travels with the model — parity is a property of the artifact, not
of the driver stack.

## Recommendations

1. When v0.5-comparable output matters, run the in-graph path — since the
   `prediction_read_strategy` flag landed it IS the default (`graph`;
   ≈1.5× the wall-clock of the fast path). The GDAL/skimage/cupy variants
   stay selectable through the same flag or `WINMOL_BENCH_READ` for A/B —
   possibly *more* accurate, but unresolvable until the training-time
   resize recipe or field ground truth exists.
2. Ask the training side for the training-pipeline resize recipe; it
   settles the remaining few-percent AA question (train-faithful vs
   signal-faithful input).
3. From rc12, adopt ideas not code: uint8 through the queue,
   prediction-only D2H, and the installer-time CUDA smoke test. The CuPy
   stack itself is unnecessary — this branch's cliff fix reaches higher
   throughput without it.
4. Gate any future resize change on `bench_resize_parity.py` at a ≥2.3×
   factor.
