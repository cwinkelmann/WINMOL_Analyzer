# WINMOL PyTorch → Analyzer bridge (winmol_unet)

**Last verified:** 2026-07-09 against this repo's source (`utils/IO.py`, `utils/Prediction.py`, `utils/PredictWorkers.py`, `config.json`).

## What this is

The tree-stem segmentation U-Net is being re-implemented in **PyTorch** in the sibling repo
`WINMOL_segmentor_pt` (package `winmol_unet`). That repo can export a trained PyTorch model
into a form this analyzer consumes, so the analyzer can run PyTorch-trained models without
abandoning its existing TensorFlow-based `.hdf5` models. There are two export targets.

## Path A — Keras HDF5 drop-in (works with this analyzer UNMODIFIED) ✅

`winmol_unet.export_keras.export_to_keras_hdf5(torch_model, path)` builds a mirror Keras
U-Net, copies the PyTorch weights into it, and saves a standard `.hdf5`. The resulting file
is a drop-in replacement for the existing Zenodo models — **no analyzer code change**.

Why it works here, confirmed against source:

- **Loading:** `utils/IO.load_model_from_path` (line 186) calls
  `keras.models.load_model(model_path, compile=False)` (line 208). The exported file uses
  **standard Keras layers only**, so it loads on that first attempt — it does not even need
  the custom-layer fallback (lines 212–216) that some legacy models require for their
  `Conv2DTranspose(groups=…)` / float-seed `Dropout` quirks.
- **Inference:** every prediction site calls `model.predict_on_batch(tile_tensor)`
  (`Prediction.py:168, 609, 742`; `PredictWorkers.py:152`). The exported model is a genuine
  `tf.keras.Model`, so this works natively — nothing to adapt.
- **I/O layout:** the model is **NHWC**, input `(512, 512, 3)` → output `(512, 512, 1)`
  sigmoid. This matches what `_prepare_inference_batch` feeds and the analyzer's channel-last
  indexing `pred[idx, crop:…, crop:…, 0]` (`Prediction.py:173`, `615`, `743`;
  `PredictWorkers.py:157`) expects.
- **Tile size:** `config.img_width/height` = **512**; production models are
  `model_UNet_..._512_...hdf5`. The contract pins 512.

Numerical faithfulness is enforced in `winmol_unet` by `test_export_parity_torch_vs_keras`:
the loaded Keras output matches `torch.sigmoid(unet(x))` within **atol=1e-4**. Two details
make that hold and must not regress: conv kernels are transposed for the NHWC
`[kh,kw,in,out]` convention, and Keras `BatchNormalization` epsilon is pinned to **1e-5** to
match PyTorch `nn.BatchNorm2d` (Keras default 1e-3 would break parity). Weight transfer
fails loudly on any layer-count/shape mismatch.

### "Full model" vs `include_optimizer` — does the export need the optimizer?

No. In the original R segmentor (`WINMOL_segmentor`) the production models were saved by the
training checkpoint callback (`callbacks.R:4-6`, `save_best_only=TRUE`, no
`save_weights_only`), so Keras wrote a **full model**: architecture + weights + optimizer +
the custom compile objects (`F1Score_loss`, `Precision`, `Recall`, `F1Score`). The
`save_model_hdf5(..., include_optimizer = TRUE)` line often cited is actually **commented
out** in `main_training.R:69` — but it would have produced the same full model.

"Full model" is really two things:

- **Architecture + weights** — REQUIRED, because this analyzer calls `keras.models.load_model`
  (not `load_weights`). The PyTorch→Keras export provides this: `build_keras_unet()` +
  `model.save(save_format="h5")` writes a complete, self-contained model. It is a full model
  in the sense that matters. (It must never be a weights-only file — it isn't.)
- **Optimizer state + compile info (loss/metrics)** — what `include_optimizer=TRUE` adds. Used
  only to *resume training* or run Keras `.evaluate()/.fit()`. **Irrelevant to inference.**

Both consumers load with `compile=False` — this analyzer (`IO.py:208`) and the original R
predictor (`main_prediction.R:23`) — which discards the optimizer and compile objects on load.
So the export omitting them changes nothing for the analyzer. It is actually *cleaner*: the
original full models carry a custom loss/metrics and a `Conv2DTranspose(groups=…)` that force
this analyzer's `compile=False` + custom-layer fallback (`IO.py:197-216`); the exported model
uses only standard layers and no custom objects, so it loads on the first attempt.

The only thing the export cannot do is *resume R/Keras training* — but training is moving to
PyTorch, so that path is retired by design.

## Path B — ONNX (wired in) ✅

`winmol_unet.export_to_onnx` + `winmol_unet.runtime.OnnxSegmenter` provide an `onnxruntime`
inference path that duck-types the Keras model (same `predict_on_batch` / NHWC boundary, with
OOM normalized to a retryable exception). **This is the path for non-UNet architectures**
(DeepLabV3+, HRNet): the Path A Keras mirror is UNet-specific, so `winmol_unet` exports those
architectures to ONNX only (see `training/model_factory.py`).

`utils/IO.load_model_from_path` now dispatches on extension: `.onnx` →
`_load_onnx_model` → `winmol_unet.runtime.OnnxSegmenter(model_path)`; everything else →
the existing `keras.models.load_model` path (unchanged). Because `OnnxSegmenter` exposes
`predict_on_batch(NHWC)`, nothing else in the analyzer changes — run it exactly like an
`.hdf5`: `python winmol_run.py model.onnx input.tif stem_map.tif out Trees`.

**Requirement:** the `winmol_unet` package (which pulls in `onnxruntime`) must be installed
in the analyzer env — `pip install -e /path/to/WINMOL_segmentor_pt`. If it is missing,
`_load_onnx_model` raises a clear `RuntimeError` telling you to install it. Covered by
`tests/test_load_model_onnx.py` (dispatch + missing-package error) and verified end-to-end by
loading an exported DeepLabV3+ ONNX and running `predict_on_batch` (NHWC → `(N,512,512,1)`).

**Dependency caveat:** installing `onnxruntime`/`onnx` alongside TensorFlow 2.16 upgrades
`ml-dtypes` past TF's declared `~=0.3.1` pin (pip prints a resolver warning). In practice TF
2.16.2 still imports, still sees the Metal GPU, and still computes on it with `ml-dtypes`
0.5.4 (verified), so the warning is currently benign — but pin `onnx`/`ml-dtypes` if you hit
numeric issues on the Keras path.

**Apple acceleration (CoreML):** `OnnxSegmenter._default_providers()` now prefers
`CoreMLExecutionProvider` on macOS (Apple GPU + Neural Engine) when no CUDA is present,
falling back to CPU for unsupported subgraphs. Measured on this machine with an exported
DeepLabV3+ (real orthomosaic tiles, batch 8): **CoreML 15.4 ms/tile vs CPU 198.8 ms/tile —
~13× faster**, with CoreML offloading 119/126 graph nodes.

Parity: CoreML computes some ops in fp16, so outputs differ from the CPU fp32 reference by a
**max ~3e-4** (mean ~9e-5) per-pixel probability. That is far below the 0.5 stem threshold's
sensitivity — a pixel only flips if its true probability sits within ~3e-4 of 0.5 — so the
binary stem mask is effectively identical for a trained model. When exact fp32 parity is
required (e.g. validating against the PyTorch `atol=1e-4` reference), force CPU:

- `WINMOL_ONNX_FORCE_CPU=1` — CPU only.
- `WINMOL_ONNX_PROVIDERS="CPUExecutionProvider"` — pin an explicit provider list.

UNet keeps its separate fast Keras+Metal path (unchanged).

## Normalization caveat (applies to both paths)

The analyzer feeds RGB normalized to `[0, 1]` (see `IO.load_orthomosaic`, `/255`). A
PyTorch-trained model only produces correct predictions if it was **trained on the same
`[0, 1]` normalization**. Export parity guarantees the exported file reproduces the PyTorch
model's outputs bit-for-bit-close; it does not guarantee the PyTorch model was trained
consistently with this pipeline. (Moot today: the PyTorch **training** package does not exist
yet — `winmol_unet` currently ships the model definition + both export paths, not a trainer.)

## Cross-repo contract

`winmol_unet/contract.py` is the frozen interface (NCHW ONNX graph, dynamic batch, fixed 512,
opset 17, sigmoid baked in). Any change to it is a breaking change requiring coordinated
updates here and a re-run of the parity suites in both repos. This analyzer pip-installs
`winmol_unet` (editable during dev) only when adopting Path B; Path A needs nothing installed
here beyond the existing TensorFlow.
