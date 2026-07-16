# Zenodo model conversion to ONNX — parity report

The four shipped Zenodo Keras (HDF5) U-Net models are converted to ONNX once,
offline, by `scripts/convert_models_to_onnx.py` (dev-only deps in
`requirements/convert.txt`). The analyzer then runs the `.onnx` via the vendored
`OnnxSegmenter` (onnxruntime) with **no TensorFlow**.

## Method

- `tf2onnx.convert.from_keras(model, opset=17)`, keeping the native **NHWC**
  layout (input `[N,512,512,3]`, sigmoid output `[N,512,512,1]`), graph I/O
  renamed to `input`/`output`. The vendored `OnnxSegmenter` is layout-aware, so
  it feeds these NHWC models directly and still handles NCHW models (the
  PyTorch exports).
- Parity measured on 4 real 512×512 tiles from `tests/fixtures/crop_input.tif`,
  ONNX (onnxruntime CPU) vs Keras (`model.predict`).
- The pipeline consumes the **binarized** mask (`> 0.5`), so the decisive metric
  is threshold disagreement; the raw-probability max diff is a loose sanity
  bound (TF vs onnxruntime fp accumulation across a deep conv net).

## Results (2026-07-16)

| model | raw max-abs-diff | raw mean-diff | **0.5-threshold disagreement** |
|-------|------------------|---------------|--------------------------------|
| General         | 2.637e-04 | 2.2e-08 | **0.0000 %** |
| Beech           | 2.614e-04 | 3.1e-08 | **0.0000 %** |
| Spruce          | 1.873e-04 | 1.9e-08 | **0.0000 %** |
| Spruce_Deadwood | 2.254e-04 | 3.7e-08 | **0.0000 %** |

**Gate: PASS.** Every converted model produces a **bit-identical binary stem
map** to its Keras original on the fixture tiles. The sub-1e-3 raw differences
never cross the 0.5 threshold, so nothing downstream (skeletonization,
diameters, volumes) changes.

## Reproduce

```bash
pip install -r requirements/convert.txt          # dev-only (TF + tf2onnx)
PYTHONHASHSEED=0 python scripts/convert_models_to_onnx.py \
    --out-dir standalone/model_onnx
```

Outputs land in `standalone/model_onnx/` (gitignored, ~124 MB each). Hosting is
deferred — these are uploaded to a durable URL and referenced from
`config.json`; verification (fixtures, CI, compute-contract test) uses the local
files, so hosting is not on the critical path.
