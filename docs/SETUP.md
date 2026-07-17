# Setup

WINMOL Analyzer runs inference with **ONNX models via onnxruntime — no
TensorFlow**. Any recent Python 3.9–3.12 works; onnxruntime and the geo stack
(rasterio/geopandas/shapely) ship wheels for macOS, Linux and Windows.

## Standalone / CLI

```shell
python -m venv .venv && source .venv/bin/activate    # 3.9–3.12
pip install -r requirements/base.txt                 # geo stack + onnxruntime
```

Run the pipeline (model is `.onnx`; `.hdf5`/`.keras` also load if you separately
install TensorFlow):

```shell
python -u winmol_run.py <model.onnx> <input.tif> <stem_map.tif> <out_prefix> <Stems|Trees|Nodes>
```

Apple Silicon / NVIDIA GPUs are used automatically by onnxruntime (CoreML /
CUDA execution providers). Force CPU with `WINMOL_ONNX_FORCE_CPU=1`, or pin
providers with `WINMOL_ONNX_PROVIDERS=...`.

## QGIS plugin

The plugin creates its own environment (onnxruntime + geo stack) on first use,
**or** you can point it at an existing interpreter (a conda env / any venv with
`requirements/plugin.txt` installed) via the plugin setting
`winmol/python_executable` — the robust option if the auto-setup is blocked by a
firewall / offline machine.

Install from the release ZIP (QGIS → Plugins → Install from ZIP), or for local
development `make deploy` (copies the full plugin, including the compute core,
into your QGIS profile).

## Converting Keras (.hdf5) models to ONNX (dev-only)

The four Zenodo Keras models are converted once, offline:

```shell
pip install -r requirements/convert.txt      # TensorFlow + tf2onnx (dev only)
PYTHONHASHSEED=0 python scripts/convert_models_to_onnx.py --out-dir standalone/model_onnx
```

See `docs/onnx-conversion-parity.md` for the parity results.
