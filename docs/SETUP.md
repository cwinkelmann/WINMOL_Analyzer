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

### Where the plugin's environment lives — and how to remove it

Everything the plugin builds for itself lives **outside** the plugin folder,
under the QGIS profile directory:

```
<QGIS profile>/winmol/
├── winmol_venv/            the managed virtual environment (~1–2 GB)
├── py311/                  a downloaded Python 3.11 runtime, if one was needed
└── .winmol_autotune.json   the batch-size autotune cache
<QGIS profile>/python/plugins/WINMOL_Analyzer/
└── models/                 downloaded .onnx models (31 MB – 375 MB each)
```

`<QGIS profile>` is what *Settings → User Profiles → Open Active Profile
Folder* opens (e.g. `~/Library/Application Support/QGIS/QGIS3/profiles/default`
on macOS, `~/.local/share/QGIS/QGIS3/profiles/default` on Linux,
`%APPDATA%\QGIS\QGIS3\profiles\default` on Windows).

The venv sits beside the plugin folder rather than inside it on purpose: QGIS
uninstalls a plugin by recursively deleting its folder, and that delete used to
fail on the venv's symlinks.

> **Uninstalling the plugin does NOT remove `<profile>/winmol`.** This is not
> an oversight — the QGIS plugin API has no uninstall hook.
> `pyplugin_installer/installer.py::uninstallPlugin` is `unloadPlugin(key)`
> followed by `removeDir(<profile>/python/plugins/WINMOL_Analyzer)`, and
> `unloadPlugin` calls the plugin's `unload()`, which QGIS also calls on
> *disable*, on *reload* and at *application shutdown*. Nothing in that
> callback can tell an uninstall from QGIS simply closing, so deleting the
> environment there would wipe a multi-gigabyte venv every time you quit.

Two ways to remove it, both explicit:

* **Setup tab → “Delete environment…”** — pick venv / downloaded runtime /
  downloaded models, then confirm an itemised list with sizes. “Open folder”
  next to it opens `<profile>/winmol` if you would rather look first. Do this
  *before* uninstalling the plugin.
* **By hand**, afterwards:

  ```shell
  rm -rf "<QGIS profile>/winmol"                     # macOS / Linux
  rmdir /s /q "%APPDATA%\QGIS\QGIS3\profiles\default\winmol"   # Windows
  ```

If you want to keep the environment (it is an ordinary venv), copy it
somewhere else first and point `winmol/python_executable` at the copy.

## Converting Keras (.hdf5) models to ONNX (dev-only)

The four Zenodo Keras models are converted once, offline:

```shell
pip install -r requirements/convert.txt      # TensorFlow + tf2onnx (dev only)
PYTHONHASHSEED=0 python scripts/convert_models_to_onnx.py --out-dir standalone/model_onnx
```

See `docs/onnx-conversion-parity.md` for the parity results.
