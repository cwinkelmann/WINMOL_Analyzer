# Which requirements file do I install?

One file, once. Pick the row that describes you and stop reading.

| I want to…                                        | install                                            |
| ------------------------------------------------- | -------------------------------------------------- |
| run the analyzer (CLI or QGIS plugin, any OS)      | `pip install -r requirements/cpu.txt`               |
| …and I have an NVIDIA GPU (Linux/Windows x86_64)   | `pip install -r requirements/gpu.txt`               |
| …add CUDA to an environment I already have         | `pip install -r requirements/cuda.txt` — uninstall `onnxruntime` first, see [docs/GPU.md](../docs/GPU.md) |
| run the standalone Jupyter notebooks               | `pip install -r requirements/notebook.txt`          |
| reproduce the CI test environment exactly          | `pip install -r requirements/ci.txt`                |
| convert an `.hdf5` model to ONNX (dev only)        | `pip install -r requirements/convert.txt`           |
| lint and run the test suite locally                | `pip install -r requirements/cpu.txt -r requirements/dev.txt` |

`core.txt` is not in the table on purpose: it is a shared fragment that every
other file pulls in with `-r`, and it installs no inference runtime at all.
Installing it on its own gives you an environment that cannot run the model.

## The two rules

**1. Exactly one inference runtime.** `onnxruntime` (CPU) and `onnxruntime-gpu`
(CUDA) both provide the `onnxruntime` Python module. With both installed, one
shadows the other's shared libraries and imports fail in ways that look like a
broken CUDA install. So `cpu.txt` and `gpu.txt`/`cuda.txt` are mutually
exclusive — swap, never add. `plugin_utils/installer.py` uninstalls the other
distribution before installing either, and
`tests/test_requirements_layout.py` fails the build if any file in this
directory ever pulls in both.

**2. No TensorFlow anywhere except `convert.txt`.** The analyzer runs ONNX
models through onnxruntime. TensorFlow is needed only to *convert* a legacy
Keras `.hdf5` model into one.

## The layout

```
core.txt      geo/raster stack, no runtime      <- shared fragment
  ├── cpu.txt       + onnxruntime + psutil      <- CLI, plugin venv, standalone
  │     └── notebook.txt  + ipykernel, matplotlib
  ├── gpu.txt       + cuda.txt + psutil         <- CUDA container, plugin GPU venv
  │     └── cuda.txt      onnxruntime-gpu[cuda,cudnn]>=1.26,<1.27
  └── ci.txt        + exact pins + pytest       <- the CI image only

convert.txt   TensorFlow + tf2onnx              <- standalone tool, throwaway env
dev.txt       flake8 + pytest                   <- tooling, add to any of the above
```

Read off the algebra: `cpu = core + onnxruntime`, `gpu = core + cuda`,
`notebook = cpu + plots`, `ci = core + exact pins`. Nothing installable inherits
a runtime it did not ask for, because the only shared ancestor has none.

Every file starts with a header saying who installs it and when. If you add a
file here, add it to the table above too — the layout test checks that it is
listed.
