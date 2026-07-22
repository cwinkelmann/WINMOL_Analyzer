# Which requirements file do I install?

One file, once. Pick the row that describes you and stop reading.

| I want to…                                        | install                                            |
| ------------------------------------------------- | -------------------------------------------------- |
| run the analyzer (CLI or QGIS plugin, any OS)      | `pip install -r requirements/cpu.txt`               |
| …and I have an NVIDIA GPU (Linux/Windows x86_64)   | `pip install -r requirements/gpu.txt`               |
| …add CUDA to an environment I already have         | `pip install -r requirements/gpu.txt` — uninstall `onnxruntime` first, see [docs/GPU.md](../docs/GPU.md) |
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
broken CUDA install. So `cpu.txt` and `gpu.txt` are mutually
exclusive — swap, never add. `plugin_utils/installer.py` uninstalls the other
distribution before installing either, and
`tests/test_requirements_layout.py` fails the build if any file in this
directory ever pulls in both.

**2. No TensorFlow anywhere except `convert.txt`.** The analyzer runs ONNX
models through onnxruntime. TensorFlow is needed only to *convert* a legacy
Keras `.hdf5` model into one.

**3. CI installs what ships.** `ci.txt` is `-r cpu.txt` plus `pytest`, so the
environment the test suite runs in is the environment a user gets. See
[The four environments](#the-four-environments).

## The layout

```
core.txt      geo/raster stack, no runtime      <- shared fragment
  ├── cpu.txt       + onnxruntime + psutil      <- CLI, plugin venv, standalone
  │     ├── notebook.txt  + ipykernel, matplotlib
  │     └── ci.txt        + pytest              <- the CI image
  └── gpu.txt       + onnxruntime-gpu + psutil  <- CUDA container, plugin GPU venv

convert.txt   TensorFlow + tf2onnx              <- standalone tool, throwaway env
dev.txt       flake8 + pytest                   <- tooling, add to any of the above
```

Read off the algebra: `cpu = core + onnxruntime`, `gpu = core + onnxruntime-gpu`,
`notebook = cpu + plots`, `ci = cpu + pytest`. Nothing installable inherits
a runtime it did not ask for, because the only shared ancestor has none.

Every file starts with a header saying who installs it and when. If you add a
file here, add it to the table above too — the layout test checks that it is
listed.

There is no `cuda.txt`. It existed to single-source the CUDA version window
between `gpu.txt` and `plugin-gpu.txt`; those are now one file, so the drift it
guarded against cannot happen. It was inlined into `gpu.txt` because every `-r`
names a file that **must exist in the shipped plugin ZIP** at install time —
each hop is another way a mis-packaged archive fails halfway through pip on a
user's machine.

## The four environments

Four places run this code. They are the *same* environment except where this
table says otherwise, and each difference below is deliberate.

| environment | built from | how | differs from the shipped environment |
| --- | --- | --- | --- |
| **CLI / local** | `cpu.txt` (or `gpu.txt`) | `pip install -r …` by hand | — it *is* the shipped environment |
| **QGIS plugin** | `cpu.txt`, or `gpu.txt` when `nvidia-smi` reports a usable driver | `plugin_utils/installer.py` into `winmol_venv` | nothing; same files, same ranges |
| **Docker** | `docker/ci/Dockerfile` → `ci.txt`; `docker/gpu/Dockerfile` → `gpu.txt` | image build | CI image adds `pytest`; GPU image adds nothing |
| **CI** | `ci.txt` = `cpu.txt` + `pytest` | `docker/ci/Dockerfile`, run by `.github/workflows/tests.yml` | `pytest` only |

**The CPU/CI difference is `pytest` and nothing else.**
`tests/test_ci_parity.py` re-derives the `-r` closure of both files on every
run and fails with the name of the offending package if any requirement is
declared differently, or if `ci.txt` grows an addition that is not on its
test-only allowlist. `.github/workflows/plugin-env.yml` separately does a real
`pip install` of `cpu.txt` on Linux, Windows and macOS, so the resolve itself
is exercised on all three platforms.

**The GPU path holds by construction.** `docker/gpu/Dockerfile` and the
plugin's GPU venv install *the same file*, `gpu.txt` — there is no second
declaration that could drift. This is what the `plugin-gpu.txt` merge bought:
before it, the container's copy of the CUDA window had already lost its `<1.27`
ceiling while the plugin's kept it.

**What CI still cannot see.** The golden-master suite runs in a Linux x86_64
container, so it cannot catch a Windows or macOS bug, and it has no GPU — the
CUDA provider is verified at image-build time by loading the shared objects
(`docker/gpu/Dockerfile`), which exercises the linker but not a device.
`plugin-env.yml` covers the per-OS install; hardware coverage remains manual.

### History

`ci.txt` used to hang off `core.txt` with its own pins, and the divergence was
measured, not theoretical: CI installed `onnxruntime==1.19.2` while the
plugin's venv resolved `1.27.0` the same day, and CI installed no `psutil` at
all — the package `HardwareInfo` needs to report RAM and `utils/Prediction.py`
needs for the autotune memory ceiling. That safety code was reachable in CI
only through test stubs. The rationale for the split (fixture reproducibility)
did not survive checking; `ci.txt`'s own header records what was checked and
what the fixtures actually depend on.
