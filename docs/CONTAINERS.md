# Containers

Four kinds of image, four jobs. They are not interchangeable, and the
differences matter.

| Image | Built by | Purpose | Inference |
|---|---|---|---|
| `winmol-analyzer-ci` | `.github/workflows/ci-image.yml` | runs the test suite | **CPU, forced** |
| `winmol-analyzer-gpu` | `.github/workflows/gpu-image.yml` | production / batch on CUDA | GPU |
| root `Dockerfile` | not built in CI | QGIS plugin GUI testing over X11 | none — no TensorFlow, no ONNX runtime of its own |
| root `Dockerfile-1`, `Dockerfile.blackwell`, `Dockerfile_carrot`, `Dockerfile_olive` | not built, untracked | legacy, machine-specific | TensorFlow |

The tracked root `Dockerfile` used to be filed here as a legacy TensorFlow
image. It is not one: it installs no Python packages beyond `python3-venv` /
`python3-pip`, and it is the supported way to test the plugin against a QGIS
version you do not want on your workstation.

## The CUDA image

Published to `ghcr.io/<owner>/winmol-analyzer-gpu` **on every tag push**
(`v1.2.3` → `:v1.2.3` and `:latest`). Pull requests touching `docker/gpu/**` or
the GPU requirements build it **without** pushing, so a broken Dockerfile fails
in review rather than at release time.

```bash
docker pull ghcr.io/cwinkelmann/winmol-analyzer-gpu:latest

docker run --rm --gpus all \
  -v /path/to/models:/models:ro \
  -v /path/to/orthos:/input:ro \
  -v /path/to/results:/output \
  ghcr.io/cwinkelmann/winmol-analyzer-gpu:latest General
```

Full build/run/troubleshooting detail: [`docker/gpu/README.md`](../docker/gpu/README.md).

### It is TensorFlow-free, and that is the point

The analyzer runs ONNX through `onnxruntime`. TensorFlow is needed only to
*convert* models (`requirements/convert.txt`). Dropping it is what lets the
image use a `python:3.11-slim` base instead of a multi-gigabyte CUDA/TF one —
the older Dockerfiles carried that weight purely to satisfy
`tensorflow[and-cuda]`.

The CUDA userspace arrives as wheels via `onnxruntime-gpu[cuda,cudnn]`; the host
driver and `nvidia-container-toolkit` supply the rest. That avoids the
base-image-versus-wheel CUDA mismatch the old files fought by hand —
`Dockerfile_olive` literally `pip uninstall`s `nvidia-cudnn-cu11` to unpick one.
**Take CUDA from one source, not two.**

### Verify the GPU before trusting any timing

onnxruntime falls back to CPU **silently** when the CUDA provider fails to load.
That reads as "the GPU is slow", not "the GPU never ran".

```bash
docker run --rm --gpus all --entrypoint python \
  ghcr.io/cwinkelmann/winmol-analyzer-gpu:latest -c \
  "import onnxruntime as o; print(o.get_available_providers())"
```

Must list `CUDAExecutionProvider`. The release workflow asserts this too, but on
a CPU runner it can only confirm the GPU *build* is installed — actual CUDA
execution can only be verified on the target machine.

### Correctness before speed

| Input | Expected |
|---|---|
| Barnekow ortho, `General` | **484 stems, 439.228 m³** |

Repeated runs must give **identical** counts — that is what `PYTHONHASHSEED=0`,
baked into the image, guarantees. A faster wrong answer is not a result.

## The CI image

`ghcr.io/<owner>/winmol-analyzer-ci`, consumed by `tests.yml` as a job
container. Two-stage: TensorFlow lives only in a throwaway convert stage; the
runtime is `python:3.11-slim` with `General.onnx` baked in at
`/opt/winmol/model_onnx/`, so no model ever enters git.

It sets **`WINMOL_ONNX_FORCE_CPU=1` deliberately** — the golden fixtures are
bit-reproducible only on CPU. Never copy that into a GPU image; it would
silently disable the hardware the image exists for.

## The QGIS GUI image

Root `Dockerfile` + `startDocker.sh`. It runs QGIS itself, with this repo
bind-mounted as the plugin and the GUI passed out to the host's X server.

```bash
./startDocker.sh                        # current LTR, builds on first run
QGIS_TAG=4.2.0-trixie ./startDocker.sh  # trial QGIS 4 / Qt6 — read below first
```

The QGIS version is a build arg, so bumping it never means editing a file. The
image name carries the tag (`winmol_analyser_docker:<tag>`), so a 3.44 and a
4.2 image coexist instead of clobbering each other.

### Which tag

Verified against `hub.docker.com/v2/repositories/qgis/qgis/tags` on 2026-07-21
(441 tags):

| Tag | Resolves to | Notes |
|---|---|---|
| `3.44.12-noble` | Ubuntu 24.04 LTS | **the default**; current LTR |
| `3.44.12-trixie` | Debian 13 | same QGIS, newer base |
| `ltr`, `3.44`, `3.44.12` | Ubuntu **25.10** | one shared digest with `-questing` |
| `4.2.0-trixie`, `4.2.0` | current *stable*, Qt6 | `stable` points here |
| `final-3_28_13` | old pin, last pushed 2023-11-24 | 2.6 years stale, 2.7 GB |

Two traps worth stating plainly. **The unsuffixed release tags are not the LTS
build** — `ltr`, `3.44` and `3.44.12` all share digest `sha256:6fe4b8d4…` with
`3.44.12-questing`, i.e. Ubuntu 25.10, a 9-month interim release. Pin `-noble`
for anything long-lived. And **every qgis/qgis tag is amd64-only**; on Apple
Silicon the image runs only under `--platform linux/amd64` emulation, and the
X11 plumbing in `startDocker.sh` is Linux-only regardless. Use a Linux host.

### The container's Python is not the plugin's Python

The old base (QGIS 3.28 on jammy) shipped Python 3.10; noble ships 3.12 and
trixie/questing ship 3.13. All of these are outside the 3.9–3.11 the repo
targets, and none of it matters: `plugin_utils/installer.py` pins
`MIN_PY = MAX_PY = (3, 11)`, rejects the system interpreter, and falls back to
`plugin_utils/py311.py`, which downloads a relocatable CPython 3.11.15
(python-build-standalone, SHA-256 pinned). The plugin runs on 3.11 whatever the
image ships.

The cost is a dependency that is easy to misdiagnose: **the container needs
outbound network to github.com on the first plugin run.** Offline or behind a
proxy, that download fails, and the error does not obviously point back at the
base image.

Related: noble and trixie are PEP 668 "externally managed" environments. The
plugin installs into a venv and is unaffected, but a manual `pip install` in
the container now needs `--break-system-packages` or a venv of its own.

### Trialling QGIS 4

QGIS 4.2.0 is real and current (pushed 2026-07-05), and the plugin is closer to
ready than it looks: `metadata.txt` already declares `supportsQt6=True`, and all
GUI imports go through the `qgis.PyQt` shim. What is missing is *validation*,
not a port. Before `QGIS_TAG=4.2.0-trixie ./startDocker.sh` can tell you
anything:

1. **Raise `qgisMaximumVersion` in `metadata.txt`, or the plugin will not even
   load.** The field is currently omitted, and QGIS defaults it to
   `qgisMinimumVersion[0] + ".99"` — that indexes the *string*, so
   `qgisMinimumVersion=3.22` yields `3.99`. QGIS 4 lists the plugin as
   incompatible and refuses to enable it, with no hint that one metadata line
   is the cause. (Source: `python/pyplugin_installer/installer_data.py`.) It is
   left unraised deliberately: shipping an unvalidated compatibility claim to
   users is worse than an explicit trial step. `tests/test_qgis_container.py`
   fails if the default image is bumped to 4.x without this.
2. Expect the first real break at `uic.loadUiType()`
   (`winmol_analyzer_dialog.py:60`) loading a Qt5-authored `.ui` under Qt6.
   This is GUI-only; no test in this repo can catch it.
3. `Makefile:102` uses `pyrcc5`, which has no Qt6 equivalent (`pyrcc` was
   removed in Qt6), so `resources.py` can no longer be regenerated for a
   Qt6-only world. `winmol_analyzer.py` already treats the compiled resource as
   optional and falls back to `icon.png`, so the fix is to drop
   `resources.qrc` / `resources.py` and the `compile` target, not to port them.

### What is verified, and what is not

`tests/test_qgis_container.py` checks the contract between `Dockerfile`,
`startDocker.sh` and `metadata.txt` — that the tag is parameterised, pinned and
LTS-suffixed, that the two defaults agree, and that no version-pinned
`python3.X-venv` package creeps back in (that one line is what made the old
file fail to build on any newer base).

**No build or run of this image has been validated.** There is no Docker daemon
in the environment the bump was made in, and no X server. The ladder someone on
a Linux host should walk:

```bash
docker build --build-arg QGIS_TAG=3.44.12-noble -t winmol_analyser_docker:3.44.12-noble .
docker run --rm winmol_analyser_docker:3.44.12-noble python -c 'import sys; print(sys.version)'
./startDocker.sh    # then install + enable the plugin, watch it provision 3.11
```

Steps 1–2 would be cheap to add to CI; the GUI steps cannot be automated
without a headless X server.

## The legacy Dockerfiles

`Dockerfile-1`, `Dockerfile.blackwell`, `Dockerfile_carrot`, `Dockerfile_olive`
in the repo root are kept, untracked and unbuilt by CI. They predate the ONNX
migration: they install `tensorflow[and-cuda]`, and they `git clone`
`StefanReder/WINMOL_Analyzer`, so they cannot contain local work. (The tracked
root `Dockerfile` is *not* one of these — see "The QGIS GUI image" above.)

Note one name collision: `benchmark/bench_containers.py` documents
`--old-image winmol_analyser_docker`, meaning a locally built
`Dockerfile.blackwell`. The QGIS GUI image now always carries a version tag
(`winmol_analyser_docker:3.44.12-noble`), so the untagged name is unambiguous
again.

One historical note worth recording, because it is a real regression rather
than a stale file. Their `CMD ["winmol_batch.py", "general"]` **did work when
written**: model names were a hardcoded lowercase dict
(`spruce`/`beech`/`general`). On 2026-01-26 they moved to `config.json` keys,
which are capitalised, and every existing lowercase invocation started failing —
a silent break in the CLI's public interface. `winmol_batch.py` now matches
model names **case-insensitively**, so those callers work again.

## Model directory

`winmol_batch.py` takes `--model-dir`, defaulting to `$WINMOL_MODEL_DIR` and
then `./standalone/model_onnx`. The GPU image presets `WINMOL_MODEL_DIR=/models`.

Fetch models with:

```bash
gh release download models-onnx-v1 --repo cwinkelmann/WINMOL_Analyzer --dir ./models
```

Note three directories exist and are easy to confuse: `standalone/model_onnx/`
holds the `.onnx` files actually used, `standalone/model/` the legacy Keras
`.hdf5` originals, and older scripts referenced a `standalone/models/` that no
longer exists.
