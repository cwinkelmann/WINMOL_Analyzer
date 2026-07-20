# Containers

Three images, three jobs. They are not interchangeable, and the differences
matter.

| Image | Built by | Purpose | Inference |
|---|---|---|---|
| `winmol-analyzer-ci` | `.github/workflows/ci-image.yml` | runs the test suite | **CPU, forced** |
| `winmol-analyzer-gpu` | `.github/workflows/gpu-image.yml` | production / batch on CUDA | GPU |
| root `Dockerfile*` | not built in CI | legacy, machine-specific | TensorFlow |

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

## The legacy Dockerfiles

`Dockerfile-1`, `Dockerfile.blackwell`, `Dockerfile_carrot`, `Dockerfile_olive`
in the repo root are kept, untracked and unbuilt by CI. They predate the ONNX
migration: they install `tensorflow[and-cuda]`, and they `git clone`
`StefanReder/WINMOL_Analyzer`, so they cannot contain local work.

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
