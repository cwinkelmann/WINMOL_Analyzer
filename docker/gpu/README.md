# CUDA compute image

TensorFlow-free GPU image for the analyzer: ONNX via `onnxruntime-gpu`, models
mounted at runtime.

It does **not** replace the machine-specific Dockerfiles in the repo root
(`Dockerfile.blackwell`, `Dockerfile_carrot`, `Dockerfile_olive`,
`Dockerfile-1`), which are kept as-is. Those predate the ONNX migration: they
install `tensorflow[and-cuda]`, which nothing in the analyzer uses any more, and
they clone upstream `StefanReder/WINMOL_Analyzer`, so they cannot contain local
work.

## Build

From the **repository root** (the build context must be the repo):

```bash
docker build -f docker/gpu/Dockerfile -t winmol-gpu .
```

Models, git history, docs and local data are excluded via
`docker/gpu/Dockerfile.dockerignore`.

## Run

```bash
docker run --rm --gpus all \
  -v /path/to/models:/models:ro \
  -v /path/to/orthos:/input:ro \
  -v /path/to/results:/output \
  winmol-gpu General --input /input --output /output
```

Model names come from `config.json` and are **capitalised** — `General`,
`Beech`, `Spruce`, `Spruce_Deadwood`. Lowercase is rejected by argparse.

`/models` must contain the files named in `config.json`, i.e. `General.onnx`
etc. Fetch them from the release:

```bash
gh release download models-onnx-v1 --repo cwinkelmann/WINMOL_Analyzer --dir ./models
```

`WINMOL_MODEL_DIR` is preset to `/models`; `--model-dir` overrides it.

## Verify the GPU is actually being used

**Do this before trusting any timing.** onnxruntime falls back to CPU
*silently* when the CUDA provider fails to load, which reads as "the GPU is
slow" rather than "the GPU never ran".

```bash
# 1. the container can see the GPU at all
docker run --rm --gpus all winmol-gpu --help >/dev/null && echo "entrypoint ok"
docker run --rm --gpus all --entrypoint nvidia-smi winmol-gpu

# 2. onnxruntime offers CUDA — this is the check that matters
docker run --rm --gpus all --entrypoint python winmol-gpu -c \
  "import onnxruntime as o; print(o.get_available_providers())"
```

The second command **must** list `CUDAExecutionProvider`. If it does not, see
*Troubleshooting*.

To force a provider explicitly rather than relying on auto-selection:

```bash
-e WINMOL_ONNX_PROVIDERS=CUDAExecutionProvider
```

Provider precedence is `WINMOL_ONNX_PROVIDERS` > `WINMOL_ONNX_FORCE_CPU` >
CUDA-if-available > CoreML > CPU (`utils/onnx_runtime.py`).
`WINMOL_ONNX_FORCE_CPU` is deliberately **not** set in this image — the CI image
sets it so results match the golden fixtures, and inheriting that here would
disable the GPU.

## Correctness before speed

A faster wrong answer is not a result. Reproduce the known macOS baseline
before drawing any performance conclusion:

| Input | Expected |
|---|---|
| Barnekow ortho, `General` | **484 stems, 439.228 m³** |

Stem counts must also be **identical across repeated runs** — that is what the
`PYTHONHASHSEED=0` baked into this image guarantees.

## Benchmarking

`benchmark/bench_orig_vs_changed.py` compares the original pipeline against the
current one; see `benchmark/README.md`. Note the **ORIGINAL side needs
TensorFlow-CUDA**, which this image deliberately does not provide. Either run
only the changed side, or build a separate TF image for the comparison.

## Troubleshooting

**`CUDAExecutionProvider` missing.** Almost always a CUDA-version mismatch.
`onnxruntime-gpu` 1.27 moved its extras to **CUDA 13** (`nvidia-*-cu13`);
earlier releases use `cu12`. If the host driver is older than CUDA 13 requires,
pin a cu12-era release in `requirements/gpu.txt`:

```
onnxruntime-gpu[cuda,cudnn]==1.22.*
```

**Blackwell (sm_120).** Needs a recent CUDA. If kernels fail to load on a
50-series / B-series card, that is the first thing to check — and it can only be
settled on the hardware.

**Fallback to a CUDA base image.** If the pip-provided CUDA userspace does not
work on a given host, switch the `FROM` line to
`nvidia/cuda:<ver>-cudnn-runtime-ubuntu22.04`, add `python3.11`, and pin an
`onnxruntime-gpu` built for that CUDA. Doing both — a CUDA base *and* the pip
CUDA extras — is what produced the `pip uninstall nvidia-cudnn-cu11` lines in
the older Dockerfiles; pick one source of CUDA, not both.
