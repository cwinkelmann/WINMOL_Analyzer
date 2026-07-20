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

## Running it on a remote machine

The image is published to GHCR **only on a tag push or a manual run** — pull
request builds deliberately do not push, so a broken Dockerfile fails in review
without publishing anything. If `docker pull` reports *not found*, nothing has
been published yet. Publish one with:

```bash
# from anywhere with gh installed
gh workflow run "GPU image" --repo cwinkelmann/WINMOL_Analyzer --ref <branch>
# → ghcr.io/cwinkelmann/winmol-analyzer-gpu:sha-<short>
```

Then on the GPU machine:

```bash
# GHCR needs auth even for public images when pulling by digest/private pkg.
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <user> --password-stdin

docker pull ghcr.io/cwinkelmann/winmol-analyzer-gpu:latest

mkdir -p models input output
gh release download models-onnx-v1 --repo cwinkelmann/WINMOL_Analyzer --dir models
cp /path/to/ortho.tif input/

docker run --rm --gpus all \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  ghcr.io/cwinkelmann/winmol-analyzer-gpu:latest General
```

Prerequisites on the host: an NVIDIA driver and
[`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Without the toolkit, `--gpus all` fails; without it *and* without `--gpus`, the
container silently runs on CPU.

## Comparing against the old TensorFlow image

`benchmark/bench_containers.py` runs two images over the same orthomosaic N
times each and reports the median — the containerised counterpart of
`bench_orig_vs_changed.py`, which compares git refs instead.

```bash
# build the legacy image (clones upstream, TensorFlow, models baked in)
docker build -f Dockerfile.blackwell -t winmol-blackwell .

python benchmark/bench_containers.py \
  --old-image winmol-blackwell --old-profile legacy --old-model general \
  --new-image ghcr.io/cwinkelmann/winmol-analyzer-gpu:latest \
  --new-profile gpu --new-model General \
  --ortho /data/20220212_Barnekow_4.tiff \
  --models /data/models \
  --outdir benchmark/out/containers --repeats 3
```

The two images have **different interfaces**, which is why each side takes a
profile:

| | legacy | gpu |
|---|---|---|
| models | baked into the image | mounted at `/models` |
| input/output | `<workdir>/standalone/{input,output}` | `/input`, `/output` |
| entrypoint | `python -u` (script name passed) | `python -u winmol_batch.py` |
| model name | `general` (lowercase, upstream) | `General` |

Both sides run the **same weights** — `General.onnx` was converted from the
GenDS `.hdf5` at 0.0000 % binary-mask disagreement — so any difference in stem
count or volume is the pipeline, not the model.

**Read the result carefully.** Wall time here is end-to-end old-stack versus
new-stack: it includes TensorFlow → onnxruntime as well as the pipeline work,
which is what a user experiences after upgrading, but is *not* an isolated
measure of the pipeline changes. And check the `stems`/`deterministic` columns
before the timing — the old stack has no `PYTHONHASHSEED` guard, so its stem
count is expected to vary between runs while the new one should not.

For reference, the macOS (CPU/CoreML) result on the same orthomosaic was
**793 s → 67.8 s, 449 → 484 stems**, with the old side returning three
different stem counts from three identical runs. The GPU ratio will differ:
inference shrinks dramatically while the CPU-bound vector phase does not, so
the profile changes shape, not just scale.

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
