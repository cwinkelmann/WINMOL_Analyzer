# CUDA compute image for WINMOL Analyzer — TensorFlow-free.
#
# The analyzer runs ONNX through onnxruntime; TensorFlow is only needed to
# CONVERT models (requirements/convert.txt) and has no place here. That is what
# makes a slim base viable: the older GPU Dockerfiles in the repo root needed a
# multi-gigabyte CUDA/TF base purely to satisfy tensorflow[and-cuda].
#
# Base is plain python:3.11-slim and the CUDA userspace arrives as wheels via
# `onnxruntime-gpu[cuda,cudnn]` (see requirements/cuda.txt). The host driver plus
# nvidia-container-toolkit supply the kernel side. This removes the class of
# failure the old files were fighting by hand -- Dockerfile_olive literally
# `pip uninstall`s nvidia-cudnn-cu11 to unpick a base-image/wheel mismatch.
#
# If that does not work on a given GPU, the documented fallback is an
# nvidia/cuda:<ver>-cudnn-runtime base with a matching pinned onnxruntime-gpu;
# see docker/gpu/README.md.
#
# Models are NOT baked in — mount them. See README.md for the run command.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="WINMOL Analyzer (CUDA)" \
      org.opencontainers.image.description="TensorFlow-free ONNX/CUDA compute image" \
      org.opencontainers.image.source="https://github.com/cwinkelmann/WINMOL_Analyzer"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # Deterministic stem counts: connect_stems iterates a set of string-hashed
    # objects, so an unpinned seed changes the result run to run.
    PYTHONHASHSEED=0 \
    # Defaults for the mount points. These are ENV, not CMD, on purpose:
    # CMD is REPLACED WHOLESALE the moment a user passes any argument, so
    # `docker run <image> Spruce_Deadwood` would silently drop --input/--output
    # baked into CMD and fall back to ./standalone/input. ENV survives that.
    WINMOL_MODEL_DIR=/models \
    WINMOL_INPUT_DIR=/input \
    WINMOL_OUTPUT_DIR=/output

# libexpat1 is needed by the geo stack's XML handling (same as the CI image).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, in their own layer, so code changes do not reinstall the
# ~2 GB of CUDA wheels.
# core.txt and cuda.txt too: gpu.txt is `-r core.txt` + `-r cuda.txt` + psutil,
# and pip resolves those includes relative to the file. Omitting either breaks
# the build with a missing-file error -- see requirements/README.md.
COPY requirements/core.txt requirements/cuda.txt requirements/gpu.txt /tmp/req/
RUN pip install --no-cache-dir -r /tmp/req/gpu.txt \
 && rm -rf /tmp/req

# Put the pip-installed CUDA libraries on the dynamic loader path.
#
# onnxruntime-gpu[cuda,cudnn] drops them under site-packages/nvidia/*/lib/,
# which ld.so does not search. Without this, onnxruntime still ADVERTISES
# CUDAExecutionProvider — get_available_providers() only reads a list — but
# creating it fails at runtime with:
#
#   Failed to load library .../libonnxruntime_providers_cuda.so with error:
#   libcublasLt.so.12: cannot open shared object file
#
# and onnxruntime falls back to CPU, which looks like "the GPU is slow".
# ldconfig is used rather than LD_LIBRARY_PATH so child processes inherit it —
# the multi-GPU path spawns workers, and an env var set only in the parent
# would not reliably reach them.
RUN python - <<'PY' > /etc/ld.so.conf.d/nvidia-pip.conf \
 && ldconfig \
 && echo "registered CUDA library paths:" && cat /etc/ld.so.conf.d/nvidia-pip.conf
import glob
import os
import site
roots = set(site.getsitepackages() + [site.getusersitepackages()])
seen = []
for root in roots:
    for d in sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib"))):
        if d not in seen:
            seen.append(d)
            print(d)
PY

# Fail the build if the CUDA provider's own dependencies cannot be resolved.
# This needs no GPU: loading the shared objects only exercises the linker, so
# it works on a CPU-only CI runner — which is precisely where the missing
# libcublasLt would otherwise go unnoticed.
RUN python -c "\
import ctypes; \
[ctypes.CDLL(n) for n in ('libcublasLt.so.12', 'libcublas.so.12', 'libcudnn.so.9')]; \
print('CUDA runtime libraries resolve')"

# The repo is COPYed from the build context, not git-cloned: the image then
# contains the code being tested rather than whatever upstream main happens to
# be. (The old Dockerfiles cloned StefanReder/WINMOL_Analyzer, so they could
# never contain local work.)
WORKDIR /opt/winmol
COPY . /opt/winmol

# Fail the build rather than ship an image that cannot import its own core.
# This includes winmol_batch (the ENTRYPOINT) and every pipeline stage, not
# just a sample: the first version checked only utils.IO/onnx_runtime/Config,
# and it was an eager `from matplotlib import pyplot` inside utils.IO that broke
# the build — matplotlib is intentionally absent here, since only save_image
# needs it and it is now imported lazily.
RUN python -c "\
import utils.IO, utils.onnx_runtime, utils.Prediction, utils.Skeletonization, \
       utils.Vectorization, utils.Quantification, classes.Config, winmol_batch; \
print('compute core + CLI OK')"

VOLUME ["/models", "/input", "/output"]

# NOTE: WINMOL_ONNX_FORCE_CPU is deliberately NOT set. docker/ci/Dockerfile sets
# it so CI matches the golden fixtures; setting it here would silently disable
# the GPU this image exists for.
ENTRYPOINT ["python", "-u", "winmol_batch.py"]
# Only the model name — the directories come from ENV above, so overriding this
# (e.g. `docker run <image> Spruce_Deadwood`) cannot lose them.
# Names: General, Beech, Spruce, Spruce_Deadwood (case-insensitive).
CMD ["General"]
