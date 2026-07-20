# CUDA compute image for WINMOL Analyzer — TensorFlow-free.
#
# The analyzer runs ONNX through onnxruntime; TensorFlow is only needed to
# CONVERT models (requirements/convert.txt) and has no place here. That is what
# makes a slim base viable: the older GPU Dockerfiles in the repo root needed a
# multi-gigabyte CUDA/TF base purely to satisfy tensorflow[and-cuda].
#
# Base is plain python:3.11-slim and the CUDA userspace arrives as wheels via
# `onnxruntime-gpu[cuda,cudnn]` (see requirements/gpu.txt). The host driver plus
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
    # Where --model-dir looks unless overridden. Mount models here.
    WINMOL_MODEL_DIR=/models

# libexpat1 is needed by the geo stack's XML handling (same as the CI image).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, in their own layer, so code changes do not reinstall the
# ~2 GB of CUDA wheels.
COPY requirements/geo.txt requirements/gpu.txt /tmp/req/
RUN pip install --no-cache-dir -r /tmp/req/gpu.txt \
 && rm -rf /tmp/req

# The repo is COPYed from the build context, not git-cloned: the image then
# contains the code being tested rather than whatever upstream main happens to
# be. (The old Dockerfiles cloned StefanReder/WINMOL_Analyzer, so they could
# never contain local work.)
WORKDIR /opt/winmol
COPY . /opt/winmol

# Fail the build rather than ship an image that cannot import its own core.
RUN python -c "import utils.IO, utils.onnx_runtime, classes.Config; print('compute core OK')"

VOLUME ["/models", "/input", "/output"]

# NOTE: WINMOL_ONNX_FORCE_CPU is deliberately NOT set. docker/ci/Dockerfile sets
# it so CI matches the golden fixtures; setting it here would silently disable
# the GPU this image exists for.
ENTRYPOINT ["python", "-u", "winmol_batch.py"]
# Model names come from config.json and are capitalised: General, Beech,
# Spruce, Spruce_Deadwood.
CMD ["General", "--input", "/input", "--output", "/output"]
