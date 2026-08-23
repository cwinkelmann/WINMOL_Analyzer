# WINMOL Analyzer — batch processing of a folder of orthomosaics.
#
# Two images from this one file, chosen by VARIANT:
#
#   docker build -f docker/Dockerfile --build-arg VARIANT=cpu -t winmol:cpu .
#   docker build -f docker/Dockerfile --build-arg VARIANT=gpu -t winmol:gpu .
#
# They are separate images on purpose: `onnxruntime` and `onnxruntime-gpu`
# both provide the `onnxruntime` module and MUST NOT be co-installed
# (requirements/gpu.txt spells this out) — pip will happily leave both in
# place and the result is undefined.
#
# The GPU image does NOT need an nvidia/cuda base. onnxruntime-gpu[cuda,cudnn]
# ships the CUDA and cuDNN wheels, and utils/onnx_runtime.preload_native_libs()
# dlopens them before the session is created. Only `--gpus all` and a host
# driver are needed.
ARG VARIANT=cpu

FROM python:3.11-slim AS base
ARG VARIANT

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # winmol_run.py re-execs itself when this is unset (its determinism
    # guard). Pinning it here spares every child that restart -- and on
    # Windows-style spawn semantics the re-exec breaks process supervision.
    PYTHONHASHSEED=0

# gdaladdo (for building overviews on inputs that lack them -- without
# them GDAL reads every source pixel per tile and throughput collapses)
# plus the runtime libs the geo wheels dlopen.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libexpat1 \
        procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first so the (slow, large) dependency layer caches
# independently of the source.
COPY requirements/ /app/requirements/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements/${VARIANT}.txt \
    # Fail the BUILD rather than the run if the wrong runtime landed: the
    # two must never coexist, and a silent CPU fallback in a GPU image is
    # very hard to notice later.
    && python - <<'PY'
import importlib.metadata as md
names = {d.metadata['Name'].lower() for d in md.distributions()
         if d.metadata['Name']}
cpu, gpu = 'onnxruntime' in names, 'onnxruntime-gpu' in names
assert not (cpu and gpu), "onnxruntime AND onnxruntime-gpu both installed"
assert cpu or gpu, "no onnxruntime installed"
import numpy
assert numpy.__version__.startswith('1.'), (
    f"numpy {numpy.__version__}: the geo stack (shapely/geopandas) is built "
    "against numpy 1.x and breaks on 2.x")
print(f"runtime ok: {'onnxruntime-gpu' if gpu else 'onnxruntime'}, "
      f"numpy {numpy.__version__}")
PY

COPY . /app

# Mount points. Models persist here so they are downloaded (and
# checksum-verified by plugin_utils/model_registry) only once.
#
# Group 0 + g=u, not winmol:winmol. A bind mount is governed by the HOST's
# ownership, so anyone whose host uid is not 1000 -- most users of a shared
# server -- has to run `--user $(id -u):$(id -g)` to write to /data/output.
# Measured 2026-08-23: under the old ownership that user could then write
# NOTHING, not /app, not /data, not $HOME, because every path belonged to
# uid 1000 alone. Owning them by group 0 and granting the group what the
# owner has makes an arbitrary --user work; 1777 on the mount points covers
# the case where one of them is left unmounted.
RUN mkdir -p /data/input /data/output /data/models \
    && useradd -m -u 1000 winmol \
    && chown -R winmol:0 /app /data \
    && chmod -R g=u /app /data \
    && chmod 1777 /data/input /data/output /data/models
USER winmol

# XDG_CACHE_HOME: $HOME belongs to uid 1000, so an arbitrary --user cannot
# cache the prediction autotune there. /tmp is writable by every uid.
ENV WINMOL_INPUT=/data/input \
    WINMOL_OUTPUT=/data/output \
    WINMOL_MODEL_DIR=/data/models \
    WINMOL_VARIANT=${VARIANT} \
    XDG_CACHE_HOME=/tmp/winmol-cache

ENTRYPOINT ["/app/docker/entrypoint.sh"]
# Model family from config.json: Spruce | Beech | Spruce_Deadwood | General
CMD ["Spruce_Deadwood"]
