# CI test image for WINMOL Analyzer.
#
# Contains the pinned test environment (requirements/ci.txt) plus the public
# Zenodo GenDS U-Net model baked in at /opt/winmol/model/ — so the 374 MB
# model never enters git and test jobs need no download step. Built and
# pushed to ghcr.io by .github/workflows/ci-image.yml; consumed by
# .github/workflows/tests.yml as the job container.
#
# Layer order: model first (changes ~never), then pip deps (change rarely) —
# a requirements bump does not re-download the model.

FROM python:3.11-slim

# --- model layer (sha256-pinned; matches tests/fixtures/manifest.json) -----
ARG MODEL_NAME=model_UNet_GenDS_512_2023-02-27_211141.hdf5
ARG MODEL_URL="https://zenodo.org/records/10491124/files/model_UNet_GenDS_512_2023-02-27_211141.hdf5?download=1"
ARG MODEL_SHA256=7314266214015f692f687f5f209e46ebbc9e94fce6d7c9ff0c3db49e4437e4fa
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/winmol/model \
 && curl -fL --retry 3 "$MODEL_URL" -o "/opt/winmol/model/$MODEL_NAME" \
 && echo "$MODEL_SHA256  /opt/winmol/model/$MODEL_NAME" | sha256sum -c -

# --- python environment -----------------------------------------------------
COPY requirements/base.txt requirements/ci.txt /tmp/req/
RUN pip install --no-cache-dir -r /tmp/req/ci.txt \
 && rm -rf /tmp/req

# Golden fixtures are only reproducible under a fixed hash seed
# (docs/CODE_REVIEW_2.md A-4); quiet TF logs in CI output.
ENV PYTHONHASHSEED=0 \
    TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /workspace
