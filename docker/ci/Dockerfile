# CI test image for WINMOL Analyzer.
#
# Contains the pinned test environment (requirements/ci.txt) plus the public
# Zenodo GenDS U-Net model baked in at /opt/winmol/model/ — so the 374 MB
# model never enters git and test jobs need no download step. Built and
# pushed to ghcr.io by .github/workflows/ci-image.yml; consumed by
# .github/workflows/tests.yml as the job container.

FROM python:3.11-slim

# --- model layer (sha256-pinned; matches tests/fixtures/manifest.json) -----
# ADD --checksum verifies at build time and caches by digest, keeping this
# layer independent of the apt/pip layers below.
ADD --checksum=sha256:7314266214015f692f687f5f209e46ebbc9e94fce6d7c9ff0c3db49e4437e4fa \
    "https://zenodo.org/records/10491124/files/model_UNet_GenDS_512_2023-02-27_211141.hdf5?download=1" \
    /opt/winmol/model/model_UNet_GenDS_512_2023-02-27_211141.hdf5

# --- system libraries -------------------------------------------------------
# rasterio/fiona/pyogrio manylinux wheels vendor GDAL but dynamically load
# libexpat from the system, which python:*-slim strips.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

# --- python environment -----------------------------------------------------
COPY requirements/base.txt requirements/ci.txt /tmp/req/
RUN pip install --no-cache-dir -r /tmp/req/ci.txt \
 && rm -rf /tmp/req

# Golden fixtures are only reproducible under a fixed hash seed
# (docs/CODE_REVIEW_2.md A-4); quiet TF logs in CI output.
ENV PYTHONHASHSEED=0 \
    TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /workspace
