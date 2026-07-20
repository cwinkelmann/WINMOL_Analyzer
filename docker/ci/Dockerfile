# CI test image for WINMOL Analyzer.
#
# Two stages:
#   1. convert  — downloads the public Zenodo GenDS Keras model and converts it
#                 to ONNX with tf2onnx (TensorFlow lives ONLY here).
#   2. runtime  — python:3.11-slim + onnxruntime + the pinned test env, with the
#                 converted General.onnx baked in at /opt/winmol/model_onnx/.
#                 NO TensorFlow in the final image; the model never enters git.
#
# Built/pushed to ghcr.io by .github/workflows/ci-image.yml; consumed by
# .github/workflows/tests.yml as the job container. Rebuilds only when its
# inputs (docker/ci/**, requirements/**, scripts/convert_models_to_onnx.py)
# change, so the conversion cost is paid rarely.

# ---------- stage 1: convert (throwaway; TensorFlow confined here) ----------
FROM python:3.11-slim AS convert
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:7314266214015f692f687f5f209e46ebbc9e94fce6d7c9ff0c3db49e4437e4fa \
    "https://zenodo.org/records/10491124/files/model_UNet_GenDS_512_2023-02-27_211141.hdf5?download=1" \
    /src/model/model_UNet_GenDS_512_2023-02-27_211141.hdf5
COPY requirements/convert.txt /tmp/convert.txt
RUN pip install --no-cache-dir -r /tmp/convert.txt
COPY scripts/ /src/scripts/
# Convert General only (the fixture model). Parity is verified locally; this
# build stage has no fixture, so convert without the parity assertion.
ENV TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3
RUN python /src/scripts/convert_models_to_onnx.py \
      --only General --no-parity \
      --model-dir /src/model --out-dir /opt/winmol/model_onnx

# ---------- stage 2: runtime (no TensorFlow) --------------------------------
FROM python:3.11-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=convert /opt/winmol/model_onnx/ /opt/winmol/model_onnx/
# geo.txt too: base.txt is `-r geo.txt` + the CPU runtime, so omitting it here
# would break the build with a missing-file error.
COPY requirements/geo.txt requirements/base.txt requirements/ci.txt /tmp/req/
RUN pip install --no-cache-dir -r /tmp/req/ci.txt \
 && rm -rf /tmp/req
# Golden fixtures are only reproducible under a fixed hash seed (review A-4);
# force CPU onnxruntime so CI matches the fixtures.
ENV PYTHONHASHSEED=0 \
    WINMOL_ONNX_FORCE_CPU=1
WORKDIR /workspace
