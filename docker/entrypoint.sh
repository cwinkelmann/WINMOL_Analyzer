#!/usr/bin/env bash
# Size the run to what this CONTAINER has, then hand over to winmol_batch.
#
# Every value below can be overridden by setting the env var explicitly;
# the autodetect only fills in what you did not specify. The resolved plan
# is printed, so a bad detection is visible in the logs rather than silent.
set -euo pipefail

MODEL_FAMILY="${1:-Spruce_Deadwood}"
INPUT="${WINMOL_INPUT:-/data/input}"
OUTPUT="${WINMOL_OUTPUT:-/data/output}"
MODEL_DIR="${WINMOL_MODEL_DIR:-/data/models}"
PROCESS_TYPE="${WINMOL_PROCESS_TYPE:-Nodes}"

# --- what we actually have -------------------------------------------------
read -r CORES MEM_GB GPUS <<<"$(python - <<'PY'
from plugin_utils import container
from classes.HardwareInfo import HardwareInfo
hw = HardwareInfo.detect()
print(container.cpu_count(),
      round(container.total_memory_bytes() / (1024 ** 3), 2),
      hw.gpu_count)
PY
)"

# --- how many orthos at once ----------------------------------------------
# A prediction process measured ~4.5 GB RSS; 6 GB leaves headroom for the
# vector phase that follows it in the same process.
PER_JOB_GB="${WINMOL_PER_JOB_GB:-6}"
if [ -z "${WINMOL_JOBS:-}" ]; then
  BY_MEM=$(python -c "print(max(1, int($MEM_GB // $PER_JOB_GB)))")
  BY_GPU=$([ "$GPUS" -gt 0 ] && echo "$GPUS" || echo 1)
  JOBS=$(( BY_MEM < BY_GPU ? BY_MEM : BY_GPU ))
else
  JOBS="$WINMOL_JOBS"
fi
[ "$JOBS" -lt 1 ] && JOBS=1

# --- GDAL block cache ------------------------------------------------------
# The single most effective knob measured. GDAL's 5% default collapses a
# large run (3050 -> 55 tiles/min on a 99k-tile ortho); 20% held 2675
# tiles/min flat. Starving it is far worse than over-feeding it, so this is
# deliberately generous -- and divided by JOBS because the cache is
# per-process and the jobs run concurrently.
if [ -z "${GDAL_CACHEMAX:-}" ]; then
  GDAL_CACHEMAX=$(python -c "
budget = $MEM_GB * 1024 * 0.20 / $JOBS
print(max(1024, int(budget)))")
fi
export GDAL_CACHEMAX

# --- per-job resource budget ----------------------------------------------
# Each job plans independently, so without dividing here two jobs would each
# size a full-machine vector pool and over-commit the container.
if [ -z "${WINMOL_CONFIG_OVERRIDES_JSON:-}" ]; then
  WINMOL_CONFIG_OVERRIDES_JSON=$(python -c "
import json
jobs = $JOBS
print(json.dumps({
    'max_cpu_workers': max(1, $CORES // jobs),
    'vector_ram_fraction': round(0.40 / jobs, 3),
}))")
fi
export WINMOL_CONFIG_OVERRIDES_JSON

cat <<EOF
================ WINMOL batch =================
 variant        ${WINMOL_VARIANT:-unknown}
 usable cores   ${CORES}
 usable memory  ${MEM_GB} GB
 GPUs visible   ${GPUS}
 parallel jobs  ${JOBS}   (${PER_JOB_GB} GB budgeted per job)
 GDAL_CACHEMAX  ${GDAL_CACHEMAX} MB per process
 overrides      ${WINMOL_CONFIG_OVERRIDES_JSON}
 model family   ${MODEL_FAMILY}
 process type   ${PROCESS_TYPE}
 input          ${INPUT}
 output         ${OUTPUT}
===============================================
EOF

# --- overviews -------------------------------------------------------------
# Without them GDAL reads every source pixel for each tile: ~2x the read
# time and ~4x the bytes, which is what makes throughput decay on large
# orthos. -ro writes a .ovr sidecar and leaves the input untouched.
if [ "${WINMOL_BUILD_OVERVIEWS:-0}" = "1" ]; then
  shopt -s nullglob nocaseglob
  for f in "$INPUT"/*.tif "$INPUT"/*.tiff; do
    if [ ! -f "${f}.ovr" ] && ! python -c "
import rasterio, sys
sys.exit(0 if rasterio.open('$f').overviews(1) else 1)" 2>/dev/null; then
      echo "building overviews for $(basename "$f") ..."
      gdaladdo -ro -r average "$f" 2 4 8 16 32 64 128 \
        || echo "  WARNING: gdaladdo failed for $f; continuing"
    fi
  done
  shopt -u nullglob nocaseglob
fi

mkdir -p "$OUTPUT" "$MODEL_DIR"

exec python -u winmol_batch.py "$MODEL_FAMILY" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model-dir "$MODEL_DIR" \
  --jobs "$JOBS" \
  --process-type "$PROCESS_TYPE" \
  "${@:2}"
