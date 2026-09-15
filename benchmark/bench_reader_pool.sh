#!/usr/bin/env bash
# Reader pool (#60) throughput benchmark: main vs perf/prediction-reader-pool.
#
# Runs the PREDICTION PHASE ONLY -- process_type=Stems calls run_stem_pipeline(),
# which is prediction and nothing else. The vector phase is identical in both
# arms and historically costs 38-65 min a run, so including it would triple the
# occupancy of a shared box to measure nothing.
#
# Arms:
#   main     origin/main
#   pool     perf/prediction-reader-pool
#   both     main then pool                      -- the throughput comparison
#   mainctl  main then main2 (identical code)    -- the determinism control
#
# The control exists because gate (ii) asks for bit-identical output, and this
# pipeline is not bit-identical to itself: Task 7 proved GPKG row order and
# stem_id are nondeterministic on main, and a 2026-09-15 R13 run found 1,554 of
# 25.2e9 raster pixels differing between main and pool. Before calling that a
# regression, run main against main and see what the floor actually is.
#
# Sequential by construction: the arms must never share the GPUs. The page cache
# is PRE-WARMED before EACH arm -- otherwise arm 2 inherits the cache arm 1 paid
# for and wins by default. The spec's 13,657 tiles/min baseline was measured
# page-cached, so warm is also the matching condition.
#
# Every path is overridable, so this runs on the T14 and CPU-only, which gate
# (ii) requires and a carrot-hardcoded script cannot do:
#
#   ROOT= SRC= ORTHO= MODEL= IMAGE= GPUS=none bash bench_reader_pool.sh --arm both
#
set -euo pipefail
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"

ROOT=${ROOT:-/raid/cwinkelmann/winmol}
SRC=${SRC:-$ROOT/bench/readerpool-src}
IMAGE=${IMAGE:-winmol-perf:gpu}
GPUS=${GPUS:-all}                 # "none" for a CPU-only arm
MODEL=${MODEL:-$ROOT/bench/models/model_UNet_SpecDS_Spruce_Deadwood_512_2024-12-19_194758_fp16.onnx}
ORTHO=${ORTHO:-$ROOT/orthos/WINDWURF_Tegel/Revier_13/result_Res1.3_COG.tif}
OUT=${OUT:-$ROOT/bench/readerpool-$(date +%Y%m%d-%H%M)}

ARM=both
while [ $# -gt 0 ]; do
  case "$1" in
    --arm)   ARM="$2";   shift 2 ;;
    --ortho) ORTHO="$2"; shift 2 ;;
    --out)   OUT="$2";   shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"; chmod 777 "$OUT" 2>/dev/null || true
echo "== reader-pool benchmark =="
printf 'ortho : %s\nimage : %s\ngpus  : %s\nout   : %s\n' "$ORTHO" "$IMAGE" "$GPUS" "$OUT"

preflight() {
  if [ "$GPUS" != none ]; then
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader \
      || { echo "FATAL: nvidia-smi failed" >&2; exit 1; }
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$busy" -eq 0 ] || { echo "FATAL: $busy GPU process(es) running -- refusing" >&2; exit 1; }
  fi
  [ -f "$MODEL" ] || { echo "FATAL: model missing: $MODEL" >&2; exit 1; }
  [ -f "$ORTHO" ] || { echo "FATAL: ortho missing: $ORTHO" >&2; exit 1; }
}

warm() {   # pull the ortho into page cache so no arm pays a cold-disk penalty
  echo "-- warming page cache ($(du -h "$1" | cut -f1)) --"
  dd if="$1" of=/dev/null bs=16M status=none
}

run_arm() {  # $1 = arm label, $2 = source checkout
  local arm=$1 src=$2 dir=$OUT/$1 rev
  [ -d "$src/.git" ] || { echo "FATAL: missing checkout $src" >&2; exit 1; }
  mkdir -p "$dir"; chmod 777 "$dir" 2>/dev/null || true
  rev=$(cd "$src" && git log --oneline -1)
  echo; echo "===== ARM $arm :: $rev ====="
  warm "$ORTHO"
  local gpuflag=(); [ "$GPUS" = none ] || gpuflag=(--gpus "$GPUS")
  local t0=$SECONDS
  docker run --rm "${gpuflag[@]}" --user 0:0 \
    -v "$src":/app -v "$ORTHO":/data/ortho.tif:ro -v "$MODEL":/data/model.onnx:ro -v "$dir":/out \
    -e PYTHONHASHSEED=0 \
    --entrypoint python "$IMAGE" -u winmol_run.py \
      /data/model.onnx /data/ortho.tif /out/stem_map.tif /out/trees Stems \
    > "$OUT/$arm.log" 2>&1
  echo "$arm wall: $((SECONDS - t0))s"
  echo "$rev" > "$OUT/$arm.rev"
}

preflight
case "$ARM" in
  main)    run_arm main "$SRC/main" ;;
  pool)    run_arm pool "$SRC/pool" ;;
  both)    run_arm main "$SRC/main";  run_arm pool  "$SRC/pool"  ;;
  mainctl) run_arm main "$SRC/main";  run_arm main2 "$SRC/main2" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

echo; echo "== done: $OUT =="
for f in "$OUT"/*.log; do
  echo "-- $(basename "$f" .log): $(grep -oE '[0-9.]+ tiles/min' "$f" | tail -1)"
done
