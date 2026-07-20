#!/usr/bin/env bash
# Legacy (TensorFlow/.hdf5, upstream) vs new (ONNX/CUDA, this fork), on one
# orthomosaic, N runs each. Fully detached: survives SSH logout.
#
#   results  -> /tmp/winmol-bench/SUMMARY.txt
#   progress -> /tmp/winmol-bench/progress.log
#   raw logs -> /tmp/winmol-bench/<side>_<n>.log
set -uo pipefail

BENCH=/tmp/winmol-bench
IN=/tmp/winmol-input
MODELS=/tmp/winmol-models
MODEL=Spruce_Deadwood
REPEATS=${REPEATS:-3}
NEW_IMG=winmol-gpu:test
OLD_IMG=winmol-blackwell:test
LEGACY_WD=/workspace/WINMOL-Analyzer

mkdir -p "$BENCH"
exec >> "$BENCH/progress.log" 2>&1
echo "=================================================================="
echo "started $(date -Is)  model=$MODEL  repeats=$REPEATS"

# --- wait for the legacy image, which may still be building ---------------
echo "waiting for $OLD_IMG ..."
for _ in $(seq 1 240); do            # up to 2h
    docker image inspect "$OLD_IMG" >/dev/null 2>&1 && break
    if grep -qE "^ERROR|error:" /tmp/legacy-build.log 2>/dev/null; then
        echo "LEGACY BUILD FAILED:"; tail -5 /tmp/legacy-build.log
        echo "continuing with the new image only"
        break
    fi
    sleep 30
done
docker image inspect "$OLD_IMG" >/dev/null 2>&1 && HAVE_OLD=1 || HAVE_OLD=0
echo "legacy image available: $HAVE_OLD"

run_side() {                          # $1=tag  $2=n
    local tag=$1 n=$2 out="$BENCH/${1}_${2}_out" t0 t1
    rm -rf "$out"; mkdir -p "$out"
    echo "--- $tag run $n  $(date -Is)"
    t0=$(date +%s)
    if [ "$tag" = new ]; then
        docker run --rm --gpus all \
            -v "$MODELS:/models:ro" -v "$IN:/input:ro" -v "$out:/output" \
            "$NEW_IMG" "$MODEL" > "$BENCH/${tag}_${n}.log" 2>&1
    else
        docker run --rm --gpus all \
            -v "$IN:$LEGACY_WD/standalone/input:ro" \
            -v "$out:$LEGACY_WD/standalone/output" \
            "$OLD_IMG" winmol_batch.py "$MODEL" \
            > "$BENCH/${tag}_${n}.log" 2>&1
    fi
    local rc=$?
    t1=$(date +%s)
    echo "$tag $n $rc $((t1-t0))" >> "$BENCH/timings.txt"
    echo "    exit=$rc wall=$((t1-t0))s"
}

: > "$BENCH/timings.txt"
for n in $(seq 1 "$REPEATS"); do
    [ "$HAVE_OLD" = 1 ] && run_side legacy "$n"
    run_side new "$n"
done

# --- summarise. The NEW image already has geopandas, so use it as the tool
# rather than depending on anything installed on the host.
#
# The summariser is a FILE mounted into the container, not a heredoc piped to
# `python -`. A heredoc needs `docker run -i` to attach stdin; without it docker
# silently feeds python EOF and you get a 0-byte summary while every run
# actually succeeded. Mounting the file removes the failure mode entirely
# instead of relying on remembering a flag.
echo "summarising $(date -Is)"
cp "$(dirname "$0")/summarise_containers.py" "$BENCH/summarise_containers.py"
docker run --rm -v "$BENCH:/bench" --entrypoint python "$NEW_IMG" \
    /bench/summarise_containers.py > "$BENCH/SUMMARY.txt" 2>&1

echo "DONE $(date -Is) -> $BENCH/SUMMARY.txt"
