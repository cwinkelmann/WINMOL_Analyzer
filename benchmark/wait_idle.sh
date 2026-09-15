#!/usr/bin/env bash
# Run a command once this machine is genuinely idle.
#
#   bash wait_idle.sh -- bash benchmark/bench_reader_pool.sh --arm both
#
# "Idle" is three conditions, all of them, for NEEDED consecutive samples:
#   no GPU compute processes, 1-minute load below LOAD_MAX, no winmol-* container.
#
# One empty sample is not idle -- a neighbour between job stages looks empty for
# a moment, and starting then both wrecks our numbers and steals their GPUs when
# they resume.
#
# The load and container checks are NOT redundant with the GPU check, and this
# is the whole reason this script exists rather than a GPU-only one. The reader
# pool is CPU-decode-bound by construction (up to 128 reader threads), so a
# neighbour in a CPU vector phase -- GPUs already released -- corrupts the
# measurement while reading as perfectly idle. Observed 2026-09-15:
# winmol-trt:bench held 0 GPU processes and load 10 for minutes, and a GPU-only
# watcher had already counted two idle samples toward launching.
set -u
INTERVAL=${INTERVAL:-120}
NEEDED=${NEEDED:-5}
LOAD_MAX=${LOAD_MAX:-4.0}
MAX_WAIT=${MAX_WAIT:-$((36*3600))}
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"

[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: wait_idle.sh -- <command...>" >&2; exit 2; }

streak=0; waited=0
echo "watcher started $(date -Is): need ${NEEDED}x${INTERVAL}s idle (gpu+load+containers)"
while [ "$waited" -lt "$MAX_WAIT" ]; do
  gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  load=$(cut -d' ' -f1 /proc/loadavg)
  cont=$(docker ps --format '{{.Image}}' 2>/dev/null | grep -c '^winmol-' || true)
  ok=$(awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{print (l<m)?1:0}')
  if [ "$gpu" -eq 0 ] && [ "$ok" -eq 1 ] && [ "$cont" -eq 0 ]; then
    streak=$((streak+1))
    echo "$(date +%H:%M) idle ${streak}/${NEEDED} (load $load)"
    if [ "$streak" -ge "$NEEDED" ]; then
      echo "$(date -Is) IDLE -- launching: $*"
      exec "$@"
    fi
  else
    [ "$streak" -gt 0 ] && echo "$(date +%H:%M) busy again (gpu=$gpu load=$load winmol=$cont) -- reset"
    streak=0
  fi
  sleep "$INTERVAL"; waited=$((waited+INTERVAL))
done
echo "$(date -Is) gave up after ${MAX_WAIT}s -- never idle"
exit 1
