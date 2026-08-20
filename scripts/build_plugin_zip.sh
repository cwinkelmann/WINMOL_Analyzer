#!/usr/bin/env bash
# Build the installable QGIS plugin zip.
#
# Single source of truth for what ships to end users ("Install from ZIP").
# The runtime is TensorFlow-free (ONNX via onnxruntime); this script packages
# exactly the CLI/plugin code paths winmol_run.py needs, stripping dev-only
# tooling (tests, docs, CI config, benchmark/convert scripts, docker files).
#
# Why not .gitattributes export-ignore? Because it is not scoped to packaging:
# `git archive` also backs GitHub's tarball API, which actions/checkout falls
# back to whenever git is absent. Marking tests/ export-ignore would delete
# tests/ from CI's own checkout too. Pruning here keeps the repository intact
# for every consumer and confines the exclusion to the one place that wants it.
#
# Usage: scripts/build_plugin_zip.sh <git-ref> <output.zip> [version]
#
# Resolves the repo from the script's own location rather than the caller's
# cwd, so it can be invoked from anywhere (Makefile, CI, or a clean tmp dir
# in the test suite) — only the output path is relative to the caller.
set -euo pipefail

REF="${1:-HEAD}"
OUT="${2:-WINMOL_Analyzer.zip}"
VERSION="${3:-}"
PLUGINNAME=WINMOL_Analyzer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

# Development-only paths, tracked but never imported by the runtime.
EXCLUDE=(
  .github .gitignore
  Makefile setup.cfg
  startDocker.sh scripts
  tests benchmark docker
  docs documentation standalone
  resources.qrc
)

# Resolve the output path now, while we are still in the caller's directory.
mkdir -p "$(dirname "$OUT")"
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

git -C "$REPO_ROOT" archive --prefix="$PLUGINNAME/" "$REF" | tar -x -C "$BUILD"

cd "$BUILD/$PLUGINNAME"
rm -rf "${EXCLUDE[@]}"
rm -f Dockerfile Dockerfile.* Dockerfile-* Dockerfile_*
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find . -name '.DS_Store' -delete 2>/dev/null || true

# metadata.txt carries the version QGIS displays; a tagged release overrides it.
if [ -n "$VERSION" ]; then
  if grep -q '^version=' metadata.txt; then
    sed -i.bak "s/^version=.*/version=$VERSION/" metadata.txt && rm -f metadata.txt.bak
  else
    echo "version=$VERSION" >> metadata.txt
  fi
fi

# Fail loudly rather than shipping a package missing its entry point.
for required in __init__.py metadata.txt config.json \
                winmol_run.py winmol_batch.py \
                winmol_analyzer.py winmol_analyzer_dialog.py \
                winmol_analyzer_dialog_base.ui tasks_threads.py \
                plugin_utils utils classes requirements \
                resources.py icon.png \
                requirements/cpu.txt requirements/gpu.txt \
                plugin_utils/model_registry.py plugin_utils/childenv.py \
                plugin_utils/gpu_probe.py; do
  [ -e "$required" ] || { echo "FATAL: $required missing from package" >&2; exit 1; }
done

cd "$BUILD"
rm -f "$OUT_ABS"
zip -9rq "$OUT_ABS" "$PLUGINNAME"
echo "Created $OUT_ABS ($(du -h "$OUT_ABS" | cut -f1))"
