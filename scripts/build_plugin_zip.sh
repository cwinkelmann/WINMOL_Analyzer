#!/usr/bin/env bash
# Build the installable QGIS plugin zip.
#
# Single source of truth for what ships to end users. Used by `make package`
# and by .github/workflows/on-push-tags.yml, so the local build and the
# released artifact can never drift apart.
#
# Why not .gitattributes export-ignore? Because it is not scoped to packaging:
# `git archive` also backs GitHub's tarball API, which actions/checkout falls
# back to whenever git is absent (our CI container). Marking tests/ export-ignore
# therefore deleted tests/ from CI's own checkout. Pruning here keeps the
# repository intact for every consumer and confines the exclusion to the one
# place that wants it.
#
# Usage: scripts/build_plugin_zip.sh <git-ref> <output.zip> [version]
set -euo pipefail

REF="${1:-HEAD}"
OUT="${2:-WINMOL_Analyzer.zip}"
VERSION="${3:-}"
PLUGINNAME=WINMOL_Analyzer

# Development-only paths. Verified by an AST scan that no runtime module
# imports anything under them.
EXCLUDE=(
  .github .gitignore .gitattributes CLAUDE.md
  Makefile pb_tool.cfg plugin_upload.py pylintrc setup.cfg
  docker startDocker.sh scripts
  tests benchmark
  docs documentation standalone
  resources.qrc
)

# Resolve the output path now, while we are still in the caller's directory.
mkdir -p "$(dirname "$OUT")"
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

git archive --prefix="$PLUGINNAME/" "$REF" | tar -x -C "$BUILD"

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
for required in __init__.py metadata.txt winmol_run.py plugin_utils utils classes \
                requirements resources.py; do
  [ -e "$required" ] || { echo "FATAL: $required missing from package" >&2; exit 1; }
done

cd "$BUILD"
rm -f "$OUT_ABS"
zip -9rq "$OUT_ABS" "$PLUGINNAME"
echo "Created $OUT_ABS ($(du -h "$OUT_ABS" | cut -f1))"
