# reimpl/packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Single-task feature; task review doubles as final review.

**Goal:** Feature 8: make the reimpl lineage installable in QGIS for the user's
test round. Port rr6's self-contained `scripts/build_plugin_zip.sh` adapted to
THIS branch's file set; refresh `metadata.txt`. Branch `reimpl/packaging` off
`reimpl/gpu-accelerator`. NO PR. The old `make deploy` ships only flat
main-era files (Makefile:46-54) — it would silently omit every new package;
the ZIP route ("Install from ZIP" in QGIS) needs no Makefile surgery.
`plugin-gui` (Setup-tab UX) is deliberately deferred — the current dialog is
functional (combo + on-run env setup + model download + GPU offer).

## Task 1: build_plugin_zip.sh + metadata + smoke test (~150 gross)

**Files:** Create `scripts/build_plugin_zip.sh` (port
`git show origin/fix/rr6-win-dll-shadowing:scripts/build_plugin_zip.sh`,
adapt); Modify `metadata.txt` (version=0.7.0-reimpl1, description line
mentioning ONNX/TF-free); Test `tests/test_plugin_package.py` (~50).

- Script: `git archive HEAD | tar -x` into a temp dir under prefix
  `WINMOL_Analyzer/` (QGIS keys the installed plugin on the inner dir name);
  strip dev-only paths (adapt the EXCLUDE list to THIS branch: .github,
  .gitignore, CLAUDE-less…, Makefile, setup.cfg, docs, documentation, tests,
  standalone, benchmark, scripts, docker if present, Dockerfile*); required-
  file check FATAL if missing: `__init__.py metadata.txt config.json
  winmol_run.py winmol_batch.py winmol_analyzer.py winmol_analyzer_dialog.py
  winmol_analyzer_dialog_base.ui tasks_threads.py plugin_utils utils classes
  requirements` (+ resources.py + icon.png if present on this branch — check);
  zip -9rq; version override into metadata.txt via arg 3 (keep rr6's sed).
- Test: run the script into tmp (bash + git archive work in the test env),
  unzip -l, assert the required list present, the excluded dirs absent, inner
  dir name `WINMOL_Analyzer/`. Skip-guard if `zip`/`git` missing.
- TDD; flake8 (test file); `bash -n` the script; FULL suite green (149 + new).
- Commit `feat(package): self-contained plugin ZIP builder for the reimpl lineage`;
  build the zip once into the repo root as `WINMOL_Analyzer-reimpl.zip` proof
  (gitignored by the `WINMOL_Analyzer-*.zip` rule — verify) and report its size.
- Push `reimpl/packaging`. NO PR.
