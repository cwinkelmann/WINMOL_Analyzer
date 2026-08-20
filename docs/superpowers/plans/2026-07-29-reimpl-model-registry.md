# reimpl/model-registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Program spec: `docs/superpowers/specs/2026-07-28-reimpl-off-main-design.md`.

**Goal:** Feature 6 (~700 LOC gross): on-the-fly ONNX model registry with
device-aware defaults. Replaces rr6's 865-LOC registry + 150-LOC model_status +
31 KB config.json with a simpler schema and ~330 LOC. Branch
`reimpl/model-registry` off `reimpl/gpu-container-batch` (7df9c35). NO PR.
Reference: RR6 = `origin/fix/rr6-win-dll-shadowing`.

**Design decisions (from exploration):**
- Simplified schema-2: `{schema:2, gui_default, recommended[], families{label,
  default}, models{id:{label, family, precision, url, file, sha256, size_mb}}}`
  — NO per-family cpu/gpu/coreml keys; the device rule lives in code:
  `_device_variant`: cpu→int8, gpu→fp16, coreml→fp32, fall back to the family
  default when that precision doesn't exist. Explicit model id is NEVER
  rewritten.
- Models dir: `<profile>/winmol/models` (what the dialog's models_dir already
  resolves to); CLI/batch default `standalone/model/` overridable.
- URLs: GitHub release `models-v1` on `cwinkelmann/WINMOL_segmentor_pt`
  (22 .onnx + SHA256SUMS verified present). Every entry sha256-pinned —
  no-unverifiable-download property.
- Download core: urllib stream → `.part`, sha256 on the fly, atomic
  `os.replace`, 30 s socket timeout, `ensure_model` short-circuits on
  verified-existing, heals stale, `no_download` refuses. Fetcher injectable
  for tests. No retries/resume (rr6 didn't either).
- CUT (→ plugin-gui): stat-memo caches, installed_state, verify_entry,
  remove_model(s), model_status.py entirely.

## Task 1: `plugin_utils/model_registry.py` (~330) + tests (~150)

Registry core (`load_registry` incl. v1-flat fallback, `get/resolve/
default_entry/_device_variant`), `detect_device` (`WINMOL_DEVICE` override >
Apple-silicon probe > nvidia-smi probe > cpu), `local_path`, `verify_file`,
download core + `ensure_model`. Port from RR6 `plugin_utils/model_registry.py`
trimmed per the KEEP list; docstrings 1–3 lines.
Tests (`tests/test_model_registry.py`): device→variant mapping incl. cpu→int8
and explicit-id-never-rewritten; every-entry-has-sha256 (no-unverifiable-
download, reads the REAL config.json once Task 2 lands — write it against a
fixture dict now); download atomicity + checksum-mismatch via fake fetcher;
ensure_model short-circuit/heal/no_download; v1-flat fallback maps names.
TDD; flake8; commit `feat(models): concise device-aware ONNX model registry`.

## Task 2: config.json (trimmed schema-2) + winmol_batch wiring (~40)

- Replace config.json wholesale: classic families (Spruce, Beech,
  Spruce_Deadwood, General) × available precisions + UNet_PT trio — exactly
  the ids present on the `models-v1` release; sha256 from the release's
  SHA256SUMS (fetch via `gh release download -R cwinkelmann/WINMOL_segmentor_pt
  models-v1 -p SHA256SUMS*` — do NOT guess digests; sizes from `gh release
  view --json assets`). `gui_default` = Spruce_Deadwood int8 entry.
- winmol_batch.py: replace `load_model_paths/url_to_filename` resolution with
  registry `resolve(name-or-id, device=detect_device()) → ensure_model` into
  `--model-dir` (default `standalone/model/`); add `--list-models`; keep the
  CLI's positional model name compatible (family name still works).
- Extend the registry test that pins the real config.json (every entry sha256,
  ids resolvable, gui_default exists).
TDD; flake8; suite green; commit
`feat(models): schema-2 config pinned to models-v1; batch resolves via registry`.

## Task 3: minimal dialog wiring (GUI-exempt file, keep delta small)

- Combo populated from registry (labels, recommended first, device-matched
  default preselected); `.hdf5` filter → `.onnx`; on Run: `resolve` +
  `ensure_model` in a small worker thread (reuse the EnvSetupWorker pattern in
  tasks_threads.py — `ModelEnsureWorker`, ~30 LOC, done(path)/failed/log),
  gated by the same `_setup_running`-style guard; browse-custom-path still
  works (explicit path bypasses registry).
- Tests: none GUI-runnable — ast + flake8 + grep verification per Task-4
  precedent; ModelEnsureWorker logic testable off-Qt only if trivially
  separable (don't force it).
Commit `feat(gui): registry-driven model selection + on-demand download`.

## Task 4: verify + push
Full suite; flake8; LOC report vs ~700; push `reimpl/model-registry`. NO PR.
Ledger checkpoint (autonomous continuation).
