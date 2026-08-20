# reimpl/plugin-gui Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Program spec: `docs/superpowers/specs/2026-07-28-reimpl-off-main-design.md`.

**Goal:** Feature 9 (final): the rr-era GUI — Setup tab, Run-at-top/Export-at-
bottom, run progress bar, laptop-fit, banners — **behaving the same as the rr
plugin** but wired to the reimpl's simplified backend. Branch
`reimpl/plugin-gui` off `reimpl/packaging` (c85982d). NO PR.
Reference RR6 = `origin/fix/rr6-win-dll-shadowing`.

**Deliberate behavior differences vs rr6 (approved cuts, state in final report):**
no "Clear autotune cache" button (no cache on this lineage); GPU install button
uses the simple conflict-safe path with the provider-truth warning instead of
rr6's 250-line verify/verdict stack; no py311 auto-download (needs a 3.11 on
PATH or BYO); no EnvProbeWorker (cheap marker-read + gpu_probe instead —
dialog stays responsive by construction).

**Budget:** non-GUI helpers ~890 LOC (production total lands ≈3.6k of the 3–5k
budget). GUI files (`winmol_analyzer_dialog.py`, `*_dialog_base.ui`) exempt.
Tests: essentials only (~250 total).

## Task 1: run wiring helpers — run_progress + output_selection + config_overrides

**Files:** Create `plugin_utils/run_progress.py` (port RR6 verbatim, 193 —
its counter regexes match our pipeline's prints as-is, verified:
Prediction.py:913, PredictWorkers.py:483, VectorTilePipeline.py:389/463,
IO.py:1101/1137; the autotune regex is simply dead here);
`plugin_utils/output_selection.py` (44); `plugin_utils/config_overrides.py`
(54). Tests `tests/test_run_progress.py` (~70 trimmed from RR6's 334: each
counter prefix parses; bands monotonic 0→100; unknown lines ignored) +
~30 for output_selection/config_overrides essentials.
Commit `feat(gui-core): run progress parser + output/override helpers`.

## Task 2: setup backend — installer removal, setup_state-lite, model_status, workers

**Files:** Modify `plugin_utils/installer.py` (+`directory_size` ~18,
`invalidate_marker` ~20, `remove_environment` + `_rmtree`/`_chmod_retry`
~110 — port RR6 installer:440,525–614; refuses paths outside the managed
tree); Create `plugin_utils/setup_state.py` (~250 lite: `env_info`,
status/button-state texts, `human_bytes`, `deletion_plan` — DROP the
accelerator-verdict machinery); Create `plugin_utils/model_status.py`
(~150: scan() rows for the tree widget, flags the device default); Modify
`tasks_threads.py` (+`EnvRemoveWorker` ~55 incl. dry-run pricing,
`ModelMaintenanceWorker` ~60 for download-all/verify/delete). Tests (~150):
deletion_plan refuses outside-managed paths; remove_environment dry-run
prices then deletes only inside the tree (tmp dirs); model_status.scan flags
`default_entry(device)`'s row; invalidate_marker forces not-ready.
Commit `feat(gui-core): setup-tab backend — env removal, state texts, model rows`.

## Task 3: the dialog — .ui wholesale + core re-author

**Files:** Replace `winmol_analyzer_dialog_base.ui` with RR6's (1400,
data-only, no connections — portable); rewrite `winmol_analyzer_dialog.py`
core sections against OUR backend (GUI-exempt, ~1200 here): init/tab
structure/`_fit_to_available_screen`/scroll wiring; model combo (keep our
registry logic, adapt to new widget names); params/outputs via
output_selection + config_overrides; run/export/cancel with the progress bar
driven by `run_progress`; banners; layer loading. Keep our guards
(`_setup_running`, `_model_ensuring`) and workers. NO Setup-tab logic yet
(stub the tab's signals to no-ops so the dialog runs).
Verification: ast + flake8 (zero new) + grep every objectName referenced in
code exists in the .ui (scripted check, include in report); suite green.
Commit `feat(gui): rr-era dialog layout — tabs, progress bar, laptop-fit`.

## Task 4: Setup tab wiring

**Files:** `winmol_analyzer_dialog.py` (+~700, GUI-exempt): env group
(status/sizes via setup_state + directory_size on a worker-less cheap path —
sizes computed in EnvRemoveWorker dry-run style off-thread if slow), create/
repair (invalidate_marker + EnvSetupWorker), delete (EnvRemoveWorker with
itemised confirm), choose interpreter, open folder; model group
(models_treeWidget from model_status.scan, download/download-default/verify/
delete via ModelMaintenanceWorker, refresh); GPU install button (WINMOL_GPU
path + `uninstall_conflicting_runtime`, provider-truth warning is the
verification story); `setup_go_detect_button`. All long ops on workers, all
guarded against double-start (existing pattern). unload() must NEVER delete
the environment (the rr lesson — grep-assert in tests if cheap).
Verification: ast/flake8/objectName check/suite; manual QGIS smoke = user's
test round.
Commit `feat(gui): Setup tab — env + model management on the simplified backend`.

## Task 5: packaging closure + verify + push

Update `tests/test_plugin_package.py` required list (+run_progress,
setup_state, model_status, output_selection, config_overrides) and the
import-closure guard's file list if needed; rebuild the zip; FULL suite;
flake8; LOC report (non-GUI vs budget; GUI exempt tally); push
`reimpl/plugin-gui`. NO PR. Final report incl. the behavior-difference list.
