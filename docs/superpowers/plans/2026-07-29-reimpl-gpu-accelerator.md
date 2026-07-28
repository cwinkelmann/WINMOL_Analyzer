# reimpl/gpu-accelerator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Program spec: `docs/superpowers/specs/2026-07-28-reimpl-off-main-design.md`.

**Goal:** Feature 7 (~900 LOC gross): accelerator truth + GPU runtime install +
autotune safety. Branch `reimpl/gpu-accelerator` off `reimpl/model-registry`
(436f6f4). NO PR. Reference RR6 = `origin/fix/rr6-win-dll-shadowing`.

**Deferred:** rr6's 250-line install-time real-session CUDA proof (the Task-1
demotion warning delivers the same truth at first run, loud); autotune cache
(264 LOC — cost is re-tuning per run); CUDA container → packaging feature
(ort>=1.26 self-preloads its CUDA deps, LD-path fix obsolete).

## Task 1: provider truth (~150 + ~60 tests)

**Files:** `utils/onnx_runtime.py` (+ `verify_session_providers`,
`active_accelerator`, `_verify_providers` in OnnxSegmenter emitting a LOUD
demotion warning + `_LAST_ACTIVE`/`last_active_report`, `preload_native_libs`
— port rr6 onnx_runtime :42-77, :126-228, :279-310 trimmed);
`winmol_run.py` `check_DL_env` prints ACTIVE providers after model load (~15).
Tests: 3 demotion-warning cases (requested-but-not-available → build hint;
offered-but-didn't-bind → factual message; no demotion → silent), monkeypatched
sessions, no GPU needed.
Commit `feat(onnx): report what actually bound — provider demotion warnings`.

## Task 2: GPU runtime install (~300 + ~90 tests)

**Files:** Create `requirements/gpu.txt` (port RR6 verbatim — the
`onnxruntime-gpu>=1.26,<1.27` window is load-bearing BOTH ends: floor kills
silent CPU fallback, ceiling kills cu13 ImportError); Create
`plugin_utils/gpu_probe.py` (trim RR6's 187 → ~120: nvidia-smi detection,
wants_gpu_runtime verdict); Modify `plugin_utils/installer.py`
(`plugin_requirements_path(gpu=...)`, variant-aware sentinel marker,
`distribution_installed` + `uninstall_conflicting_runtime` — MANDATORY, both
dists ship the same module and pip never uninstalls the other); minimal dialog
offer: on Run with nvidia present + cpu-variant venv → one log-line offer +
QSettings dismissed key (mirror existing guard patterns; GUI-exempt, small).
Tests: gpu.txt/cpu.txt never-cross-pull (extend the requirements algebra),
requirements_choice matrix, 4 gpu_probe cases (monkeypatched nvidia-smi).
Commit `feat(gpu): opt-in onnxruntime-gpu venv — probe, conflict-safe install`.

## Task 3: autotune safety (~150 + ~60 tests)

Our branch ALREADY has the sweep (`_autotune_batch_size` Prediction.py:405-502,
patience/min_improve/stop_on_oom in Config :24-29) and `_predict_batch_adaptive`
halving (:328). Port ONLY rr6's safety set: `_memory_batch_ceiling` (+free-mem
helpers; rr6 Prediction :330-363) applied BEFORE timing (host-RAM never
swapped), `prediction_batch_override` (pins batch, skips sweep; Config knob +
env-overridable), absolute `min_improve_s` bar + post-OOM working ceiling
folded into the existing sweep loop. CUT: runaway/degradation guards, cache.
Tests: ceiling-before-timing, low-memory-never-sweeps, jitter-not-progress
(abs bar), override-pins — 4 focused cases.
Commit `feat(autotune): memory ceiling, absolute improvement bar, manual override`.

## Task 4: verify + push
Full suite; flake8; LOC vs ~900; push `reimpl/gpu-accelerator`; NO PR; ledger.
Real-GPU validation deferred to the user's e2e round (nothing here can bind
CUDA on this Mac).
