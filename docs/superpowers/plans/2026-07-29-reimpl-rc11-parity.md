# reimpl/rc11-parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Feature 10: close the three deliberate cuts so the plugin behaves the
SAME as rc11 — the user's parity requirement. The env-related cut (py311
download) regressed the very problem class that motivated the refactor.
Branch `reimpl/rc11-parity` off `reimpl/plugin-gui` (current tip). NO PR.
Reference RR6 = `origin/fix/rr6-win-dll-shadowing`. Budget: ~550 non-GUI
(lands ≈4.35k of 5k).

**BINDING REQUIREMENT (user, 2026-07-29):** the plugin must work on Linux,
macOS, and Windows regardless of user skill or Python experience — install
QGIS with default settings, install the plugin ZIP, be ready to go. No
terminal, no brew/apt, no pre-installed Python beyond QGIS itself. Task 1 is
therefore load-bearing: the py311 download must cover win-x86_64,
mac-arm64/x86_64, linux-x86_64 and be the silent fallback whenever no
suitable 3.11 is on PATH.

## Task 1: py311 auto-download (the env-problems fix)

**Files:** Create `plugin_utils/py311.py` (port RR6's 223 trimmed ~170:
python-build-standalone download for the host platform/arch, extract under
managed_root/py311, return interpreter path; checksum if RR6 pins one);
Modify `plugin_utils/installer.py` (`managed_base_python`: prefer a 3.11 on
PATH, else `py311.ensure_python311(managed_root/py311)` — port RR6
:173-188; `choose_base_python` falls back to it instead of raising);
`setup_state.py`/dialog texts if they mention the no-3.11 dead end (grep).
Tests (~60): PATH-3.11-preferred (monkeypatch shutil.which); download path
invoked only when absent (fake fetcher — NO network); extraction layout;
failure → clear RuntimeError.
Commit `feat(env): auto-download a relocatable Python 3.11 when none is on PATH`.

## Task 2: autotune cache (per-run sweep cost parity)

**Files:** Create `plugin_utils/autotune_cache.py` (port RR6's 264 trimmed
~180: JSON cache keyed (hardware, model, EP, tile geometry) exactly as RR6
keys it — read its key derivation; load/store/clear; path from
`WINMOL_AUTOTUNE_CACHE` env else beside the venv); Modify
`utils/Prediction.py` (`_autotune_batch_size`: consult cache before sweeping,
persist after; respect `WINMOL_BATCH_AUTOTUNE=off` if RR6 honors it — check);
dialog: UNHIDE `autotune_clear_button`, wire to cache clear (+ pass
`WINMOL_AUTOTUNE_CACHE` env to the child like RR6's dialog:1487-1491).
Tests (~70): cache hit skips sweep (timing fn not called); key changes
(model/tile) miss; clear works; corrupt cache file ignored not fatal.
Commit `feat(autotune): tune once per hardware/model/EP/tile — persistent cache + clear button`.

## Task 3: GPU install-time verification (visible verdict parity)

**Files:** Modify `plugin_utils/installer.py` or `gpu_probe.py` (+~50): after
a gpu-variant install, run the CHILD venv python with a bounded probe:
`import onnxruntime; print(version + get_available_providers())` via
child_env; verdict text = CUDA present/absent with the same actionable
phrasing rc11 used (read RR6's verify messages for tone, deliver the short
form); EnvSetupWorker surfaces it in setup_detail_log + accel label refresh.
NOT porting the 250-line real-session proof — the runtime demotion warning
remains the deep check; this restores the *visible install-time verdict*.
Tests (~30): probe parses provider list; failure → verdict text not crash.
Commit `feat(gpu): install-time provider verdict from the child environment`.

## Task 4: verify + zip + push
Full suite; flake8; rebuild `WINMOL_Analyzer-reimpl.zip` (0.7.0-reimpl3) with
py311.py + autotune_cache.py pinned into the package test's REQUIRED list;
LOC report; push `reimpl/rc11-parity`. NO PR.
