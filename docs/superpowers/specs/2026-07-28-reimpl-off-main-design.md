# Reimplementation off `main` — design spec

Date: 2026-07-28
Status: draft, awaiting approval. Nothing merged; fully reversible.

## Problem

The fork's work reaches `main` through a 10-PR stack culminating in
`fix/rr6-win-dll-shadowing` (rr6-tip): **170 commits, +31,543/−2,589 across 166
files**. The history is churny (features merged incrementally, sub-branches
deleted), there is no reviewer, and the code carries AI slop: over-verbose
implementations, 51 test files / ~10.9k test LOC of mixed value, ~6k LOC of
docs. The top of the stack contains every line, so the intermediate stack
preserves nothing worth keeping.

Composition of `main..rr6-tip` additions:

| category | LOC |
|---|---|
| runtime/plugin code | +9,280 |
| GUI dialog (+.ui) | +3,659 |
| tests | +11,156 |
| docs | +5,979 |
| benchmark/scripts | +1,469 |

## Goal

Rebuild the divergence as a clean, stacked lineage of `reimpl/*` branches off
`main` — same behavior where it matters, a fraction of the code. rr6-tip is the
**behavioral reference**, never copied wholesale.

## Global constraints (binding)

1. **Non-GUI LOC budget: 3,000–5,000 total** changed code across all features.
   Only `winmol_analyzer_dialog.py` and `winmol_analyzer_dialog_base.ui` are
   exempt (GUI wiring is inherently verbose).
2. **Important tests only.** Behavior-pinning tests (dispatch, layout-awareness,
   rejection paths, golden stem parity, install/remove safety). No tautological
   or padding suites; rr6's test corpus is not ported wholesale.
3. **Reversible / low-trace.** All work on `reimpl/*` branches; **no PRs** until
   explicitly requested; `main` and the existing PR stack (#1–#15) untouched as
   fallback. Full abandon = delete the `reimpl/*` branches.
4. Concise, idiomatic code; flake8 (setup.cfg, max-line 80); no speculative
   abstraction.

## Decomposition — branch map

Stacked; each branch based on the previous. LOC = non-GUI changed-code target
(sum ≈ 4.0k).

| # | branch | base | contains | replaces (old PR/branch) | LOC |
|---|---|---|---|---|---|
| 1 | `reimpl/onnx-runtime` | `main` | OnnxSegmenter, IO dispatch, converter script | ONNX core of #1/#2 | ~450 |
| 2 | `reimpl/plugin-onnx` | 1 | installer/childenv/venv shell-out, win-DLL sanitize, winmol_run glue | rest of #2, merged #6, rr6 DLL fix | ~800 |
| 3 | `reimpl/deterministic-stems` | 2 | PYTHONHASHSEED pin + determinism | #4 | ~50 |
| 4 | `reimpl/perf-vectorization` | 3 | cubic resample, edge fix, vector-index hoist | #3, #9 | ~400 |
| 5 | `reimpl/gpu-container-batch` | 4 | CUDA container, `winmol_batch --jobs` per-GPU | #10 | ~400 |
| 6 | `reimpl/model-registry` | 5 | model_registry/model_status, config.json schema 2, device-aware defaults | rr6 model-zoo slice | ~700 |
| 7 | `reimpl/gpu-accelerator` | 6 | provider verify/report, GPU runtime install, autotune safety | rr6 rr3/rr4 slices | ~900 |
| 8 | `reimpl/plugin-gui` | 7 | Setup tab, dialog overhaul, run-progress | rr6 GUI slice | exempt |
| 9 | `reimpl/packaging-cleanup` | 8 | requirements→7 files, build zip, root tidy, CI | rr6 cleanup slice (+#5 docs if kept) | ~300 |

Out of scope: `feat/segmentation-edge-fill` (explicitly not for release; stays
its own branch), #12/#13 perf experiments (decided separately), the old #1–#15
stack (untouched until the reimpl lineage is accepted).

## Method

- Per feature: read rr6-tip's version to extract the intended behavior, then
  write the **minimal clean implementation**. Lean modules may be ported
  verbatim; everything else is re-authored.
- Key architectural lever: `OnnxSegmenter.predict_on_batch(NHWC)` is
  Keras-API-compatible, so the ONNX runtime drops into main's existing
  prediction flow without rewriting `Prediction.py` up front.
- One superpowers implementation plan per feature
  (`docs/superpowers/plans/2026-07-28-reimpl-<feature>.md`), executed via
  subagent-driven development. Mandatory checkpoint after each feature:
  cumulative LOC vs budget reported; user decides continue/adjust/stop.

## Verification

- Behavior, not tree equality: golden stem-count/volume parity on the fixture
  scene; model resolution picks the device-correct variant; provider selection
  prefers CUDA/CoreML with CPU fallback; plugin loads and the Setup tab
  creates/removes the env (manual QGIS smoke — no CI loads QGIS).
- Per feature: kept tests green (`PYTHONHASHSEED=0`, conda env
  `WINMOL_Analyzer`), flake8 clean, LOC within that feature's target.
- Program-level: `git diff --shortstat main..<top>` minus the two GUI files
  within 3–5k.

## Test environment

Conda env `WINMOL_Analyzer` (Python 3.11);
`PYTHONHASHSEED=0` always (conftest re-execs otherwise). Models download from
the `models-onnx-v1` release; the int8 Spruce_Deadwood build (31 MB) is the
default e2e model.