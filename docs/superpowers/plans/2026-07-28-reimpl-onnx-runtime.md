# reimpl/onnx-runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Proof feature of the reimplementation program (spec:
`docs/superpowers/specs/2026-07-28-reimpl-off-main-design.md`): a TF-free,
layout-aware ONNX inference runtime on a clean branch off `main`, in ~450 LOC
where rr6 spends ~900 on the same scope.

**Architecture:** `utils/onnx_runtime.py` vendors an `OnnxSegmenter` whose
`predict_on_batch(NHWC)` is deliberately Keras-API-compatible, so it drops into
main's existing prediction flow. `IO.load_model_from_path` dispatches `.onnx`
to it and rejects legacy Keras models with a converter-naming error. The
behavioral reference is `origin/fix/rr6-win-dll-shadowing` (rr6-tip) — read it,
do not copy it wholesale.

**Tech Stack:** Python 3.11 (conda env `WINMOL_Analyzer`), onnxruntime, numpy,
pytest, flake8 (setup.cfg, max-line 80).

## Global Constraints

- Non-GUI LOC budget for this feature: **≤ ~450** total including tests
  (program budget 3–5k; see spec).
- Important tests only — the trimmed `tests/test_load_model_onnx.py` is the
  whole suite for this feature.
- All work on branch `reimpl/onnx-runtime` (off `main`, tip `a81276d`).
  **No PR.** Push the branch only at the end of Task 3.
- `main` and the existing PR stack #1–#15 are never touched.
- Run every pytest with `PYTHONHASHSEED=0` in the `WINMOL_Analyzer` conda env:
  `PYTHONHASHSEED=0 /Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest …`

---

## Task 0: Cleanup of the misfired split + proof branch

**Files:** none (git/GitHub state only)

- [ ] **Step 1:** Close the accidental PR #17 and mechanical split PRs #18–#23,
  each with the comment "Superseded by the clean reimplementation off main":

```bash
for n in 17 18 19 20 21 22 23; do
  gh pr close "$n" --comment "Superseded by the clean reimplementation off main"
done
```

- [ ] **Step 2:** Delete the split branches:

```bash
git push origin --delete \
  rr6-split/1-packaging-cleanup rr6-split/2-model-registry \
  rr6-split/3-setup-tab rr6-split/4-gui-polish rr6-split/5-accel-autotune \
  rr6-split/6-gpu-requirements rr6-split/7-review-findings
git branch -D rr6-split/1-packaging-cleanup rr6-split/2-model-registry \
  rr6-split/3-setup-tab rr6-split/4-gui-polish rr6-split/5-accel-autotune \
  rr6-split/6-gpu-requirements rr6-split/7-review-findings
```

- [ ] **Step 3:** Create the proof branch and verify its base:

```bash
git checkout -b reimpl/onnx-runtime main
git log -1 --format=%h   # expected: a81276d
```

- [ ] **Step 4:** Commit the spec and this plan (they live on the deletable
  branch by design):

```bash
git add docs/superpowers/specs/2026-07-28-reimpl-off-main-design.md \
        docs/superpowers/plans/2026-07-28-reimpl-onnx-runtime.md
git commit -m "docs(reimpl): spec + plan for the clean reimplementation off main"
```

## Task 1: Vendored ONNX segmenter (`utils/onnx_runtime.py`)

**Files:**
- Create: `utils/onnx_runtime.py` (~150–180 LOC; rr6's is 348)
- Test: `tests/test_load_model_onnx.py` (created here, extended in Task 2)
- Reference: `git show origin/fix/rr6-win-dll-shadowing:utils/onnx_runtime.py`

**Interfaces (later features rely on these exact names):**
- Produces: `OnnxSegmenter(model_path, providers=None)` with
  `.predict_on_batch(nhwc: np.ndarray) -> np.ndarray (NHWC)`, `.providers`,
  `.session`; `selected_providers() -> list[str]`.
- Provider precedence: env `WINMOL_ONNX_PROVIDERS` (comma list) >
  `WINMOL_ONNX_FORCE_CPU` (truthy) > CUDA > CoreML (Darwin/arm64 only) > CPU.
  CPU is always appended as fallback.

**KEEP from rr6:** provider precedence + env overrides; layout detection
(`_layout(shape, channels)`: NHWC/NCHW by which axis holds the channel count,
default NHWC); NCHW transpose on input and output; float32 coercion; runtime
OOM normalized to `MemoryError`.

**CUT (deferred to `reimpl/gpu-accelerator`):** `preload_native_libs`,
`verify_session_providers` / `_LAST_ACTIVE` / `runtime_report` / accelerator
labels, profiling hooks (`WINMOL_ONNX_PROFILE`).

- [ ] **Step 1: Write the failing real-inference tests.** Create
  `tests/test_load_model_onnx.py` with the layout tests, ported trimmed from
  rr6's file (its on-the-fly model builder): build a tiny NHWC model and a tiny
  NCHW model with the `onnx` package, run both through `OnnxSegmenter`, assert
  the output is NHWC of the expected shape in both cases, and assert
  `selected_providers()` returns `["CPUExecutionProvider"]` when
  `WINMOL_ONNX_FORCE_CPU=1` (monkeypatched env).

- [ ] **Step 2: Run to verify failure.**
  `PYTHONHASHSEED=0 … -m pytest tests/test_load_model_onnx.py -q`
  Expected: FAIL (`utils.onnx_runtime` does not exist).

- [ ] **Step 3: Implement `utils/onnx_runtime.py`** per the KEEP list only.

- [ ] **Step 4: Run to verify pass**, then `flake8 utils/onnx_runtime.py
  tests/test_load_model_onnx.py` → 0.

- [ ] **Step 5: Commit.**

```bash
git add utils/onnx_runtime.py tests/test_load_model_onnx.py
git commit -m "feat(onnx): vendored layout-aware OnnxSegmenter, TF-free"
```

## Task 2: Model-loading dispatch (`utils/IO.py`)

**Files:**
- Modify: `utils/IO.py` — `load_model_from_path` only (net delta vs main ≈ +40)
- Test: extend `tests/test_load_model_onnx.py`
- Reference: rr6's `utils/IO.py`

**Behavior:** `.onnx` (case-insensitive) → `OnnxSegmenter`; onnxruntime missing
→ `RuntimeError` naming the `onnxruntime` install; `.hdf5`/`.h5`/`.keras` →
`RuntimeError` pointing at `scripts/convert_models_to_onnx.py`. TensorFlow is
never imported on any path.

- [ ] **Step 1: Write the failing dispatch tests** (ported trimmed from rr6):
  fake-segmenter fixture (stub `utils.onnx_runtime` in `sys.modules`); assert
  `.onnx` dispatches, `.ONNX` dispatches (case-insensitive), missing
  onnxruntime raises the helpful error, `.hdf5` raises naming the converter,
  and `"tensorflow" not in sys.modules` after all of the above.
- [ ] **Step 2: Run to verify the new tests fail.**
- [ ] **Step 3: Implement the dispatch** in `IO.load_model_from_path`.
- [ ] **Step 4: Run to verify pass**; `flake8 utils/IO.py` → 0.
- [ ] **Step 5: Commit.**

```bash
git add utils/IO.py tests/test_load_model_onnx.py
git commit -m "feat(io): route .onnx to OnnxSegmenter; reject legacy Keras models"
```

## Task 3: Converter script + feature verification

**Files:**
- Create: `scripts/convert_models_to_onnx.py` (port rr6's, ~219 LOC, dev-only;
  the Task 2 rejection message references it; TF import stays inside `main()`)

- [ ] **Step 1:** Port the script;
  `python scripts/convert_models_to_onnx.py --help` exits 0 without TF
  installed; `flake8` → 0.
- [ ] **Step 2:** Full feature check:
  `PYTHONHASHSEED=0 … -m pytest tests/test_load_model_onnx.py -q` all green;
  report `git diff --stat main..reimpl/onnx-runtime` (expect ≤ ~450 LOC
  incl. tests and the two docs files).
- [ ] **Step 3:** Commit; push the branch only — **no PR**:

```bash
git add scripts/convert_models_to_onnx.py
git commit -m "feat(convert): dev-only HDF5->ONNX converter script"
git push -u origin reimpl/onnx-runtime
```

- [ ] **Step 4 (manual, optional): e2e smoke.** Download the int8
  Spruce_Deadwood model (models-onnx-v1 release), run
  `python winmol_run.py <model.onnx> <fixture.tif> <out.tif> <prefix> Stems`
  in the conda env. If main's prediction flow needs adaptation, STOP and
  report — that is `reimpl/plugin-onnx` scope, not silent scope creep here.

## Checkpoint after Task 3 (mandatory stop)

Report: feature LOC vs the ~450 target, tests kept vs rr6's ~900-LOC
equivalent scope, any behavior gaps. The user decides: continue to
`reimpl/plugin-onnx`, adjust the method, or delete `reimpl/*` and stop.