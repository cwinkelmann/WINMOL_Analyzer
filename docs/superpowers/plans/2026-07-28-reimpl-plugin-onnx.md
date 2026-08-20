# reimpl/plugin-onnx Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> Program spec: `docs/superpowers/specs/2026-07-28-reimpl-off-main-design.md`.

**Goal:** Feature 2 of the reimplementation program: a **TF-free end-to-end CLI
pipeline** plus a **concise, correct plugin shell-out** (venv build + sanitized
child environment incl. the Windows DLL-shadowing fix) on branch
`reimpl/plugin-onnx`, stacked on `reimpl/onnx-runtime`.

**Architecture:** Three layers. (1) De-TF the pipeline: one shared
`_resize_batch` numpy/skimage helper replaces `tf.image.resize`; all TF
*configuration* code is deleted, provider choice lives in
`utils/onnx_runtime` (env knobs `WINMOL_ONNX_FORCE_CPU`/`WINMOL_ONNX_PROVIDERS`).
(2) `plugin_utils/childenv.py`: pure-stdlib child-environment sanitizer (strip
QGIS interpreter/GDAL vars, Windows PATH DLL sanitize, PYTHONHASHSEED=0).
(3) Concise installer + workers: requirements-hash sentinel, streamed
venv/pip build, `resolve_environment` status dict; every child spawn wraps
`child_env()`/`safe_child_cwd()`. Reference (read, never copy wholesale):
`origin/fix/rr6-win-dll-shadowing` (call it RR6).

**Tech Stack:** Python 3.11, numpy + scikit-image (already deps), onnxruntime,
pytest, flake8 (max-line 80). Conda env `WINMOL_Analyzer`, `PYTHONHASHSEED=0`,
tests run from `tests/` dir with `PYTHONPATH=<repo>` (repo-root pytest broken on
main-era `__init__.py` — pre-existing).

## Global Constraints

- Anti-slop. Feature target ≈ **800 LOC gross** (code+tests, non-GUI;
  `winmol_analyzer_dialog.py` exempt). rr6 spends ~3.5k+ on the same scope.
  Trim aggressively; the reference's docstring walls become 1–3 line notes.
- Important tests only. No padding suites.
- Branch `reimpl/plugin-onnx` off `reimpl/onnx-runtime`. **No PR.** Push only at
  the end. `main` and old PRs #1–#15 untouched.
- Carry-over obligations from the proof feature's final review (MUST land here):
  (a) OOM catch tuple gains `MemoryError`; (b) `tf.image.resize` replaced;
  (c) module-level TF import dropped from `utils/Prediction.py`.
- OUT of scope (later features): model download/registry (config.json still
  points at .hdf5 — plugin E2E with real models awaits `reimpl/model-registry`),
  GPU runtime install/verify (`gpu-accelerator`), Setup-tab UX / progress bar
  (`plugin-gui`), py311 download, autotune cache placement.

---

## Task 0: Feature branch + plan commit

- [ ] `git checkout -b reimpl/plugin-onnx reimpl/onnx-runtime` (tip 0aa05a3)
- [ ] `git add docs/superpowers/plans/2026-07-28-reimpl-plugin-onnx.md && git commit -m "docs(reimpl): plugin-onnx feature plan"`

## Task 1: De-TF the CLI pipeline (net-negative diff)

**Files:** Modify `winmol_run.py`, `utils/Prediction.py`,
`utils/PredictWorkers.py`; Test `tests/test_prediction_tf_free.py` (new).

**Exact site map (from the exploration, verified file:line on this branch):**

- `utils/Prediction.py:14` module-level `import tensorflow as tf` → DELETE.
- `utils/Prediction.py:134-156` `_prepare_inference_batch`: replace both
  `tf.convert_to_tensor` + `tf.image.resize` calls (bicubic for imagery,
  nearest for masks) with one shared helper:
  `_resize_batch(batch_nhwc, size, order)` — skimage `resize` with
  `order=3, mode='edge', anti_aliasing=False, preserve_range=True` →
  `.astype(np.float32)` for imagery; `order=0` for masks; identity fast path
  when (h, w) already == target. Reference implementation: RR6
  `utils/Prediction.py:164-181` (19 lines). Numerics note: TF bicubic (Keys
  a=-0.5) vs skimage order=3 (cubic B-spline) is not bit-identical but
  measured-equivalent (RR6 IoU 0.996; the 0.5 binarization threshold absorbs
  it) — cubic-vs-bilinear is what matters (parity history c99d727/e4d3720).
- `utils/Prediction.py:295` OOM catch — current text
  `except (tf.errors.ResourceExhaustedError, RuntimeError) as exc:` →
  `except (RuntimeError, MemoryError) as exc:`; gate: `isinstance(exc,
  MemoryError)` is always-OOM, keep the phrase list for RuntimeError
  ('oom'/'out of memory'; drop 'resourceexhausted' or keep harmlessly).
  MemoryError matters: numpy's host-side allocation failure ("Unable to
  allocate …") must trigger batch-halving, not kill the run.
- `utils/Prediction.py:703-762` legacy `predict()` is DEAD on the CLI path
  (only `predict_stream_to_raster` is called from winmol_run.py:167) → DELETE
  the whole function (~-60 LOC). Check for stragglers with grep first;
  standalone/WINMOL_Analyzer.py calls `predict_with_resampling_per_tile`,
  which stays.
- `utils/PredictWorkers.py:47-74` duplicate `_prepare_inference_batch` →
  DELETE; `from utils.Prediction import _prepare_inference_batch`.
- `utils/PredictWorkers.py:230,233-238` and `:273,276-281` TF import +
  memory-growth loops → DELETE; KEEP `os.environ['CUDA_VISIBLE_DEVICES']`
  pins at :229/:272 (onnxruntime honors them).
- `winmol_run.py:24-52` `_import_tensorflow`, `_configure_tensorflow_runtime`,
  `_force_tensorflow_cpu_only` → DELETE all three.
- `winmol_run.py:162` cpu_stream branch → set
  `os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"` (before the model load at :165).
- `winmol_run.py:274-301` `check_DL_env`: keep the nvidia-smi block; replace
  the TF dump with ~6 lines: `import onnxruntime as ort` +
  print `ort.__version__`, `ort.get_available_providers()`, and
  `utils.onnx_runtime.selected_providers()`.
- `winmol_run.py:306-310` and `:359-365` "skip parent TF init" branches →
  DELETE (always run the env check; nothing to configure).

**Steps (TDD):**

- [ ] **Step 1:** Write failing tests in `tests/test_prediction_tf_free.py`:
  (1) `_resize_batch` imagery: (2,700,700,3)→(2,512,512,3) float32, and
  identity fast path returns the same object; (2) masks order=0 keeps values
  binary {0,1}; (3) OOM: a fake model whose `predict_on_batch` raises
  `MemoryError` makes `_predict_batch_adaptive` halve the batch (not raise);
  same for `RuntimeError("...out of memory...")`; a plain
  `RuntimeError("boom")` still re-raises; (4) TF-free import:
  `import utils.Prediction, utils.PredictWorkers` with
  `sys.modules["tensorflow"]=None`-style block active → no ImportError and
  `"tensorflow" not in sys.modules`.
- [ ] **Step 2:** Run → FAIL. Apply the site map. Run → PASS
  (`cd tests && PYTHONHASHSEED=0 PYTHONPATH=<repo> <conda>/bin/python -m
  pytest test_prediction_tf_free.py test_load_model_onnx.py -q`).
- [ ] **Step 3:** flake8 clean on the three modified files + the test; commit
  `feat(pipeline): TF-free prediction path — numpy/skimage resize, ORT-aware OOM retry`.

## Task 2: `plugin_utils/childenv.py` (the DLL fix, ~160 LOC)

**Files:** Create `plugin_utils/childenv.py`;
Test `tests/test_child_env.py` (new, ~120 LOC trimmed from RR6's 256).

**Interfaces (Task 3/4 rely on exact names):**
`child_env(extra=None, python_exe=None) -> dict`,
`sanitize_windows_path(path, parent_env, keep_roots=()) -> str`,
`safe_child_cwd(python_exe) -> str`.

Port from RR6 `plugin_utils/childenv.py:1-201,276-322` (CUT the nvidia
loader-path block :204-273 — GPU feature). Behavior to preserve exactly:
- STRIPPED interpreter vars (PYTHONHOME/PYTHONPATH/PYTHONSTARTUP/
  PYTHONEXECUTABLE/__PYVENV_LAUNCHER__/VIRTUAL_ENV + the rest of RR6's list)
  and GDAL/PROJ vars; marker vars captured BEFORE stripping.
- win32: PATH rewritten via `sanitize_windows_path` with
  `keep_roots=(venv root from python_exe,)`; uses `ntpath` explicitly so it
  is testable on macOS; entries under keep_roots always survive; drop rule =
  under a marker-derived root OR `_looks_like_qgis` segment heuristic.
- Pins `PYTHONHASHSEED=0` (deterministic stem counts) and
  `PYTHONNOUSERSITE=1`; `extra` applied LAST; never mutates `os.environ`.
- `safe_child_cwd`: venv root, else tempdir (cwd is on the Windows DLL
  search order).

- [ ] **Step 1:** Failing tests: strip list, hashseed+nousersite pins,
  extra-wins-over-strip, os.environ not mutated; sanitize table (OSGeo4W
  root dropped, `\apps\` heuristic, System32 kept, keep_roots venv under a
  QGIS profile path survives); POSIX PATH untouched (skipif win32, per the
  PR#16 lesson); safe_child_cwd venv-root/tempdir.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS; flake8; commit
  `feat(plugin): sanitized child environment — env-leak + Windows DLL-shadowing fix`.

## Task 3: Concise installer + requirements (~370 LOC vs RR6's 1464)

**Files:** Rewrite `plugin_utils/installer.py` (replaces the 277-LOC TF-era
file); Create `requirements/core.txt`, `requirements/cpu.txt`; Delete
`requirements/tensorflow.txt`, `requirements/tensorflow-win.txt` (grep for
referents first; rewrite/remove callers); Test `tests/test_plugin_installer.py`
(new, ~70 LOC).

**Requirements content (pins verbatim from RR6):**
- `core.txt` (names NO runtime): Pillow==10.1.0, Shapely==2.0.2,
  geopandas==0.14.0, numpy==1.26.4, rasterio>=1.4,<1.5,
  scikit-image==0.22.0, scipy==1.11.3, pyogrio>=0.9
- `cpu.txt`: `-r core.txt` + `onnxruntime>=1.17` + `psutil>=5.9`
- Leave `base.txt` in place only if something still references it (grep);
  otherwise delete.

**Installer keep-set (re-author compactly from RR6 installer.py):**
paths+selection (`managed_root` under the QGIS profile dir — NOT inside the
plugin dir, uninstall = rmtree(plugin_dir); `venv_location`,
`get_venv_python_path`, `plugin_requirements_path`→cpu.txt,
`_python_version` via `[exe, "-I", ...]` with `env=child_env()`,
`choose_base_python` 3.11-pinned, `_has_compute_deps` import-probe,
`configured_python_executable` QgsSettings read incl. the stale-managed-
pointer guard from RR6 :1426-1443); sentinel (`_file_hash`, `_marker_path`,
`marker_matches`, `is_ready`, `_write_marker` — current hash only, no legacy
digests); `_run_streamed` (single reader thread + queue, heartbeat, hard
timeout, error tail — compact re-author of RR6 :638-771);
`create_venv` (NO `--copies`; `env=child_env()` — the call that died on
Windows; Debian python3-venv hint) + `ensure_pip` (ensurepip ONLY);
`install_requirements` (pip `--upgrade --no-input --progress-bar off -r`);
`setup_environment` (is_ready → create → pip → marker; returns
`{'python': ...}`); `resolve_environment(plugin_dir, build=False)` (BYO
validation, ready/needs_setup, never raises).

**MUST NOT port (slop/deferred):** `pkg_resources` import +
`dependencies_installed` (dead; crashes on setuptools≥81 at plugin load),
`sudo apt`/`brew` calls, get-pip.py download/vendored blob, `--copies`,
QMessageBox prompt, GPU/py311/model/autotune/setup_state machinery,
"N of M" pip-line rewriter, legacy marker hashes.

- [ ] **Step 1:** Failing tests: module imports without QGIS/PyQt (and
  without pkg_resources); `plugin_requirements_path().name == "cpu.txt"`;
  per-platform venv python path; marker roundtrip (absent→not ready;
  write→ready; hash change→invalidated); requirements algebra (core.txt
  names no runtime; cpu.txt is core + exactly one runtime + psutil).
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS; flake8; commit
  `feat(plugin): concise ONNX installer — sentinel, streamed build, sanitized probes`.

## Task 4: Wiring — `__init__.py`, `tasks_threads.py`, minimal dialog glue

**Files:** Modify `__init__.py`, `tasks_threads.py`,
`winmol_analyzer.py`, `winmol_analyzer_dialog.py` (GUI file — LOC-exempt,
keep the delta minimal anyway); Test `tests/test_env_leak_regression.py`
(new, ~70 LOC).

- `__init__.py`: classFactory → `resolve_environment(plugin_dir,
  build=False)` only (no venv build, no pip at QGIS load); DELETE
  `_add_venv_site_packages` (injecting the compute venv into QGIS's
  interpreter is a version-conflict + reverse-DLL vector; the child-process
  design needs none of it). Hand the env dict to WINMOLAnalyzer.
- `tasks_threads.py`: re-author `Worker` compactly —
  `Popen(command, env=child_env(extra, python_exe=command[0]),
  cwd=safe_child_cwd(command[0]), stdout=PIPE, stderr=STDOUT, win32
  STARTUPINFO hide)`, line-streaming to the log signal, exit-code →
  succeeded/error, `cancel()` terminate→kill. Add minimal `EnvSetupWorker`
  (venv_location → setup_environment off the GUI thread; done/failed/log).
  Drop the hardcoded expected-line-count progress heuristic (:63-69) — emit
  raw lines; the progress bar is plugin-gui scope.
- Dialog/plugin glue (minimal): accept the env dict, `python_exe =
  env['python']`; on Run with status `needs_setup` → run EnvSetupWorker
  first, then the command. No other GUI changes.
- [ ] **Step 1:** Failing tests (live-child, POSIX): (a) anti-vacuity — a
  child spawned with QGIS-style PYTHONHOME/PYTHONPATH pollution FAILS;
  (b) the same child under `child_env()` succeeds; (c)
  `installer._python_version` works under pollution (kills the
  rebuild-loop bug).
- [ ] **Step 2:** Run → FAIL where appropriate → implement → PASS; flake8;
  commit `feat(plugin): wire sanitized shell-out — classFactory, Worker, env setup off-thread`.

## Task 5: TF-free compute contract (e2e) + feature verification

**Files:** Test `tests/test_plugin_compute_contract.py` (new, ~80 LOC,
self-contained — do NOT port RR6's fixture dependencies).

- [ ] **Step 1:** The contract test: build a tiny valid ONNX segmentation
  model on the fly (reuse the helper pattern from
  `tests/test_load_model_onnx.py`) + a small synthetic RGB GeoTIFF
  (rasterio, ~512×512, real CRS/transform); run
  `<conda python> -u winmol_run.py <model.onnx> <img.tif> <out.tif>
  <prefix> Stems` as a subprocess whose `sitecustomize` makes
  `import tensorflow` raise (PYTHONPATH injection dir); assert exit 0, the
  stem-map raster exists and opens, and the child's stdout confirms the
  ONNX provider line. (Random-weight model ⇒ don't assert stem counts.)
- [ ] **Step 2:** Full feature run: all new + Task-1 + onnx-runtime tests
  green; flake8 clean on every touched file.
- [ ] **Step 3:** LOC report: `git diff --stat reimpl/onnx-runtime..HEAD`
  (target ≈ 800 gross, honest number either way) and cumulative
  `git diff --stat main..HEAD` minus GUI files vs the 3–5k budget.
- [ ] **Step 4:** Push `reimpl/plugin-onnx` (branch only, NO PR). Commit
  message `feat(contract): TF-free end-to-end Stems run pinned by test`.

## Checkpoint after Task 5 (mandatory stop)

Report: feature LOC vs ~800, cumulative non-GUI LOC vs 3–5k, tests kept vs
RR6 equivalents, behavior gaps (esp.: plugin still lacks real-model download —
model-registry feature; GPU runtime — gpu-accelerator). User decides.

## Verification (feature-level)

1. All feature tests + the proof feature's tests green (tests/ cwd,
   PYTHONHASHSEED=0, conda env).
2. flake8 clean on all touched files.
3. `grep -rn "import tensorflow" winmol_run.py utils/ plugin_utils/` → no
   executable hits on the CLI/plugin path.
4. No PR; old stack untouched; branch deletable.
