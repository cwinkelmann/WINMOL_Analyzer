# reimpl/gui-parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> The authoritative difference inventory (with rr6 file:line quotes and port
> notes per item) is the audit JSON:
> `/private/tmp/claude-501/-Users-christian-hnee-WINMOL-Analyzer/559aaf2b-9b10-48a2-aece-83e1f63a4621/tasks/wewu7jcyv.output`
> — every implementer MUST read its relevant section before coding.

**Goal:** Close the 38 audited behavioral gaps between our dialog/plugin shell
and rr6 (`origin/fix/rr6-win-dll-shadowing`, the reference for every port).
Branch `reimpl/gui-parity` off `reimpl/rc11-parity`. NO PR. GUI files
LOC-exempt; non-GUI additions small (registry variant kwarg, setup_state
texts, worker progress signal).

**Out of scope (documented cuts, still standing):** EnvProbeWorker,
onnx_diagnostics verdicts, py311 download UI beyond the deletion checkbox,
install-time CUDA session proof.

## Task 1 (HIGH): shell safety + the layer picker (~220 GUI LOC)

Audit refs: runexp#1/shell#1 (thread teardown), inputs#1/shell#2 (layer
picker), shell locale + qt6 guards.
- **Thread teardown**: port rr6's `closeEvent` + `_shutdown_threads` +
  `_reap_thread` + module-scope `_ALIVE_BG` parking (rr6 dialog:1285-1351,
  :131) adapted to OUR thread inventory: (thread,worker), (setup_thread,
  setup_worker), (ensure_thread, ensure_worker), (remove_thread,
  remove_worker), (maint_thread, maint_worker), (price_thread, price_worker).
  `winmol_analyzer.py::unload()` calls `dlg._shutdown_threads()`.
- **Layer picker**: rr6 dialog:1657-1700 near-verbatim — QgsMapLayerComboBox
  (RasterLayer filter, allow-empty, tooltip), 'Loaded layer:' row into
  gridLayout (3,0,1,3), `layerChanged` connected after ctor + `setLayer(None)`;
  `_on_uav_layer_changed` strips `|`-subdataset, sets uav_lineEdit +
  check_input_file. Restore the 'or choose a loaded raster layer' wording.
- **Locale guard** (shell): plugin loads when QSettings locale is unset (rr6
  winmol_analyzer.py's guarded locale read). **Qt6 guards**: file-based
  toolbar icon + guarded resources import per rr6.
Commit `feat(gui): layer picker, thread teardown on close/unload, shell guards`.

## Task 2: model UX cluster (~350 GUI + ~45 non-GUI)

Audit refs: inputs#2-#7, runexp#3, setup "Model download confirmation".
- `plugin_utils/model_registry.py`: `resolve(name, device=..., variant=None)`
  (explicit precision > 'default' device rule > 'auto' lossless-only) + a
  `lossless` bool on ModelEntry sourced from config.json (add the field to
  entries; int8/fp16 false, fp32 true) + `ModelEntry.hidden` forward-compat.
  Tests for the variant kwarg matrix (extend test_model_registry, ~30).
- Dialog: family-per-row combo + trailing family-less entries; variant
  selector (VARIANT_ITEMS/VARIANT_TOOLTIP, preset to device default, per-family
  graying + size labels, snap-to-0) per rr6 dialog:86-113, 630-773;
  model-info line (installed / 'will download N MB on Run'), degrade to
  'label | file — state' where rr6 metadata fields don't exist; download
  CONSENT QMessageBox on Run for missing models (label, size, URL; No aborts)
  + failure modal with manual-download guidance; combo fallback trio when
  config.json is unreadable; models_dir READ-FALLBACK: check
  `<plugin_dir>/models` before declaring not-installed (parity with rr6
  installs; our managed dir stays the write target).
Commit `feat(gui): variant selector, model info + consent, rr6 model UX parity`.

## Task 3: run gating + status narration (~280 GUI + ~60 setup_state)

Audit refs: runexp#2,#4-#9, setup gating rows, shell run-gate.
- Run gate: disabled Run with reason tooltip; blocked click routes to Setup
  (announcement); blocked-to-ready transition feedback; first-open guidance
  (_enter_first_run lite).
- Pre-run idle-GPU modal: lite `pre_run_decision` in setup_state (variant==cpu
  AND cached gpu probe present AND token not dismissed) + TXT_PRERUN_* texts;
  two-button modal (Install → _go_to_setup + _setup_install_gpu; Run-anyway →
  persist token via QSETTINGS_GPU_PROMPT_KEY — note ours currently READS a key
  nothing writes). Accel banner precondition/text fidelity.
- Status narration: RUN/IDLE status line never empty, 1 Hz elapsed ticker,
  worker-line echo, cancel-click feedback line.
Commit `feat(gui): run gate + pre-run GPU offer + status narration parity`.

## Task 4: Setup-tab UX parity (~380 GUI + ~40 workers)

Audit refs: setup section (all rows not covered above).
- Activity narration: determinate download % (ModelEnsureWorker +
  ModelDownloadWorker-style progress pyqtSignal(int) — add to
  tasks_threads workers; setup_progress_bar flips determinate; terminal
  value retention), status echo, idle/run status.
- Choose interpreter: offer to pip-install missing deps into a BYO
  interpreter (rr6's flow, using our installer.install_requirements);
  forget-interpreter offer reachable for BYO; rescan re-resolves env.
- Create env asks the GPU question up front on NVIDIA boxes; GPU runtime
  install path for BYO interpreters.
- Models tree: family grouping, per-row tooltips, recommended row follows the
  Detection-tab selection; Verify per-selected-row w/ honest unpinned refusal.
- Deletion scope: 'Downloaded Python 3.11 runtime' checkbox (backend
  remove_runtime already exists). Failure modals for env build/setup jobs.
Commit `feat(gui): Setup-tab parity — narration, interpreter flows, model tree`.

## Task 5: verify + zip + push
Full suite (tests from tests/, conda, PYTHONHASHSEED=0) green; flake8; ast +
objectName check on the dialog; static busy-guard/арity checks still pass;
rebuild zip 0.7.0-reimpl6; push. NO PR. Ledger + report with per-audit-item
closed/deferred table.

## Verification
Beyond suite/flake8: hand-trace per task in reports (teardown reap list,
layer-picker path flow, consent flow, GPU-offer token). Live-QGIS smoke stays
the user's round — flag that the layer picker + teardown especially want a
manual click-through.
