# Workers-per-GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run two prediction worker processes on a single GPU so the second process's CPU-side work overlaps the first's kernels, lifting the T14 from 4,700 to ≥ 5,800 tiles/min.

**Architecture:** The planner gains a `workers_per_gpu` field (2 on single-GPU boxes with ≥ 8 GB VRAM, else 1). `run_prediction_phase` expands `gpu_ids` to `[0, 0]`; the existing coordinator shards the jobs across both workers and each worker keeps its own reader pool. No worker-side logic changes: the F4 `CUDA_VISIBLE_DEVICES` indexing already handles duplicate ids, and autotune's atomic cache write already handles two writers.

**Tech Stack:** Python 3.11, dataclasses, multiprocessing (spawn), pytest. Tests run with the project conda python: `pytest tests/ --junitxml=/tmp/junit.xml -q` (stdout is swallowed — read the junit file or `-rA`).

**Spec:** `docs/superpowers/specs/2026-09-15-prediction-reader-pool-design.md`, section "Amendment 2026-09-16 — two workers per GPU on single-GPU machines".

**Branch:** `perf/workers-per-gpu` (off `perf/reader-pool-trt` @ 075d43e). Never commit to `main`, `fix-first-run` or `feat/tensorrt-provider`.

## Global Constraints

- Output must stay bit-identical to `main` on the same input (Task 7 of the reader-pool plan proved assembly is order-independent; the shard boundary moving must not change a pixel or a GPKG row).
- `SINGLE_GPU`: `workers_per_gpu = 2` iff `gpu_memory_gb >= 8`, else 1. `MULTI_GPU`: 1. `CPU_ONLY`: 1. Env override `WINMOL_WORKERS_PER_GPU` (positive int) wins everywhere; non-int is ignored.
- With 2 workers on a single GPU: `reader_chunk = 4`, `prediction_batch_size <= 2`.
- `_reader_threads` divides by the TOTAL worker count; the single-GPU per-worker cap `READERS_MAX_SINGLE_GPU = 3` is unchanged.
- No new module. No refactors beyond what a task lists. flake8 clean on touched files. Commit trailer:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MbEjPqfjLm34ZYjrmAN6AZ
  ```

---

### Task 1: Planner — `workers_per_gpu` field, sizing, reader threads by total workers

**Files:**
- Modify: `classes/ExecutionPlan.py` (dataclass ~line 58-75; `_reader_threads` ~line 313-326; `build_execution_plan` single-GPU branch ~line 411-450, multi ~451-480, cpu ~363-390; the `return ExecutionPlan(...)` ~line 495)
- Test: `tests/test_execution_plan_readers.py`

**Interfaces:**
- Consumes: `_reader_threads(hw_cpu, n_gpu, cpu_only)`, `_cfg`, `_gpu_memory_gb`, scenario constants `CPU_ONLY/SINGLE_GPU/MULTI_GPU`.
- Produces: `ExecutionPlan.workers_per_gpu: int`; `ENV_WORKERS_PER_GPU = 'WINMOL_WORKERS_PER_GPU'`; `_workers_per_gpu(scen: str, gpu_mem_gb: float) -> int`. `_reader_threads` keeps its signature — callers pass the total worker count as `n_gpu`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_execution_plan_readers.py` (it already has `_plan(cpu_count, gpu_count, gpu_memory_gb, **cfg)` and `_Hardware`; import `_workers_per_gpu` alongside `_reader_threads`):

```python
from classes.ExecutionPlan import _workers_per_gpu  # add to the existing import


@pytest.mark.parametrize("scen,gpu_mem_gb,expected", [
    ('gpu', 16.0, 2),        # T14: 4080 SUPER
    ('gpu', 8.0, 2),         # threshold inclusive
    ('gpu', 6.0, 1),         # too little VRAM for two contexts
    ('multi_gpu_dgx', 80.0, 1),  # carrot: unchanged until measured
    ('cpu_only', 0.0, 1),
])
def test_workers_per_gpu_rule(scen, gpu_mem_gb, expected):
    assert _workers_per_gpu(scen, gpu_mem_gb) == expected


def test_workers_per_gpu_env_override_wins(monkeypatch):
    monkeypatch.setenv('WINMOL_WORKERS_PER_GPU', '3')
    assert _workers_per_gpu('gpu', 16.0) == 3
    assert _workers_per_gpu('multi_gpu_dgx', 80.0) == 3
    monkeypatch.setenv('WINMOL_WORKERS_PER_GPU', 'two')
    assert _workers_per_gpu('gpu', 16.0) == 2      # ignored, rule applies


def test_single_gpu_two_workers_shrink_chunk_batch_and_readers():
    plan = _plan(12, 1, [16.0])
    assert plan.workers_per_gpu == 2
    assert plan.reader_chunk == 4
    assert plan.prediction_batch_size <= 2
    # 11 cores / 2 workers = 5 -> per-worker cap 3
    assert plan.producer_workers == 3


def test_single_gpu_small_card_keeps_one_worker():
    plan = _plan(12, 1, [6.0])
    assert plan.workers_per_gpu == 1
    assert plan.reader_chunk == 8


def test_single_gpu_few_cores_split_readers_across_workers():
    # 5 cores: (5-1)//2 = 2 readers per worker, not 3
    plan = _plan(5, 1, [16.0])
    assert plan.workers_per_gpu == 2
    assert plan.producer_workers == 2


def test_multi_gpu_and_cpu_only_keep_one_worker_per_device():
    assert _plan(32, 8, [80.0]).workers_per_gpu == 1
    assert _plan(12, 0).workers_per_gpu == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_execution_plan_readers.py -q -rA`
Expected: ImportError on `_workers_per_gpu` (collection error) — that counts as RED.

- [ ] **Step 3: Implement**

In `classes/ExecutionPlan.py`:

(a) Dataclass — add after `reader_chunk: int`:
```python
    #: Prediction worker processes per GPU. 2 on single-GPU machines with
    #: enough VRAM: one worker's CPU-side work (upload, crop, binarise,
    #: result pickling) leaves the card idle ~40% of the time under one
    #: GIL; a second process overlaps it. Measured T14 (4080 SUPER):
    #: 1 proc 5,731 -> 2 procs 6,756 tiles/min, 3+ add nothing. See the
    #: spec amendment of 2026-09-16.
    workers_per_gpu: int = 1
```
(Give it a default so the field can sit after `reader_chunk` and before `capped`.)

(b) Next to `ENV_READERS`:
```python
ENV_WORKERS_PER_GPU = 'WINMOL_WORKERS_PER_GPU'
#: Two CUDA contexts + onnxruntime arenas at batch <= 4 measured <= 3.4 GB
#: each on the T14; below this VRAM a second worker risks the OOM back-off
#: fighting itself.
WORKERS_PER_GPU_MIN_VRAM_GB = 8.0


def _workers_per_gpu(scen: str, gpu_mem_gb: float) -> int:
    override = os.environ.get(ENV_WORKERS_PER_GPU, '').strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if scen == SINGLE_GPU and gpu_mem_gb >= WORKERS_PER_GPU_MIN_VRAM_GB:
        return 2
    # Multi-GPU is left at 1 until measured on carrot (its H100 workers
    # are consumer-bound too, but the reader/RSS budget differs); the env
    # override exists for that measurement.
    return 1
```

(c) `build_execution_plan`: in the `SINGLE_GPU` branch, after `producer_queue_batches = max(2, min(4, producer_queue_batches))`, replace
```python
        reader_chunk = 8
        producer_workers = _reader_threads(hw_cpu, 1, False)
```
with
```python
        workers_per_gpu = _workers_per_gpu(scen, gpu_mem_gb)
        if workers_per_gpu > 1:
            # Two workers share the card: batch 1-2 is fastest there (see
            # spec amendment), so cap the batch and the autotune sample.
            prediction_batch_size = min(prediction_batch_size, 2)
            reader_chunk = 4
        else:
            reader_chunk = 8
        producer_workers = _reader_threads(
            hw_cpu, gpu_workers * workers_per_gpu, False)
```
In the `CPU_ONLY` branch add `workers_per_gpu = 1` next to `reader_chunk = 4`. In the multi-GPU branch add `workers_per_gpu = _workers_per_gpu(scen, gpu_mem_gb)` before `producer_workers = _reader_threads(hw_cpu, gpu_workers * workers_per_gpu, False)` (replacing the current `_reader_threads(hw_cpu, gpu_workers, False)`).

(d) Pass `workers_per_gpu=workers_per_gpu` in the `return ExecutionPlan(...)`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_execution_plan_readers.py -q -rA`
Expected: all PASS, including the pre-existing `test_reader_threads_rule` table (unchanged semantics for `n_gpu`).

- [ ] **Step 5: flake8 + commit**

```bash
flake8 classes/ExecutionPlan.py tests/test_execution_plan_readers.py
git add classes/ExecutionPlan.py tests/test_execution_plan_readers.py
git commit -m "feat(plan): workers_per_gpu — two prediction workers on single-GPU machines (#60)"
```

---

### Task 2: Dispatch — expand `gpu_ids`, print the field, prove the coordinator shards across duplicate ids

**Files:**
- Modify: `winmol_run.py` (`build_plan` print block ~line 138-152; config copy block ~line 172-179; `run_prediction_phase` ~line 181-197)
- Test: `tests/test_prediction_drain.py` (has a coordinator test using `gpu_ids=[0, 1]` at ~line 167 — read it and its fake-worker fixture first), `tests/test_reader_pool.py` (has `_minimal_worker_run(monkeypatch, tmp_path, gpu_id, ...)` ~line 388 and the two CUDA_VISIBLE_DEVICES tests ~line 428-457)

**Interfaces:**
- Consumes: `ExecutionPlan.workers_per_gpu` (Task 1); `run_multi_gpu_prediction(model_path, input_raster, stem_path, tile_jobs, gpu_ids, config)`.
- Produces: `self.config.prediction_workers_per_gpu`; `gpu_ids = [g for g in range(gpu_workers) for _ in range(workers_per_gpu)]`.

- [ ] **Step 1: Write the failing tests**

(a) In `tests/test_prediction_drain.py`, next to the existing coordinator test that passes `gpu_ids=[0, 1]`, add a sibling using the same fixture pattern (copy its setup; only the assertions differ):

```python
def test_duplicate_gpu_ids_spawn_two_workers_with_disjoint_shards(...):
    """gpu_ids=[0, 0] must start TWO workers on device 0, each with a
    non-overlapping half of the jobs -- the shard split keys on list
    position, not on the id value."""
    # ... same fake-worker/context setup as the [0, 1] test ...
    run_multi_gpu_prediction(..., tile_jobs=jobs, gpu_ids=[0, 0], config=Config())
    assert [call.gpu_id for call in spawned] == [0, 0]
    shards = [call.jobs for call in spawned]
    assert len(shards[0]) + len(shards[1]) == len(jobs)
    assert not (set(map(id, shards[0])) & set(map(id, shards[1])))
```
Use whatever the existing test records (`spawned` / args tuple) — adapt names to that fixture; the three assertions are the contract.

(b) In `tests/test_reader_pool.py`, add after `test_worker_sets_cuda_visible_devices_when_unset`:
```python
def test_two_workers_with_the_same_gpu_id_both_pin_device_zero(
        monkeypatch, tmp_path):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    seen = []
    for _ in range(2):
        def _probe(*a, **k):
            seen.append(os.environ.get('CUDA_VISIBLE_DEVICES'))
            raise RuntimeError('stop')
        _minimal_worker_run(monkeypatch, tmp_path, gpu_id=0,
                            load_model=_probe)   # adapt to the helper's kwargs
    assert seen == ['0', '0']
```
Read `_minimal_worker_run` first and use its actual mechanism for injecting `load_model_from_path` (the two neighbouring tests show it).

(c) New file `tests/test_run_prediction_gpu_ids.py`:
```python
import types
import winmol_run


def _capture(monkeypatch):
    calls = []

    def fake_run(*a, **k):
        calls.append(k['gpu_ids'])
        return {}
    monkeypatch.setattr('utils.PredictWorkers.run_multi_gpu_prediction',
                        fake_run)
    monkeypatch.setattr(winmol_run, '_release_prediction_memory',
                        lambda *a, **k: None)
    return calls


def _proc():
    p = winmol_run.ImageProcessing.__new__(winmol_run.ImageProcessing)
    p.model_path = 'm.onnx'; p.uav_path = 'in.tif'; p.stem_path = 'out.tif'
    p.config = types.SimpleNamespace(prediction_backend='auto')
    return p


def _plan(mode, gpu_workers, workers_per_gpu):
    return types.SimpleNamespace(
        prediction_mode=mode, gpu_workers=gpu_workers,
        workers_per_gpu=workers_per_gpu, producer_workers=3)


def test_single_gpu_two_workers_expands_to_0_0(monkeypatch):
    calls = _capture(monkeypatch)
    _proc().run_prediction_phase(_plan('multi_gpu', 1, 2))
    assert calls == [[0, 0]]


def test_multi_gpu_one_worker_each_is_unchanged(monkeypatch):
    calls = _capture(monkeypatch)
    _proc().run_prediction_phase(_plan('multi_gpu', 3, 1))
    assert calls == [[0, 1, 2]]


def test_multi_gpu_two_workers_each_interleaves_by_device(monkeypatch):
    calls = _capture(monkeypatch)
    _proc().run_prediction_phase(_plan('multi_gpu', 2, 2))
    assert calls == [[0, 0, 1, 1]]


def test_cpu_stream_stays_a_single_none_worker(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr('utils.onnx_runtime.selected_providers',
                        lambda: ['CPUExecutionProvider'])
    _proc().run_prediction_phase(_plan('cpu_stream', 0, 2))
    assert calls == [[None]]
```
If `run_prediction_phase` touches other attributes of `self`/`plan` than listed (read it), add them to the stubs; the stubs are the point of this test — no real model, no raster.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_run_prediction_gpu_ids.py tests/test_prediction_drain.py tests/test_reader_pool.py -q -rA`
Expected: the three `gpu_ids` expectations fail (`[0]` instead of `[0, 0]`, etc.); the duplicate-shard test may already pass (that is fine — it is the regression guard); the worker test passes once (b) is adapted correctly (also fine: it documents F4 with duplicates).

- [ ] **Step 3: Implement** in `winmol_run.py`:

```python
        else:
            per_gpu = max(1, int(getattr(plan, 'workers_per_gpu', 1) or 1))
            gpu_ids = [g for g in range(max(1, plan.gpu_workers))
                       for _ in range(per_gpu)]
```
Print block: add `print(f"  workers_per_gpu  = {plan.workers_per_gpu}")` after the `gpu_workers` line. Config copy block: add `self.config.prediction_workers_per_gpu = plan.workers_per_gpu` after `prediction_reader_chunk`.

- [ ] **Step 4: Run the tests** — same command, expected all PASS.

- [ ] **Step 5: flake8 + commit**

```bash
flake8 winmol_run.py tests/test_run_prediction_gpu_ids.py tests/test_prediction_drain.py tests/test_reader_pool.py
git add winmol_run.py tests/test_run_prediction_gpu_ids.py tests/test_prediction_drain.py tests/test_reader_pool.py
git commit -m "feat(run): expand gpu_ids by workers_per_gpu; shard across duplicate ids (#60)"
```

---

### Task 3: Full suite, spec cross-check, T14 gate protocol (measurement deferred — T14 unreachable)

**Files:**
- Modify: `benchmark/parity_prediction.py` only if its `--branch` arm needs an env passthrough (read it: it runs a checkout as a subprocess with inherited env, so `WINMOL_WORKERS_PER_GPU` needs nothing). Otherwise no code.
- Create: `.superpowers/sdd/2026-09-16-workers-per-gpu/t14-gate.md` (git-ignored) — the protocol below, filled in when the T14 is back.

- [ ] **Step 1: Full suite + flake8**

Run: `pytest tests/ -q --junitxml=/tmp/junit-wpg.xml -rA` — expected: 478 + new tests pass, 0 failures. `flake8 classes/ExecutionPlan.py winmol_run.py tests/test_execution_plan_readers.py tests/test_run_prediction_gpu_ids.py` clean.

- [ ] **Step 2: Write the gate protocol file** with exactly this content (values verbatim):

```
T14 gate — perf/workers-per-gpu (run when christian@192.168.188.166 is reachable)
Checkout: ~/hnee/wm-wpg  <- git fetch && git checkout origin/perf/workers-per-gpu
1. Probe (9 windows): ~/winmol_bench/probe-one.sh wpg-auto wm-wpg 3 0   (readers 3 = plan default; batch 0 = NO override -> autotune)
   NOTE probe-one.sh passes prediction_batch_override=$batch; for batch 0 edit a copy that omits WINMOL_CONFIG_OVERRIDES_JSON.
   Record: "Performing prediction: 2 worker(s)", both autotune lines, tiles/60 s x9, nvidia-smi util, RSS of both workers + coordinator.
   Pass: every window >= 5,800 tiles/min and last/first >= 0.95 x main's ratio (main: flat 2,512).
2. Full R13 via benchmark/parity_prediction.py, arms: main (ecf83dc) vs wpg — bit-identical raster + GPKG multiset (as Task 7),
   tree-RSS <= main + 4 GB (two interpreters + two CUDA contexts; spec exception), throughput >= main.
3. Optional carrot: WINMOL_WORKERS_PER_GPU=2 on the runbook arm a (pool+trt), R13; compare to 215 s / 51,433 tiles/min. Abort if GPUs busy.
Kill only by PID from nvidia-smi --query-compute-apps=pid; never pkill -f with your own pattern.
```

- [ ] **Step 3: Commit nothing for the gate file (git-ignored). Push the branch.**

```bash
git push -u origin perf/workers-per-gpu
```
(Pushing the fork's own topic branch is within the standing authorisation; never push to upstream.)

---

## Self-review

- Spec coverage: field + rule (T1), gpu_ids expansion + shard proof (T2), reader-thread division (T1), batch/chunk caps (T1), RSS exception + gate numbers (T3 protocol), multi-GPU stays 1 with env override (T1). ✔
- Placeholders: none — Task 2 asks the implementer to adapt fixture names, with the three contract assertions spelled out.
- Type consistency: `workers_per_gpu: int` on the dataclass; `_workers_per_gpu(scen: str, gpu_mem_gb: float) -> int`; `plan.workers_per_gpu` read in `winmol_run.py` with a `getattr` default of 1 so older plans (tests that build `SimpleNamespace` plans) keep working.
