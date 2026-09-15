# Prediction Reader Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple tile reading from inference so every GPU worker is fed by a pool of reader threads instead of reading its own tiles serially between inferences, and route single-GPU and CPU-only runs through the same code.

**Architecture:** A new `utils/reader_pool.py` owns *R* reader threads per GPU-worker process, each with its own `rasterio` handle, feeding a bounded in-process `queue.Queue`. `prediction_worker` in `utils/PredictWorkers.py` consumes from that queue and infers. The coordinator `run_multi_gpu_prediction` gains a timeout-and-liveness drain loop so a dead reader or worker fails the run instead of hanging it. `ExecutionPlan` computes readers-per-worker from `hardware.cpu_count`. Finally `winmol_run.py` sends *all* prediction modes through this path and the old thread-producer stream loop in `utils/Prediction.py` is deleted — gated on a bit-identical parity run.

**Tech Stack:** Python 3.11, `threading` / `queue` (stdlib), `multiprocessing` spawn context (existing), `rasterio` (existing), `numpy`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-15-prediction-reader-pool-design.md`

## Global Constraints

- Branch: `perf/prediction-reader-pool` (off `origin/main`). Never commit to `main` or `fix-first-run`.
- Output must be **bit-identical** to `main`: stem-map raster block-wise, every GPKG layer row for row (WKB + all attributes, file order).
- Throughput **≥** and peak RSS **≤** `main` on the T14 (1 GPU) and on a CPU-only run. Faster is allowed; slower or hungrier fails.
- Readers are **threads**, each with its **own** `rasterio` dataset handle. Upper bound 16 per worker. CPU-only → 1.
- Queue depth = existing plan field `producer_queue_batches`. Readers block on `put`.
- A reader or worker that dies must **fail the run**, never stall it. Timeout 30 s. No partial output with a warning.
- Tests: run with the `WINMOL_Analyzer` conda python from `tests/` cwd; stdout is swallowed, so use `--junitxml` to read results (see memory `winmol-local-test-env`). flake8 must stay clean.
- **The #43 cliff is a gate.** Readers deal batches round-robin so they work adjacent tiles (tested). `GDAL_CACHEMAX` is never scaled with reader count. Task 7 runs a **full ortho on the T14** and compares the **instantaneous** throughput curve; last-10 %/first-10 % ratio must be ≥ main's × 0.95. A crop cannot detect the cliff (onset ~9,500 tiles).
- Concise code: no defensive padding, no speculative options, no docs beyond docstrings that state *why*.
- Deviation from spec, recorded here: the readers-per-worker count is carried in the **existing** plan field `producer_workers` (already wired to `config.prediction_producer_workers`) rather than a new `prediction_readers_per_gpu` field. Same rule, same semantics, one fewer field. After Task 8 that field has exactly one consumer.

---

## File Structure

| File | Responsibility |
|---|---|
| `utils/reader_pool.py` (**create**) | `ReaderPool`: *R* reader threads, own handle each, bounded queue, sentinels, error propagation. No model, no GPU, no knowledge of batches' meaning. |
| `utils/PredictWorkers.py` (**modify**) | `prediction_worker` consumes from a `ReaderPool`; `_drain_results` extracted from `run_multi_gpu_prediction` with timeout + liveness; workers report errors. |
| `classes/ExecutionPlan.py` (**modify**) | `_reader_threads()` sizing rule; all three scenarios use it for `producer_workers`; `WINMOL_PREDICTION_READERS` override. |
| `winmol_run.py` (**modify**) | All prediction modes dispatch to `run_multi_gpu_prediction`. |
| `utils/Prediction.py` (**modify, Task 8**) | Delete `predict_stream_to_raster`, `_split_jobs_for_producers` and the producer thread body. |
| `benchmark/parity_prediction.py` (**create**) | Runs `main` vs branch on one input, compares raster block-wise and GPKG row-wise, records throughput and peak RSS. |
| `tests/test_reader_pool.py` (**create**) | Pool behaviour with a fake `read_fn` and with a real tiny GeoTIFF. |
| `tests/test_prediction_drain.py` (**create**) | `_drain_results`: order-independence, `error` message, dead worker, timeout. |
| `tests/test_execution_plan_readers.py` (**create**) | Sizing table, floor, cap, CPU-only, env override. |

---

### Task 1: `ReaderPool`

**Files:**
- Create: `utils/reader_pool.py`
- Test: `tests/test_reader_pool.py`

**Interfaces:**
- Produces:
  ```python
  class ReaderDied(RuntimeError): ...
  class ReaderPool:
      def __init__(self, input_raster: str, batches: list[list[dict]],
                   read_fn, n_readers: int, queue_depth: int,
                   open_fn=rasterio.open): ...
      def start(self) -> None: ...
      def get(self, timeout: float = 30.0):   # -> (batch_jobs, tiles, masks, stats) | None
      def close(self) -> None: ...
  ```
  `read_fn(src, batch_jobs) -> (raw_tiles, raw_masks, stats)`. `get()` returns `None` exactly once, after every reader has finished. It raises the reader's own exception if a reader failed, and `ReaderDied` if a reader thread is no longer alive without having finished.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reader_pool.py
"""ReaderPool: reader threads decode ahead of inference.

The pool knows nothing about tiles; `read_fn` is injected. These tests use
a fake read_fn for the protocol (sentinels, errors, liveness) and one real
tiny GeoTIFF to prove own-handle-per-thread works against rasterio.
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.reader_pool import ReaderPool, ReaderDied  # noqa: E402


class _FakeSrc:
    """Stands in for a rasterio dataset; records which thread opened it."""
    opened = []

    def __init__(self):
        _FakeSrc.opened.append(threading.get_ident())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_open(path):
    return _FakeSrc()


def _ok_read(src, batch_jobs):
    tiles = [np.full((2, 2), j['id'], np.uint8) for j in batch_jobs]
    masks = [np.ones((2, 2), bool) for _ in batch_jobs]
    return tiles, masks, {'read_s': 0.0}


def _batches(n_batches, per_batch=2):
    return [[{'id': b * per_batch + i} for i in range(per_batch)]
            for b in range(n_batches)]


def test_every_batch_is_delivered_exactly_once_then_none():
    pool = ReaderPool('x.tif', _batches(7), _ok_read,
                      n_readers=3, queue_depth=2, open_fn=_fake_open)
    pool.start()
    seen = []
    while True:
        item = pool.get(timeout=5)
        if item is None:
            break
        batch_jobs, tiles, masks, stats = item
        seen.extend(j['id'] for j in batch_jobs)
        assert len(tiles) == len(masks) == len(batch_jobs)
    pool.close()
    assert sorted(seen) == list(range(14))


def test_each_reader_opens_its_own_handle():
    _FakeSrc.opened.clear()
    pool = ReaderPool('x.tif', _batches(6), _ok_read,
                      n_readers=3, queue_depth=2, open_fn=_fake_open)
    pool.start()
    while pool.get(timeout=5) is not None:
        pass
    pool.close()
    assert len(set(_FakeSrc.opened)) == 3
    assert threading.get_ident() not in _FakeSrc.opened


def test_reader_exception_surfaces_from_get():
    def bad_read(src, batch_jobs):
        raise ValueError("decode failed on tile 3")
    pool = ReaderPool('x.tif', _batches(2), bad_read,
                      n_readers=1, queue_depth=2, open_fn=_fake_open)
    pool.start()
    with pytest.raises(ValueError, match="tile 3"):
        while pool.get(timeout=5) is not None:
            pass
    pool.close()


def test_put_blocks_when_queue_is_full():
    """Backpressure: with depth 1 and a consumer that never reads, at most
    depth + n_readers batches can have been decoded."""
    calls = []
    def counting_read(src, batch_jobs):
        calls.append(1)
        return _ok_read(src, batch_jobs)
    pool = ReaderPool('x.tif', _batches(20), counting_read,
                      n_readers=2, queue_depth=1, open_fn=_fake_open)
    pool.start()
    time.sleep(0.3)
    assert len(calls) <= 1 + 2          # queue slot + one in flight per reader
    while pool.get(timeout=5) is not None:
        pass
    pool.close()
    assert len(calls) == 20


def test_silently_dead_reader_is_detected():
    """A reader that exits without a sentinel must raise ReaderDied on the
    next get() that times out, not hang forever."""
    pool = ReaderPool('x.tif', _batches(2), _ok_read,
                      n_readers=1, queue_depth=2, open_fn=_fake_open)
    # Simulate a thread that died before running: never start it, but
    # mark it as a thread object the pool will inspect.
    pool._threads = [threading.Thread(target=lambda: None)]
    pool._threads[0].start(); pool._threads[0].join()
    with pytest.raises(ReaderDied):
        pool.get(timeout=0.2)


def test_readers_are_dealt_adjacent_batches_round_robin():
    """#43 guard: at any moment the readers hold ONE band of consecutive
    batches, so their overlapping tiles share GDAL blocks. Contiguous
    chunking (reader k gets batches [k*n/R : (k+1)*n/R]) would put R
    readers in R distant bands and multiply the block-cache working set."""
    pool = ReaderPool('x.tif', _batches(12), _ok_read,
                      n_readers=4, queue_depth=2, open_fn=_fake_open)
    first_batch_ids = [sl[0][0]['id'] for sl in pool._slices]
    assert first_batch_ids == [0, 2, 4, 6]        # consecutive batches, not quarters
    assert [len(sl) for sl in pool._slices] == [3, 3, 3, 3]


def test_real_geotiff_windows_through_own_handles(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    path = tmp_path / "t.tif"
    data = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    with rasterio.open(path, 'w', driver='GTiff', width=64, height=64,
                       count=1, dtype='uint8',
                       transform=from_origin(0, 64, 1, 1), crs='EPSG:3857',
                       tiled=True, blockxsize=16, blockysize=16) as dst:
        dst.write(data, 1)

    def read_fn(src, batch_jobs):
        tiles = [src.read(1, window=Window(j['c'], j['r'], 16, 16))
                 for j in batch_jobs]
        return tiles, [np.ones((16, 16), bool)] * len(tiles), {}

    jobs = [{'r': r, 'c': c} for r in range(0, 64, 16) for c in range(0, 64, 16)]
    batches = [jobs[i:i + 4] for i in range(0, 16, 4)]
    pool = ReaderPool(str(path), batches, read_fn, n_readers=4, queue_depth=2)
    pool.start()
    got = {}
    while True:
        item = pool.get(timeout=10)
        if item is None:
            break
        for j, t in zip(item[0], item[1]):
            got[(j['r'], j['c'])] = t
    pool.close()
    assert len(got) == 16
    for (r, c), t in got.items():
        np.testing.assert_array_equal(t, data[r:r + 16, c:c + 16])
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `tests/`): `/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest test_reader_pool.py -q --junitxml=/tmp/rp.xml`
Expected: all FAIL with `ModuleNotFoundError: No module named 'utils.reader_pool'`.

- [ ] **Step 3: Implement `utils/reader_pool.py`**

```python
"""Reader threads that decode tiles ahead of inference.

Before this, each GPU worker read its own tiles serially between
inferences: ~27 ms of CPU decode per tile against ~4 ms of GPU time, so
an 8xH100 box sat ~89% idle. A spike on carrot (2026-09-15, R13) showed
reader THREADS scale 7.5x at 16 -- rasterio releases the GIL around
GDALRasterIO and, with resize in-graph, nothing GIL-bound is left per
tile -- and that 16 threads saturate one H100. So: threads, not
processes, one handle each, a bounded queue for backpressure.

The pool knows nothing about tiles. ``read_fn(src, batch_jobs)`` is
injected by the caller; this module only owns the threads, the queue and
the failure protocol.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, List, Optional

import rasterio


class ReaderDied(RuntimeError):
    """A reader thread is gone without having reported completion."""


class _Done:
    __slots__ = ()


class _Err:
    __slots__ = ('exc',)

    def __init__(self, exc):
        self.exc = exc


class ReaderPool:
    def __init__(
        self,
        input_raster: str,
        batches: List[list],
        read_fn: Callable,
        n_readers: int,
        queue_depth: int,
        open_fn: Callable = rasterio.open,
    ):
        self._path = input_raster
        self._read_fn = read_fn
        self._open_fn = open_fn
        self._n = max(1, int(n_readers))
        # Round-robin split keeps every reader busy to the end of the shard.
        self._slices = [batches[i::self._n] for i in range(self._n)]
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, int(queue_depth)))
        self._threads: List[threading.Thread] = []
        self._finished = 0

    def start(self) -> None:
        for idx, my_batches in enumerate(self._slices):
            t = threading.Thread(
                target=self._run, args=(my_batches,),
                name=f"winmol-reader-{idx}", daemon=True)
            t.start()
            self._threads.append(t)

    def _run(self, my_batches) -> None:
        # One handle PER THREAD. Measured identical to a shared handle,
        # and a shared-handle non-boundless read across threads terminated
        # abnormally in the spike; own handles remove that class of bug.
        try:
            with self._open_fn(self._path) as src:
                for batch_jobs in my_batches:
                    tiles, masks, stats = self._read_fn(src, batch_jobs)
                    self._q.put((batch_jobs, tiles, masks, stats))
        except BaseException as exc:          # noqa: BLE001 -- must reach the consumer
            self._q.put(_Err(exc))
            return
        self._q.put(_Done())

    def get(self, timeout: float = 30.0):
        """Next decoded batch, or None once every reader has finished.

        Raises the reader's own exception if one failed, and ReaderDied if
        a thread is dead without having put its sentinel. A run must fail
        loudly; it must never wait forever on a queue nobody will fill.
        """
        while True:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                alive = sum(t.is_alive() for t in self._threads)
                if alive + self._finished < self._n:
                    raise ReaderDied(
                        f"{self._n - alive - self._finished} reader thread(s) "
                        "exited without finishing")
                continue
            if isinstance(item, _Err):
                raise item.exc
            if isinstance(item, _Done):
                self._finished += 1
                if self._finished == self._n:
                    return None
                continue
            return item

    def close(self) -> None:
        for t in self._threads:
            t.join(timeout=5.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -m pytest test_reader_pool.py -q --junitxml=/tmp/rp.xml`
Expected: 7 passed. Then `flake8 utils/reader_pool.py tests/test_reader_pool.py` from repo root: exit 0.

- [ ] **Step 5: Commit**

```bash
git add utils/reader_pool.py tests/test_reader_pool.py
git commit -m "feat(predict): ReaderPool -- reader threads decode ahead of inference (#60)"
```

---

### Task 2: `prediction_worker` consumes from a `ReaderPool`

**Files:**
- Modify: `utils/PredictWorkers.py:208-244` (`prediction_worker`)
- Test: `tests/test_reader_pool.py` (add one test)

**Interfaces:**
- Consumes: `ReaderPool`, `_group_jobs`, `_read_batch_jobs`, `_graph_out_size` (all existing).
- Produces: `prediction_worker(gpu_id: int | None, model_path, input_raster, jobs, results, config_dict)`. `gpu_id=None` means CPU: no `CUDA_VISIBLE_DEVICES` is set. On failure it puts `{'error': str, 'gpu_id': gpu_id}` on `results` and returns; on success it puts `{'done': True, 'gpu_id': gpu_id}` as before. Reader count from `config_dict['prediction_producer_workers']`, depth from `config_dict['producer_queue_batches']`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reader_pool.py`:

```python
def test_prediction_worker_reports_error_instead_of_dying(monkeypatch, tmp_path):
    """A worker that fails must put {'error': ...}, never vanish: the
    coordinator's drain loop (Task 3) turns that into a failed run."""
    from utils import PredictWorkers as PW
    results = []
    class _Q:
        def put(self, x):
            results.append(x)
    monkeypatch.setattr(PW, "_config_from_dict",
                        lambda d: (_ for _ in ()).throw(RuntimeError("boom")))
    PW.prediction_worker(0, "m.onnx", str(tmp_path / "nope.tif"), [], _Q(),
                         {'prediction_producer_workers': 2,
                          'producer_queue_batches': 2})
    assert results and 'error' in results[-1]
    assert "boom" in results[-1]['error']
    assert results[-1]['gpu_id'] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_reader_pool.py::test_prediction_worker_reports_error_instead_of_dying -q`
Expected: FAIL — `RuntimeError: boom` propagates instead of being reported.

- [ ] **Step 3: Rewrite `prediction_worker`**

Replace lines 208–244 of `utils/PredictWorkers.py` with:

```python
def prediction_worker(
    gpu_id,
    model_path: str,
    input_raster: str,
    jobs: List[dict],
    results,
    config_dict: dict,
):
    """One process per accelerator. gpu_id=None means CPU.

    Reads are NOT done here any more: a ReaderPool of threads decodes
    batches ahead of us into a bounded queue, so the GPU never waits on
    a 27 ms decode. Any failure -- ours or a reader's -- is reported on
    `results` as {'error': ...}; the coordinator fails the run on it.
    """
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        from utils.IO import load_model_from_path
        from utils.reader_pool import ReaderPool

        cfg = _config_from_dict(config_dict)
        # Wrapped, so the normalize + resize run on THIS worker's device
        # instead of on the CPU inside the timed inference block.
        model = load_model_from_path(model_path, cfg)
        out_size = _graph_out_size(cfg)
        batch_size = max(1, int(getattr(cfg, 'prediction_batch_size', None)
                                or getattr(cfg, 'prediction_batch_gpu', 4)))
        n_readers = max(1, int(config_dict.get(
            'prediction_producer_workers', 1) or 1))
        depth = max(1, int(config_dict.get('producer_queue_batches', 4) or 4))

        with rasterio.open(input_raster) as probe:
            indexes = list(range(1, min(cfg.n_channels, probe.count) + 1))

        pool = ReaderPool(
            input_raster, list(_group_jobs(jobs, batch_size)),
            lambda src, b: _read_batch_jobs(src, indexes, b, out_size),
            n_readers=n_readers, queue_depth=depth)
        pool.start()
        try:
            while True:
                item = pool.get()
                if item is None:
                    break
                batch_jobs, raw_tiles, raw_masks, read_stats = item
                infer0 = time.perf_counter()
                pred_cores = _predict_batch(raw_tiles, raw_masks, model, cfg)
                infer_s = time.perf_counter() - infer0
                n = max(len(batch_jobs), 1)
                for job, pred_core in zip(batch_jobs, pred_cores):
                    results.put({
                        'row_off': job['dst_row'],
                        'col_off': job['dst_col'],
                        'array': pred_core,
                        'read_s': read_stats['read_s'] / n,
                        'infer_s': infer_s / n,
                    })
        finally:
            pool.close()
    except BaseException as exc:      # noqa: BLE001 -- report, never vanish
        results.put({'error': f"{type(exc).__name__}: {exc}",
                     'gpu_id': gpu_id})
        return
    results.put({'done': True, 'gpu_id': gpu_id})
```

- [ ] **Step 4: Run the tests**

Run: `pytest test_reader_pool.py -q --junitxml=/tmp/rp.xml`
Expected: 8 passed. Then the full suite: `pytest -q --junitxml=/tmp/all.xml` — expected: previous count + 8, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add utils/PredictWorkers.py tests/test_reader_pool.py
git commit -m "feat(predict): GPU worker consumes from a ReaderPool; reports errors instead of dying (#60)"
```

---

### Task 3: Coordinator drain loop with timeout and liveness

**Files:**
- Modify: `utils/PredictWorkers.py:485-534` (the `while finished < len(workers)` loop inside `run_multi_gpu_prediction`)
- Test: `tests/test_prediction_drain.py`

**Interfaces:**
- Produces:
  ```python
  class PredictionWorkerFailed(RuntimeError): ...
  def _drain_results(result_q, workers, write_tile, total_tiles,
                     progress_interval_s, timeout_s=30.0,
                     print_fn=print) -> dict
  ```
  `write_tile(row_off: int, col_off: int, arr: np.ndarray) -> float` writes one tile and returns seconds spent. `workers` are objects with `.is_alive()`. Returns `{'read_s', 'infer_s', 'write_s', 'done'}`. Raises `PredictionWorkerFailed` on an `error` message, on a worker that is dead without `done`, and on a timeout with no live workers.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prediction_drain.py
"""run_multi_gpu_prediction's drain loop: order-independent assembly, and a
run that FAILS on a dead or erroring worker instead of hanging on
result_q.get() forever (which is what it did before)."""
import queue
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.PredictWorkers import _drain_results, PredictionWorkerFailed  # noqa: E402


class _Worker:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


def _canvas_writer(canvas):
    def write_tile(row_off, col_off, arr):
        h, w = arr.shape
        canvas[row_off:row_off + h, col_off:col_off + w] = arr
        return 0.0
    return write_tile


def _tile_results(n_side=4, size=8):
    out = []
    for r in range(n_side):
        for c in range(n_side):
            arr = np.full((size, size), r * n_side + c + 1, np.uint8)
            out.append({'row_off': r * size, 'col_off': c * size,
                        'array': arr, 'read_s': 0.0, 'infer_s': 0.0})
    return out


def _run(results_in_order, workers, timeout_s=1.0):
    q = queue.Queue()
    for item in results_in_order:
        q.put(item)
    canvas = np.zeros((32, 32), np.uint8)
    stats = _drain_results(q, workers, _canvas_writer(canvas), total_tiles=16,
                           progress_interval_s=1e9, timeout_s=timeout_s,
                           print_fn=lambda *a, **k: None)
    return canvas, stats


def test_assembly_is_order_independent():
    tiles = _tile_results()
    ordered, _ = _run(tiles + [{'done': True, 'gpu_id': 0}], [_Worker()])
    shuffled = tiles[:]
    random.Random(3).shuffle(shuffled)
    mixed, stats = _run(shuffled + [{'done': True, 'gpu_id': 0}], [_Worker()])
    np.testing.assert_array_equal(ordered, mixed)
    assert stats['done'] == 16


def test_error_message_fails_the_run():
    with pytest.raises(PredictionWorkerFailed, match="OOM on gpu 2"):
        _run([{'error': 'RuntimeError: OOM on gpu 2', 'gpu_id': 2}],
             [_Worker()])


def test_dead_worker_without_done_fails_the_run():
    # Nothing on the queue, the only worker is dead: must not block.
    with pytest.raises(PredictionWorkerFailed, match="exited"):
        _run([], [_Worker(alive=False)], timeout_s=0.2)


def test_waits_while_workers_are_alive_and_silent():
    """Alive but slow is fine: the loop keeps waiting past one timeout."""
    q = queue.Queue()
    import threading
    def late():
        import time; time.sleep(0.5); q.put({'done': True, 'gpu_id': 0})
    threading.Thread(target=late, daemon=True).start()
    canvas = np.zeros((8, 8), np.uint8)
    stats = _drain_results(q, [_Worker()], _canvas_writer(canvas), total_tiles=0,
                           progress_interval_s=1e9, timeout_s=0.2,
                           print_fn=lambda *a, **k: None)
    assert stats['done'] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_prediction_drain.py -q`
Expected: FAIL with `ImportError: cannot import name '_drain_results'`.

- [ ] **Step 3: Extract `_drain_results` and use it**

In `utils/PredictWorkers.py`, add above `run_multi_gpu_prediction`:

```python
class PredictionWorkerFailed(RuntimeError):
    """A prediction worker reported an error or died. The run fails."""


def _drain_results(result_q, workers, write_tile, total_tiles,
                   progress_interval_s, timeout_s=30.0, print_fn=print):
    """Consume worker results until every worker has said 'done'.

    Assembly is order-independent: each result carries its own output
    window, and windows are the tiles' non-overlapping cores. That is
    what lets readers deliver a shard's tiles in any order.

    Failure is loud. An 'error' message, a worker that is dead without
    'done', or a timeout with nobody left alive all raise; the old loop
    blocked on result_q.get() forever in each of those cases.
    """
    finished = 0
    done = 0
    start = time.monotonic()
    last_report = start
    total_read_s = total_infer_s = total_write_s = 0.0

    while finished < len(workers):
        try:
            result = result_q.get(timeout=timeout_s)
        except queue.Empty:
            alive = [w for w in workers if w.is_alive()]
            if len(alive) + finished < len(workers):
                raise PredictionWorkerFailed(
                    f"{len(workers) - len(alive) - finished} prediction "
                    "worker(s) exited without reporting completion")
            continue
        if result.get('error'):
            raise PredictionWorkerFailed(
                f"prediction worker (gpu {result.get('gpu_id')}) failed: "
                f"{result['error']}")
        if result.get('done'):
            finished += 1
            continue
        total_write_s += write_tile(int(result['row_off']),
                                    int(result['col_off']), result['array'])
        total_read_s += float(result.get('read_s', 0.0))
        total_infer_s += float(result.get('infer_s', 0.0))
        done += 1
        now = time.monotonic()
        if done == 1 or done == total_tiles \
                or (now - last_report) >= progress_interval_s:
            elapsed = max(now - start, 1e-9)
            rate = done / elapsed
            eta_s = (total_tiles - done) / rate if rate > 0 else float('inf')
            print_fn(
                f"Multi-GPU prediction {done}/{total_tiles} | "
                f"{done / max(total_tiles, 1):.1%} | {rate * 60:.1f} tiles/min"
                f" | ETA {_format_eta(eta_s)} | avg read "
                f"{total_read_s / max(done, 1):.3f}s infer "
                f"{total_infer_s / max(done, 1):.3f}s write "
                f"{total_write_s / max(done, 1):.3f}s",
                flush=True)
            last_report = now
    return {'read_s': total_read_s, 'infer_s': total_infer_s,
            'write_s': total_write_s, 'done': done}
```

Add `import queue` to the module imports. Then replace the body of `run_multi_gpu_prediction` from `total_tiles = len(all_jobs)` through the end of the `with rasterio.open(tmp_path, 'w', **out_profile) as dst:` block with:

```python
    total_tiles = len(all_jobs)
    progress_interval_s = float(getattr(config, 'progress_interval_s', 20.0))

    with rasterio.open(tmp_path, 'w', **out_profile) as dst:
        def write_tile(row_off, col_off, arr):
            write_h = min(arr.shape[0], layout['out_height'] - row_off)
            write_w = min(arr.shape[1], layout['out_width'] - col_off)
            t0 = time.perf_counter()
            dst.write(
                np.ascontiguousarray(arr[:write_h, :write_w], dtype=np.uint8),
                1, window=Window(col_off, row_off, write_w, write_h))
            return time.perf_counter() - t0

        try:
            _drain_results(result_q, workers, write_tile, total_tiles,
                           progress_interval_s)
        except PredictionWorkerFailed:
            for p in workers:
                if p.is_alive():
                    p.terminate()
            raise
```

Keep the trailing `for p in workers: p.join()`, `IO.finalize_raster(...)`, `return out_profile`.

- [ ] **Step 4: Run the tests**

Run: `pytest test_prediction_drain.py test_reader_pool.py -q --junitxml=/tmp/d.xml`
Expected: 12 passed. Full suite: 0 failures. flake8 clean.

- [ ] **Step 5: Commit**

```bash
git add utils/PredictWorkers.py tests/test_prediction_drain.py
git commit -m "fix(predict): coordinator fails the run on a dead or erroring worker instead of hanging (#60)"
```

---

### Task 4: Sizing rule in `ExecutionPlan`

**Files:**
- Modify: `classes/ExecutionPlan.py` — add `_reader_threads`; replace the three `producer_workers = ...` assignments at ~334, ~381, ~433.
- Test: `tests/test_execution_plan_readers.py`

**Interfaces:**
- Produces: `_reader_threads(hw_cpu: int, n_gpu: int, cpu_only: bool) -> int`. `ExecutionPlan.producer_workers` now means *reader threads per prediction worker*, computed by this rule, overridable by env `WINMOL_PREDICTION_READERS`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_execution_plan_readers.py
"""producer_workers is now 'reader threads per prediction worker':
R = clamp((hw_cpu - 1) // n_gpu, 1, 16); CPU-only is always 1.

Not derived from cpu_workers: that is capped at 32 by max_cpu_workers,
which on an 8-GPU box would give 4 readers per GPU -- ~136 tiles/s
against a card that consumes 250."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from classes.ExecutionPlan import _reader_threads  # noqa: E402


@pytest.mark.parametrize("hw_cpu,n_gpu,cpu_only,expected", [
    (224, 8, False, 16),    # carrot: 223 // 8 = 27 -> cap 16
    (12, 1, False, 11),     # T14
    (4, 1, False, 3),
    (2, 1, False, 1),       # floor
    (1, 1, False, 1),       # floor, never 0
    (64, 2, False, 16),     # 63 // 2 = 31 -> cap
    (224, 8, True, 1),      # CPU-only ignores cores
    (12, 1, True, 1),
])
def test_reader_threads_rule(hw_cpu, n_gpu, cpu_only, expected):
    assert _reader_threads(hw_cpu, n_gpu, cpu_only) == expected


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("WINMOL_PREDICTION_READERS", "5")
    assert _reader_threads(224, 8, False) == 5
    assert _reader_threads(224, 8, True) == 5


def test_env_override_ignored_when_not_an_int(monkeypatch):
    monkeypatch.setenv("WINMOL_PREDICTION_READERS", "lots")
    assert _reader_threads(12, 1, False) == 11
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_execution_plan_readers.py -q`
Expected: FAIL — `ImportError: cannot import name '_reader_threads'`.

- [ ] **Step 3: Implement the rule and use it in all three scenarios**

Add to `classes/ExecutionPlan.py` (near `_scenario`):

```python
#: Reader threads per prediction worker. Measured on carrot 2026-09-15:
#: reader threads scale to ~7.5x at 16 and then plateau, and 16 saturate
#: one H100 at 4 ms/tile. CPU-only inference is the bottleneck, so extra
#: readers there only spend RAM.
READERS_MAX = 16
ENV_READERS = 'WINMOL_PREDICTION_READERS'


def _reader_threads(hw_cpu: int, n_gpu: int, cpu_only: bool) -> int:
    override = os.environ.get(ENV_READERS, '').strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if cpu_only:
        return 1
    # hw_cpu - 1: the one core the plan already leaves for the OS and the
    # coordinator. No vector-phase reserve: the phases are sequential.
    return max(1, min(READERS_MAX,
                      (max(1, int(hw_cpu)) - 1) // max(1, int(n_gpu))))
```

Add `import os` to the module imports if absent. Then:

- In the `CPU_ONLY` branch replace the `producer_workers = max(1, int(_cfg(config, 'prediction_producer_workers_cpu', 1)))` block with `producer_workers = _reader_threads(hw_cpu, 1, True)`.
- In the `SINGLE_GPU` branch delete `requested_producers = ...`, `producer_caps = ...`, the `if cpu_workers < 10:` append, and the `producer_workers = _apply_caps(...)` call; replace with `producer_workers = _reader_threads(hw_cpu, 1, False)`.
- In the multi-GPU branch replace `producer_workers = max(1, int(_cfg(config, 'prediction_producer_workers_multi_gpu', 2)))` with `producer_workers = _reader_threads(hw_cpu, gpu_workers, False)`.

`hw_cpu` is already defined above the scenario branches (`hw_cpu = max(1, int(getattr(hardware, 'cpu_count', 1) or 1))`).

- [ ] **Step 4: Run the tests**

Run: `pytest test_execution_plan_readers.py -q` → 10 passed. Full suite: 0 failures. If any existing test asserted the old `producer_workers` values (grep `producer_workers` in `tests/`), update its expected value to the new rule and say so in the commit body.

- [ ] **Step 5: Commit**

```bash
git add classes/ExecutionPlan.py tests/test_execution_plan_readers.py
git commit -m "feat(plan): readers per prediction worker from cpu_count, capped at 16 (#60)"
```

---

### Task 5: Autotune and OOM back-off inside the worker

**Files:**
- Modify: `utils/PredictWorkers.py` — `prediction_worker` (from Task 2); delete `_predict_batch` (lines ~100–123).
- Test: `tests/test_reader_pool.py` (add one test)

**Interfaces:**
- Consumes: `utils.Prediction._autotune_batch_size(sample_tiles, sample_masks, model, config, initial_batch, label=...)` → int; `utils.Prediction._predict_batch_adaptive(raw_tiles, raw_masks, model, config, batch_size)` → `(pred_cores, used_batch)`.
- Why: the stream path the laptop uses today autotunes the micro-batch once and halves it on OOM. Routing laptops through this worker (Task 6) without those loses them, and a 12 GB card would OOM where it used to recover. `_predict_batch_core` (used by `_predict_batch_adaptive`) is the same computation as `_predict_batch`, so the latter is deleted rather than kept as a second copy.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reader_pool.py`:

```python
def test_worker_autotunes_once_then_uses_adaptive_batches(monkeypatch, tmp_path):
    from utils import PredictWorkers as PW
    calls = {'autotune': 0, 'adaptive': []}
    class _Cfg:
        n_channels = 1; prediction_batch_size = 4; img_height = img_width = 8
    monkeypatch.setattr(PW, "_config_from_dict", lambda d: _Cfg())
    monkeypatch.setattr(PW, "_graph_out_size", lambda cfg: None)
    monkeypatch.setattr("utils.IO.load_model_from_path", lambda p, c: object())
    monkeypatch.setattr(PW, "_read_batch_jobs",
                        lambda src, idx, b, o: ([np.zeros((8, 8, 1), np.uint8)] * len(b),
                                               [np.ones((8, 8), bool)] * len(b),
                                               {'read_s': 0.0}))
    def fake_autotune(*a, **k):
        calls['autotune'] += 1; return 2
    def fake_adaptive(tiles, masks, model, cfg, batch_size):
        calls['adaptive'].append(batch_size)
        return [np.zeros((8, 8), np.uint8) for _ in tiles], batch_size
    monkeypatch.setattr("utils.Prediction._autotune_batch_size", fake_autotune)
    monkeypatch.setattr("utils.Prediction._predict_batch_adaptive", fake_adaptive)
    import rasterio
    from rasterio.transform import from_origin
    path = tmp_path / "p.tif"
    with rasterio.open(path, 'w', driver='GTiff', width=8, height=8, count=1,
                       dtype='uint8', transform=from_origin(0, 8, 1, 1),
                       crs='EPSG:3857') as d:
        d.write(np.zeros((8, 8), np.uint8), 1)
    jobs = [{'dst_row': 0, 'dst_col': i, 'src_row': 0, 'src_col': 0,
             'src_width': 8, 'src_height': 8} for i in range(6)]
    out = []
    class _Q:
        def put(self, x): out.append(x)
    PW.prediction_worker(None, "m.onnx", str(path), jobs, _Q(),
                         {'prediction_producer_workers': 1,
                          'producer_queue_batches': 2})
    assert calls['autotune'] == 1
    assert calls['adaptive'] and all(b == 2 for b in calls['adaptive'])
    assert out[-1] == {'done': True, 'gpu_id': None}
    assert sum(1 for o in out if 'array' in o) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_reader_pool.py::test_worker_autotunes_once_then_uses_adaptive_batches -q`
Expected: FAIL — `calls['autotune'] == 0` (worker never autotunes).

- [ ] **Step 3: Wire autotune + adaptive into the worker; delete `_predict_batch`**

In `prediction_worker`, replace the consume loop from Task 2 with:

```python
        from utils.Prediction import (_autotune_batch_size,
                                      _predict_batch_adaptive)
        active_batch = None
        try:
            while True:
                item = pool.get()
                if item is None:
                    break
                batch_jobs, raw_tiles, raw_masks, read_stats = item
                if active_batch is None:
                    # Once per worker, on real tiles -- the stream path
                    # did exactly this; without it a 12 GB card OOMs where
                    # it used to back off.
                    active_batch = _autotune_batch_size(
                        raw_tiles, raw_masks, model, cfg, batch_size,
                        label='Prediction micro-batch')
                infer0 = time.perf_counter()
                pred_cores, active_batch = _predict_batch_adaptive(
                    raw_tiles, raw_masks, model, cfg, active_batch)
                infer_s = time.perf_counter() - infer0
                n = max(len(batch_jobs), 1)
                for job, pred_core in zip(batch_jobs, pred_cores):
                    results.put({
                        'row_off': job['dst_row'],
                        'col_off': job['dst_col'],
                        'array': pred_core,
                        'read_s': read_stats['read_s'] / n,
                        'infer_s': infer_s / n,
                    })
        finally:
            pool.close()
```

Delete `_predict_batch` from `utils/PredictWorkers.py` and remove `_prepare_inference_batch` from its `from utils.Prediction import (...)` line if nothing else in the file uses it.

- [ ] **Step 4: Run the tests**

Run: `pytest test_reader_pool.py -q` → 9 passed. Full suite: 0 failures. flake8 clean.

- [ ] **Step 5: Commit**

```bash
git add utils/PredictWorkers.py tests/test_reader_pool.py
git commit -m "feat(predict): worker autotunes once and backs off on OOM, as the stream path did (#60)"
```

---

### Task 6: All prediction modes dispatch to the pooled path

**Files:**
- Modify: `winmol_run.py:151-185` (`run_prediction_phase`)
- Modify: `utils/PredictWorkers.py` — `run_multi_gpu_prediction` accepts `gpu_ids=[None]` for CPU.

**Interfaces:**
- Consumes: `run_multi_gpu_prediction(model_path, input_raster, output_raster, tile_jobs, gpu_ids, config)`; `prediction_worker` with `gpu_id=None` (Task 2).
- Produces: `run_prediction_phase` returns `(None, profile, stem_path)` for every mode, as the multi-GPU branch already does.

- [ ] **Step 1: Make CPU a first-class worker target**

In `run_multi_gpu_prediction`, change the guard:

```python
    if not gpu_ids:
        raise ValueError('run_multi_gpu_prediction requires at least one '
                         'accelerator id; use [None] for CPU')
```

Nothing else in the coordinator depends on `gpu_ids` being ints (they are passed straight to `prediction_worker`).

- [ ] **Step 2: Rewrite `run_prediction_phase`**

Replace the body of `WINMOLRun.run_prediction_phase` (from `if plan.prediction_mode == 'multi_gpu_stream'` through the `Pred.predict_stream_to_raster(...)` call and its return) with:

```python
    def run_prediction_phase(self, plan):
        from utils.PredictWorkers import run_multi_gpu_prediction

        if plan.prediction_mode == 'cpu_stream':
            from utils.onnx_runtime import selected_providers
            if _cpu_stream_forces_onnx_cpu(
                    getattr(self.config, 'prediction_backend', 'auto'),
                    selected_providers()):
                os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"
            gpu_ids = [None]
        else:
            gpu_ids = list(range(max(1, plan.gpu_workers)))

        # The plugin's progress parser keys on this line; keep it.
        print("\nLoading Model...")
        print(f"\nPerforming prediction: {len(gpu_ids)} worker(s), "
              f"{plan.producer_workers} reader thread(s) each...")
        profile = run_multi_gpu_prediction(
            self.model_path,
            self.uav_path,
            self.stem_path,
            tile_jobs=None,
            gpu_ids=gpu_ids,
            config=self.config,
        )
        return (None, profile, self.stem_path)
```

Remove the now-unused `from utils import Prediction as Pred` and `IO.load_model_from_path` model load from this method (the worker loads the model). Keep `_cpu_stream_forces_onnx_cpu`.

- [ ] **Step 3: Run the full suite and a smoke run**

Run: `pytest -q --junitxml=/tmp/all.xml` → 0 failures. Then a CPU smoke run on a small ortho you have locally, e.g. the standalone sample:

```bash
cd /Users/christian/hnee/WINMOL_Analyzer
/Users/christian/opt/anaconda3/envs/WINMOL_Analyzer/bin/python -u winmol_run.py \
  standalone/model_onnx/Spruce_Deadwood.onnx <small_ortho.tif> \
  /tmp/smoke_stem.tif /tmp/smoke_out Nodes 2>&1 | grep -E "Multi-GPU prediction|Total stems|Elapsed|Error|Traceback" | tail
```

Expected: `Multi-GPU prediction N/N | 100.0%` lines, a `Total stems written` line, no traceback.

- [ ] **Step 4: Commit**

```bash
git add winmol_run.py utils/PredictWorkers.py
git commit -m "feat(predict): every mode -- CPU, one GPU, many -- runs through the pooled worker path (#60)"
```

---

### Task 7: Parity and no-regression measurement

**Files:**
- Create: `benchmark/parity_prediction.py`
- Record results in the PR description (numbers, not assertions).

**Interfaces:**
- Produces: a script that, given two checkouts and one ortho, runs both, compares the stem-map raster block-wise and every GPKG layer row-wise, and prints throughput and peak RSS for each. Exit 1 on any difference.

- [ ] **Step 1: Write the harness**

```python
#!/usr/bin/env python
"""Gate (ii) for the reader pool: bit-identical output, throughput >=,
peak RSS <= main. Run on the T14 (1 GPU), once CPU-only, and on carrot.

    python benchmark/parity_prediction.py --base /path/to/main-checkout \
        --branch /path/to/branch-checkout --model <model> --ortho <tif> \
        --out /tmp/parity [--cpu]
"""
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd


def run(checkout, model, ortho, outdir, cpu):
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "stem.tif"
    env = dict(os.environ)
    if cpu:
        env["WINMOL_ONNX_FORCE_CPU"] = "1"
    # Timestamp every stdout line as it ARRIVES: the progress line carries
    # only the cumulative average, and the #43 cliff is invisible in a
    # cumulative average until long after it happened.
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-u", "winmol_run.py", model, ortho,
         str(stem), str(outdir / "out"), "Nodes"],
        cwd=checkout, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    stamped = []                       # (seconds, done_tiles) per progress line
    log = open(outdir / "run.log", "w")
    for line in proc.stdout:
        log.write(line)
        if "tiles/min" in line and ("prediction" in line or "Written tile" in line):
            done = int(line.split("|")[0].split()[-1].split("/")[0])
            stamped.append((time.perf_counter() - t0, done))
    proc.wait(); log.close()
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{checkout}: exit {proc.returncode}, see {outdir}/run.log")
    rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {"wall_s": wall, "curve": stamped, "peak_rss_mb": rss_kb / 1024,
            "stem": stem, "gpkg": sorted(outdir.glob("**/*.gpkg"))}


def inst_rate(curve, lo, hi):
    """Instantaneous tiles/min over the slice of the run where done is in
    [lo, hi] of the total -- from successive progress lines, never from the
    cumulative average."""
    total = curve[-1][1]
    pts = [(t, d) for t, d in curve if lo * total <= d <= hi * total]
    if len(pts) < 2:
        return float("nan")
    (t0, d0), (t1, d1) = pts[0], pts[-1]
    return 60.0 * (d1 - d0) / max(t1 - t0, 1e-9)


def cliff_ratio(curve):
    """last-10% rate / first-10% rate. ~1.0 is flat; the #43 collapse on
    the T14 measured ~0.3 (2288 -> 677/min on R13)."""
    return inst_rate(curve, 0.9, 1.0) / inst_rate(curve, 0.0, 0.1)


def same_raster(a, b):
    with rasterio.open(a) as ra, rasterio.open(b) as rb:
        if ra.shape != rb.shape or ra.count != rb.count:
            return False, "shape/count differ"
        diff = 0
        for _, win in ra.block_windows(1):
            diff += int(np.count_nonzero(ra.read(1, window=win) != rb.read(1, window=win)))
        return diff == 0, f"{diff} differing px"


def same_gpkg(a, b):
    import pyogrio
    for layer in pyogrio.list_layers(a)[:, 0]:
        ga = gpd.read_file(a, layer=layer, engine="pyogrio")
        gb = gpd.read_file(b, layer=layer, engine="pyogrio")
        if len(ga) != len(gb):
            return False, f"{layer}: {len(ga)} vs {len(gb)} rows"
        if list(ga.geometry.to_wkb()) != list(gb.geometry.to_wkb()):
            return False, f"{layer}: geometry differs"
        cols = [c for c in ga.columns if c != ga.geometry.name]
        if not ga[cols].equals(gb[cols]):
            return False, f"{layer}: attributes differ"
    return True, "identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--branch", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--ortho", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    base = run(a.base, a.model, a.ortho, out / "base", a.cpu)
    br = run(a.branch, a.model, a.ortho, out / "branch", a.cpu)
    ok_r, why_r = same_raster(base["stem"], br["stem"])
    ok_g, why_g = (True, "no gpkg") if not base["gpkg"] else same_gpkg(base["gpkg"][-1], br["gpkg"][-1])
    faster = br["wall_s"] <= base["wall_s"]
    leaner = br["peak_rss_mb"] <= base["peak_rss_mb"]
    # #43 gate: the pool must not cliff where main does not.
    base_cliff, br_cliff = cliff_ratio(base["curve"]), cliff_ratio(br["curve"])
    flat = br_cliff >= base_cliff * 0.95
    report = {"raster": why_r, "gpkg": why_g,
              "base": {"wall_s": base["wall_s"], "peak_rss_mb": base["peak_rss_mb"],
                       "first10_tpm": inst_rate(base["curve"], 0, .1),
                       "last10_tpm": inst_rate(base["curve"], .9, 1), "cliff_ratio": base_cliff},
              "branch": {"wall_s": br["wall_s"], "peak_rss_mb": br["peak_rss_mb"],
                         "first10_tpm": inst_rate(br["curve"], 0, .1),
                         "last10_tpm": inst_rate(br["curve"], .9, 1), "cliff_ratio": br_cliff},
              "throughput_ok": faster, "rss_ok": leaner, "no_cliff_ok": flat}
    print(json.dumps(report, indent=2))
    sys.exit(0 if (ok_r and ok_g and faster and leaner and flat) else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the T14, GPU and CPU — on a FULL ortho**

The #43 cliff has a position (~9,500 tiles on R13). A crop passes this gate while hiding the collapse. Use Tegel R12 (75,072 tiles) or R13 from `/data/mnt/storage` on the T14, on its ZFS storage, not tmpfs. Expect ~20–60 min per arm on the GPU.

```bash
# on the T14 (ssh christian@192.168.188.166), two checkouts side by side
git clone -q https://github.com/cwinkelmann/WINMOL_Analyzer.git /tmp/wm-main   && git -C /tmp/wm-main checkout -q origin/main
git clone -q https://github.com/cwinkelmann/WINMOL_Analyzer.git /tmp/wm-branch && git -C /tmp/wm-branch checkout -q origin/perf/prediction-reader-pool
python /tmp/wm-branch/benchmark/parity_prediction.py --base /tmp/wm-main --branch /tmp/wm-branch \
  --model <Spruce_Deadwood fp16 onnx> --ortho <Barnekow or an R12 crop> --out /tmp/parity-gpu
python /tmp/wm-branch/benchmark/parity_prediction.py --base /tmp/wm-main --branch /tmp/wm-branch \
  --model <same> --ortho <same> --out /tmp/parity-cpu --cpu
```

Expected: both exit 0, `"raster": "0 differing px"`, `"gpkg": "identical"`, `throughput_ok`, `rss_ok` and `no_cliff_ok` all true. Paste both JSON reports into the PR. **If either exits 1, stop: Task 8 does not run.** If `no_cliff_ok` is the failure, the fix is a RAM term in `_reader_threads` — bound R by `GDAL_CACHEMAX / per-reader working set` — measured, not guessed; re-run this step after it.

- [ ] **Step 3: Run it on carrot, R13**

Same two-checkout pattern under `/raid/cwinkelmann/winmol/bench/`, through the rootless daemon per `reports/carrot-bench-runbook.sh` (`DOCKER_HOST`, `--user 0:0`, `loginctl enable-linger` already set). Expected: identical output; prediction throughput well above the 13,657 tiles/min baseline. Record the number.

- [ ] **Step 4: Commit the harness**

```bash
git add benchmark/parity_prediction.py
git commit -m "bench: parity + no-regression gate for the reader pool (#60)"
```

---

### Task 8: Delete the stream path (gated on Task 7)

**Files:**
- Modify: `utils/Prediction.py` — delete `predict_stream_to_raster` (~1059–1273), `_split_jobs_for_producers` (~1043–1058), the producer thread body they use, and the `import queue` / `import threading` if now unused.
- Modify: `classes/ExecutionPlan.py` / `classes/Config.py` — remove `prediction_producer_workers_cpu`, `prediction_producer_workers_gpu`, `prediction_producer_workers_multi_gpu` defaults if nothing reads them (grep first).

**Precondition:** Task 7 Step 2 and Step 3 both exited 0. Do not start this task otherwise.

- [ ] **Step 1: Confirm nothing else calls the stream path**

Run: `grep -rn "predict_stream_to_raster\|_split_jobs_for_producers" --include=*.py . | grep -v "^./tests/"`
Expected: only the definitions in `utils/Prediction.py`. If a caller appears, it is a bug in Task 6 — fix that first.

- [ ] **Step 2: Delete, then run the suite**

Delete the functions listed above. Any test that imported them (grep `tests/` for the names) tests behaviour the pooled worker now owns — port its assertion onto `prediction_worker` via the `test_reader_pool.py` pattern, or delete it if `test_reader_pool.py` already covers the same case. Run: `pytest -q --junitxml=/tmp/all.xml` → 0 failures; `flake8 utils/Prediction.py classes/ExecutionPlan.py` → exit 0.

- [ ] **Step 3: Commit**

```bash
git add utils/Prediction.py classes/ExecutionPlan.py classes/Config.py tests/
git commit -m "refactor(predict): drop the thread-producer stream path; the pooled worker is the only one (#60)"
```

---

## Self-review against the spec

- **Reader = thread, own handle, `_read_batch_jobs` unchanged, blocks on full queue** → Task 1 (`_run`, `queue_depth`).
- **GPU worker: no `rasterio.open`, `q.get(timeout)`, `None` after every reader, re-raise on exception** → Task 2 + `ReaderPool.get`.
- **Coordinator untouched except failure handling; result queue and assembly as before** → Task 3 keeps `write_tile` semantics, adds timeout/liveness.
- **CPU-only = same worker, `R = 1`** → Task 4 (rule) + Task 6 (`gpu_ids=[None]`).
- **Queue depth = `producer_queue_batches`** → Task 2 reads it from `config_dict`.
- **Order-independence proven, not assumed** → Task 3 `test_assembly_is_order_independent`.
- **Sizing `clamp((hw_cpu−1)//n_gpu, 1, 16)`, CPU-only 1, env override** → Task 4.
- **Failure semantics: sentinel per reader, exception object, `is_alive()` on timeout, no partial output** → Task 1 + Task 3 (`terminate()` on failure, `finalize_raster` never reached).
- **Gate (ii): bit-identical raster + GPKG, throughput ≥, RSS ≤, on T14 GPU, CPU-only, and carrot** → Task 7.
- **`Prediction.py` producer path replaced, not kept** → Task 8, gated.
- **Out of scope honoured**: no cross-GPU balancing, no `shared_memory`, sharding unchanged.

Type consistency: `ReaderPool.get()` returns `(batch_jobs, tiles, masks, stats)` — consumed exactly so in Tasks 2 and 5. `_drain_results` signature identical in Task 3's implementation and tests. `_reader_threads(hw_cpu, n_gpu, cpu_only)` identical in Task 4's implementation, tests and the three call sites.
