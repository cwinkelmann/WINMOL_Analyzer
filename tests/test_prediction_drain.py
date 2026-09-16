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

from utils.PredictWorkers import (  # noqa: E402
    _drain_results, PredictionWorkerFailed)
from utils import PredictWorkers as PW  # noqa: E402


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
        import time
        time.sleep(0.5)
        q.put({'done': True, 'gpu_id': 0})
    threading.Thread(target=late, daemon=True).start()
    canvas = np.zeros((8, 8), np.uint8)
    stats = _drain_results(q, [_Worker()], _canvas_writer(canvas),
                           total_tiles=0, progress_interval_s=1e9,
                           timeout_s=0.2, print_fn=lambda *a, **k: None)
    assert stats['done'] == 0


class _FakeProcess:
    """Stands in for an `mp.Process`: never actually runs `target`, just
    records whether `terminate()` was called."""

    def __init__(self, target, args):
        self.target = target
        self.args = args
        self._alive = True
        self.terminate_called = False

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminate_called = True
        self._alive = False

    def join(self, timeout=None):
        pass


class _FakeCtx:
    def __init__(self, processes):
        self._processes = processes

    def Queue(self, maxsize=0):
        return queue.Queue(maxsize=maxsize)

    def Process(self, target, args):
        p = _FakeProcess(target, args)
        self._processes.append(p)
        return p


def test_any_exception_terminates_workers_not_just_worker_failed(
        monkeypatch, tmp_path):
    """F2: a `write_tile` error, a rasterio error, or any exception besides
    `PredictionWorkerFailed` must still terminate every live worker before
    propagating. Before the fix, `except PredictionWorkerFailed:` let a
    `RuntimeError` (standing in for a write_tile/rasterio failure) skip
    `terminate()` entirely -- the workers were left running, blocked in
    `results.put()` on a pipe nobody drains, and multiprocessing's atexit
    join waits forever."""
    import rasterio
    from rasterio.transform import from_origin

    from classes.Config import Config

    path = tmp_path / "in.tif"
    with rasterio.open(
            path, 'w', driver='GTiff', width=8, height=8, count=3,
            dtype='uint8', transform=from_origin(0, 8, 1, 1),
            crs='EPSG:3857') as d:
        for band in (1, 2, 3):
            d.write(np.zeros((8, 8), np.uint8), band)

    processes = []
    monkeypatch.setattr(
        PW.mp, "get_context", lambda name: _FakeCtx(processes))
    monkeypatch.setattr(
        PW, "_drain_results",
        lambda *a, **k: (_ for _ in ())
        .throw(RuntimeError("write_tile boom")))

    with pytest.raises(RuntimeError, match="write_tile boom"):
        PW.run_multi_gpu_prediction(
            "m.onnx", str(path), str(tmp_path / "out.tif"),
            tile_jobs=[], gpu_ids=[0, 1], config=Config())

    assert len(processes) == 2
    assert all(p.terminate_called for p in processes)


def test_duplicate_gpu_ids_spawn_two_workers_with_disjoint_shards(
        monkeypatch, tmp_path):
    """gpu_ids=[0, 0] must start TWO workers on device 0, each with a
    non-overlapping half of the jobs -- the shard split keys on list
    position, not on the id value."""
    import rasterio
    from rasterio.transform import from_origin

    from classes.Config import Config

    path = tmp_path / "in.tif"
    with rasterio.open(
            path, 'w', driver='GTiff', width=8, height=8, count=3,
            dtype='uint8', transform=from_origin(0, 8, 1, 1),
            crs='EPSG:3857') as d:
        for band in (1, 2, 3):
            d.write(np.zeros((8, 8), np.uint8), band)

    jobs = [{'idx': i} for i in range(20)]

    processes = []
    monkeypatch.setattr(
        PW.mp, "get_context", lambda name: _FakeCtx(processes))
    monkeypatch.setattr(
        PW, "_drain_results",
        lambda *a, **k: {'read_s': 0.0, 'infer_s': 0.0, 'write_s': 0.0,
                         'done': 0})

    PW.run_multi_gpu_prediction(
        "m.onnx", str(path), str(tmp_path / "out.tif"),
        tile_jobs=jobs, gpu_ids=[0, 0], config=Config())

    assert [p.args[0] for p in processes] == [0, 0]
    shards = [p.args[3] for p in processes]
    assert len(shards[0]) + len(shards[1]) == len(jobs)
    assert not (set(map(id, shards[0])) & set(map(id, shards[1])))
