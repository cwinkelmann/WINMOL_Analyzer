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
