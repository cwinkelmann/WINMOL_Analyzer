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
    # queue slot + one in flight per reader
    assert len(calls) <= 1 + 2
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
    pool._threads[0].start()
    pool._threads[0].join()
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
    # consecutive batches, not quarters
    assert first_batch_ids == [0, 2, 4, 6]
    assert [len(sl) for sl in pool._slices] == [3, 3, 3, 3]


def test_close_unblocks_readers_parked_in_queue():
    """close() must unblock readers parked in put() on a full queue.
    A slow read_fn with depth=1 means readers block after first batch.
    Consumer takes one item then close()s. This must return in <1s and
    leave no threads alive."""
    calls = []

    def slow_read(src, batch_jobs):
        calls.append(1)
        time.sleep(0.1)  # Slow enough readers block on put()
        return _ok_read(src, batch_jobs)

    pool = ReaderPool('x.tif', _batches(20), slow_read,
                      n_readers=3, queue_depth=1, open_fn=_fake_open)
    pool.start()
    pool.get(timeout=5)  # Take first batch, leave readers blocked
    start = time.time()
    pool.close()
    elapsed = time.time() - start
    assert elapsed < 1.0, f"close() took {elapsed}s, should be <1s"
    assert not any(t.is_alive() for t in pool._threads)


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

    jobs = [
        {'r': r, 'c': c}
        for r in range(0, 64, 16)
        for c in range(0, 64, 16)
    ]
    batches = [jobs[i:i + 4] for i in range(0, 16, 4)]
    pool = ReaderPool(str(path), batches, read_fn, n_readers=4,
                      queue_depth=2)
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


def test_prediction_worker_reports_error_instead_of_dying(
        monkeypatch, tmp_path):
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
