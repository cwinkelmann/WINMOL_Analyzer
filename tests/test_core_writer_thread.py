"""The prediction writer runs off the GPU thread -- without losing tiles.

Moving crop/binarise/write into `_CoreWriter` buys the GPU back the ~26%
of wall-clock it used to idle through (13.5 ms/tile against 10.0 ms of
`infer` on R13). The risk that buys is a background thread quietly
dropping a write, which would leave holes in the stem map with nothing in
the log to say so. These pin the two properties that prevent it: every
submitted tile is written, and any failure reaches the caller.
"""
import os
import queue
import sys
import threading

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class _FakeDst:
    """Records windowed writes the way rasterio would accept them."""

    def __init__(self, fail_on=None):
        self.writes = []
        self.fail_on = fail_on
        self.lock = threading.Lock()

    def write(self, arr, band, window=None):
        with self.lock:
            if self.fail_on is not None and len(self.writes) == self.fail_on:
                raise OSError("disk went away")
            self.writes.append((arr.shape, window.col_off, window.row_off))


class _Cfg:
    overlap_pred = 8
    img_width = 16
    img_height = 16
    stem_binary_threshold = 0.5


def _layout():
    return {'out_height': 4096, 'out_width': 4096}


def _batch(n, start=0):
    """n tiles of model output plus their jobs, laid out without overlap."""
    pred = np.zeros((n, _Cfg.img_width, _Cfg.img_width, 1), dtype=np.float32)
    pred[:] = 0.9                       # above threshold -> foreground
    mask = np.ones((n, _Cfg.img_width, _Cfg.img_width, 1), dtype=np.float32)
    items = [({'dst_row': (start + i) * 8, 'dst_col': (start + i) * 8},
              None, None) for i in range(n)]
    return pred, mask, items


def _writer(dst):
    from utils.Prediction import _CoreWriter
    return _CoreWriter(dst, _layout(), _Cfg())


def test_every_submitted_tile_is_written():
    dst = _FakeDst()
    w = _writer(dst)
    w.start()
    total = 0
    for b in range(5):
        pred, mask, items = _batch(3, start=b * 3)
        w.submit(pred, mask, items)
        total += len(items)
    w.close()
    assert w.written == total
    assert len(dst.writes) == total, "the writer dropped tiles on drain"


def test_a_failed_write_reaches_the_caller():
    """A dropped write must not be silent -- close() re-raises."""
    dst = _FakeDst(fail_on=4)
    w = _writer(dst)
    w.start()
    for b in range(4):
        pred, mask, items = _batch(3, start=b * 3)
        try:
            w.submit(pred, mask, items)
        except Exception:               # queue closed early; close() reports
            break
    with pytest.raises(OSError, match="disk went away"):
        w.close()


def test_queue_is_bounded_so_a_slow_writer_applies_backpressure():
    """Unbounded would let the backlog grow without limit on a slow disk."""
    dst = _FakeDst()
    w = _writer(dst)
    assert isinstance(w.queue, queue.Queue)
    assert w.queue.maxsize > 0, "writer queue must be bounded"


def test_out_of_order_writes_land_in_their_own_windows():
    """Ordering is irrelevant: each tile owns a disjoint window, which is
    why the raster stays bit-identical with the write moved off-thread."""
    dst = _FakeDst()
    w = _writer(dst)
    w.start()
    pred, mask, items = _batch(4)
    w.submit(pred, mask, items)
    w.close()
    offsets = sorted((c, r) for _, c, r in dst.writes)
    assert len(offsets) == 4
    assert len(set(offsets)) == 4, "windows must not collide"
