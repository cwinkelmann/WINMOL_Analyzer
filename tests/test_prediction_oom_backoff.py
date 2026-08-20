"""The prediction OOM back-off must never lose a tile.

Regression cover for the crash in docs/BUGS.md ("Autotune still dies"):
the streaming loop called ``model.predict_on_batch`` directly, so the first
steady-state OOM aborted the run even though a back-off already existed --
and that back-off, when it did run, silently truncated the batch instead of
re-running the remainder.
"""

import numpy as np
import pytest

from utils import Prediction as Pred


class _OomAbove:
    """A model that raises an onnxruntime-style OOM above ``limit`` rows.

    ``calls`` records the batch size of every accepted inference so a test
    can assert HOW the work was split, not just that it finished.
    """

    def __init__(self, limit, exc=None):
        self.limit = limit
        self.calls = []
        self._exc = exc

    def predict_on_batch(self, tensor):
        n = len(tensor)
        if n > self.limit:
            if self._exc is not None:
                raise self._exc
            raise MemoryError(
                "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero "
                "status code returned while running Conv node. Status "
                "Message: bfc_arena.cc:358 Failed to allocate memory for "
                "requested buffer of size 402800896")
        self.calls.append(n)
        # One distinctive row per input tile: tile i -> all-i plane. This is
        # what makes a dropped or reordered tile visible.
        return np.stack([np.full((4, 4, 1), float(t[0, 0, 0]))
                         for t in tensor])


def _tiles(n):
    return np.stack([np.full((4, 4, 1), float(i)) for i in range(n)])


@pytest.mark.parametrize("limit,batch", [(4, 8), (1, 8), (3, 16), (2, 5)])
def test_every_tile_survives_the_backoff(limit, batch):
    """The full batch comes back, in order, however far it had to reduce."""
    tensor = _tiles(batch)
    model = _OomAbove(limit)

    pred, used = Pred._predict_tensor_adaptive(tensor, model, batch)

    assert len(pred) == batch, "a tile was dropped by the back-off"
    assert [p[0, 0, 0] for p in pred] == [float(i) for i in range(batch)]
    assert used <= limit
    assert max(model.calls) <= limit


def test_result_is_identical_to_an_unreduced_run():
    """Backing off changes only the batching, never the output."""
    tensor = _tiles(8)
    reference, _ = Pred._predict_tensor_adaptive(
        tensor, _OomAbove(99), 8)
    reduced, used = Pred._predict_tensor_adaptive(
        tensor, _OomAbove(2), 8)

    assert used < 8
    np.testing.assert_array_equal(reference, reduced)


def test_non_oom_errors_still_propagate():
    """A real bug must not be mistaken for memory pressure and retried."""
    model = _OomAbove(2, exc=RuntimeError("invalid input shape"))
    with pytest.raises(RuntimeError, match="invalid input shape"):
        Pred._predict_tensor_adaptive(_tiles(8), model, 8)


def test_oom_at_batch_one_propagates():
    """Nothing left to halve -- surface it rather than loop forever."""
    with pytest.raises(MemoryError):
        Pred._predict_tensor_adaptive(_tiles(1), _OomAbove(0), 1)


def test_cuda_arena_wording_counts_as_oom():
    """The BFC-arena message carries neither 'oom' nor 'out of memory'."""
    exc = RuntimeError(
        "Failed to allocate memory for requested buffer of size 402800896")
    model = _OomAbove(2, exc=exc)
    # Raised only above the limit, so a reduction below it succeeds.
    pred, used = Pred._predict_tensor_adaptive(_tiles(8), model, 8)
    assert len(pred) == 8
    assert used <= 2


def test_batch_adaptive_keeps_every_tile_too():
    """The autotune's wrapper had the same truncation bug."""
    class _Cfg:
        overlap_pred = 0
        img_width = 4
        stem_binary_threshold = 0.5

    raw = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(8)]
    masks = [np.ones((4, 4), dtype=bool) for _ in range(8)]

    captured = {}

    def _fake_prepare(tiles, tile_masks, config):
        n = len(tiles)
        captured.setdefault('sizes', []).append(n)
        return (np.stack([np.full((4, 4, 1), float(t[0, 0, 0]))
                          for t in tiles]),
                np.ones((n, 4, 4, 1)))

    Pred_prepare = Pred._prepare_inference_batch
    Pred._prepare_inference_batch = _fake_prepare
    try:
        cores, used = Pred._predict_batch_adaptive(
            raw, masks, _OomAbove(2), _Cfg(), 8)
    finally:
        Pred._prepare_inference_batch = Pred_prepare

    assert len(cores) == 8, "the autotune wrapper dropped tiles"
    assert used <= 2


def test_candidates_reach_the_low_end():
    """A machine that cannot fit the planner's batch must find 1 or 2."""
    class _Cfg:
        prediction_batch_max_gpu = 16

    assert Pred._prediction_batch_candidates(_Cfg(), 4)[0] == 1
