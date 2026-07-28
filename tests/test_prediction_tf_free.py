"""Proves the CLI prediction path is TensorFlow-free.

Four groups:
1. ``_resize_batch`` on imagery (order=3, bicubic-like) resizes NHWC batches
   and takes an identity fast path when already at the target size.
2. ``_resize_batch`` on masks (order=0, nearest) keeps values binary.
3. ``_predict_batch_adaptive`` halves the micro-batch on OOM-shaped
   ``MemoryError``/``RuntimeError`` and re-raises anything else.
4. ``utils.Prediction`` / ``utils.PredictWorkers`` import cleanly with
   TensorFlow poisoned out of ``sys.modules`` -- proving neither module
   needs it, even though this conda env has TensorFlow installed.
"""
import sys

import numpy as np
import pytest


# --- Group 1 & 2: _resize_batch -------------------------------------------

def test_resize_batch_imagery_resizes_to_target_shape():
    from utils.Prediction import _resize_batch

    batch = np.random.rand(2, 700, 700, 3).astype(np.float32)
    out = _resize_batch(batch, (512, 512), order=3)

    assert out.shape == (2, 512, 512, 3)
    assert out.dtype == np.float32


def test_resize_batch_identity_fast_path_returns_same_object():
    from utils.Prediction import _resize_batch

    batch = np.random.rand(2, 512, 512, 3).astype(np.float32)
    out = _resize_batch(batch, (512, 512), order=3)

    assert out is batch


def test_resize_batch_masks_nearest_stays_binary():
    from utils.Prediction import _resize_batch

    mask = np.zeros((2, 20, 20, 1), dtype=np.float32)
    mask[:, :10, :10, :] = 1.0
    out = _resize_batch(mask, (8, 8), order=0)

    assert out.shape == (2, 8, 8, 1)
    assert set(np.unique(out).tolist()).issubset({0.0, 1.0})


# --- Group 3: _predict_batch_adaptive OOM retry ----------------------------

def test_predict_batch_adaptive_halves_batch_on_memory_error(monkeypatch):
    from utils import Prediction as Pred

    calls = []

    def fake_core(raw_tiles, raw_masks, model, config):
        calls.append(len(raw_tiles))
        if len(raw_tiles) > 1:
            raise MemoryError("Unable to allocate 4.00 GiB for an array")
        return [f"core-for-{len(raw_tiles)}"]

    monkeypatch.setattr(Pred, "_predict_batch_core", fake_core)

    result, used = Pred._predict_batch_adaptive(
        ["t"] * 4, ["m"] * 4, object(), object(), 4)

    assert used == 1
    assert result == ["core-for-1"]
    assert calls == [4, 2, 1]


def test_predict_batch_adaptive_halves_batch_on_runtime_oom_message(
    monkeypatch,
):
    from utils import Prediction as Pred

    calls = []

    def fake_core(raw_tiles, raw_masks, model, config):
        calls.append(len(raw_tiles))
        if len(raw_tiles) > 1:
            raise RuntimeError("CUDA error: out of memory")
        return ["ok"]

    monkeypatch.setattr(Pred, "_predict_batch_core", fake_core)

    result, used = Pred._predict_batch_adaptive(
        ["t"] * 4, ["m"] * 4, object(), object(), 4)

    assert used == 1
    assert result == ["ok"]
    assert calls == [4, 2, 1]


def test_predict_batch_adaptive_reraises_non_oom_runtime_error(monkeypatch):
    from utils import Prediction as Pred

    def fake_core(raw_tiles, raw_masks, model, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(Pred, "_predict_batch_core", fake_core)

    with pytest.raises(RuntimeError, match="boom"):
        Pred._predict_batch_adaptive(
            ["t"] * 4, ["m"] * 4, object(), object(), 4)


def test_predict_batch_adaptive_reraises_oom_at_batch_size_one(monkeypatch):
    """Can't halve below 1 -- an OOM that persists at batch_size=1 must
    still propagate instead of looping forever."""
    from utils import Prediction as Pred

    def fake_core(raw_tiles, raw_masks, model, config):
        raise MemoryError("Unable to allocate")

    monkeypatch.setattr(Pred, "_predict_batch_core", fake_core)

    with pytest.raises(MemoryError):
        Pred._predict_batch_adaptive(
            ["t"], ["m"], object(), object(), 1)


# --- Group 4: TensorFlow-free import ---------------------------------------

def test_prediction_and_predictworkers_import_without_tensorflow():
    """Poison sys.modules["tensorflow"] the way test suites conventionally
    block an accidental import (`import tensorflow` raises ImportError
    immediately), force a fresh (re)import of both modules, and confirm
    neither needs TensorFlow -- even though the conda env running this
    test suite has TensorFlow installed."""
    names_to_clear = [
        name for name in sys.modules
        if name == "tensorflow" or name.startswith("tensorflow.")
        or name in ("utils.Prediction", "utils.PredictWorkers")
    ]
    saved = {name: sys.modules.pop(name) for name in names_to_clear}
    sys.modules["tensorflow"] = None
    try:
        import utils.Prediction  # noqa: F401
        import utils.PredictWorkers  # noqa: F401

        sys.modules.pop("tensorflow", None)
        assert "tensorflow" not in sys.modules
    finally:
        sys.modules.pop("utils.Prediction", None)
        sys.modules.pop("utils.PredictWorkers", None)
        sys.modules.update(saved)
