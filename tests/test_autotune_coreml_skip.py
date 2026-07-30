"""Autotune must not sweep batch sizes on CoreML.

CoreML recompiles the model for each batch shape and is fastest at
batch 1, so the sweep is pure overhead (measured ~21s on an M2). The
sweep must be skipped when the model's active accelerator is CoreML,
without timing a single candidate.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import Prediction as Pred  # noqa: E402


class _Cfg:
    prediction_batch_override = None
    prediction_batch_autotune = "auto"
    prediction_batch_max_gpu = 16
    prediction_batch_autotune_patience = 2
    prediction_batch_autotune_min_improve = 0.005
    prediction_batch_autotune_min_improve_s = 0.2
    prediction_batch_autotune_stop_on_oom = True
    prediction_batch_autotune_repeats = 2
    img_width = 512
    img_height = 512
    n_channels = 3


class _Model:
    def __init__(self, accelerator):
        self.accelerator = accelerator


def _samples(n=4):
    tiles = [np.zeros((512, 512, 3), np.float32) for _ in range(n)]
    masks = [np.ones((512, 512), bool) for _ in range(n)]
    return tiles, masks


class _SweepReached(Exception):
    pass


def test_coreml_skips_the_sweep_without_timing(monkeypatch):
    monkeypatch.delenv("WINMOL_BATCH_AUTOTUNE", raising=False)
    # If the sweep were reached it would call _memory_batch_ceiling first.
    monkeypatch.setattr(Pred, "_memory_batch_ceiling",
                        lambda *a, **k: (_ for _ in ()).throw(_SweepReached()))
    tiles, masks = _samples()
    batch = Pred._autotune_batch_size(tiles, masks, _Model("coreml"),
                                      _Cfg(), initial_batch=1)
    assert batch == 1                  # returned the base batch, no sweep


def test_non_coreml_enters_the_sweep(monkeypatch):
    # A cpu/cuda model must proceed PAST the accelerator guard into the
    # sweep (which begins with the memory-ceiling probe).
    monkeypatch.delenv("WINMOL_BATCH_AUTOTUNE", raising=False)
    monkeypatch.setattr(Pred, "_memory_batch_ceiling",
                        lambda *a, **k: (_ for _ in ()).throw(_SweepReached()))
    import pytest
    tiles, masks = _samples()
    with pytest.raises(_SweepReached):
        Pred._autotune_batch_size(tiles, masks, _Model("cpu"),
                                  _Cfg(), initial_batch=1)
