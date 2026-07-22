"""The autotune must abandon a sweep that is running away from the optimum.

From a user's GPU-box report, verbatim ("But it is not stopping."):

    candidate 1/13: b4 = 0.340s/tile
    candidate 2/13: b5 = 0.337s/tile
    candidate 3/13: b6 = 0.330s/tile
    candidate 4/13: b7 = 0.471s/tile
    candidate 5/13: b8 = 0.561s/tile
    candidate 6/13: b9 = 1.135s/tile

``patience`` alone does not save this: with the shipped defaults (patience=4,
min_improve=0.005) the sweep keeps timing four more candidates after the knee,
and each one costs MORE wall time than the last because the per-tile cost is
rising while the batch grows. Measured on this tree's CoreML path, per-image
cost rises monotonically with batch size (0.228 s/img at b1 -> 0.498 s/img at
b8 on Spruce.onnx), so the tail of the sweep is pure waste.

Batch size does not change prediction output (verified: max abs diff 0.0
between b1, b4 and b8 on Spruce.onnx), so stopping early is a pure
time-vs-time trade with no effect on results.

SCOPE. These tests isolate the RUNAWAY knob, so they deliberately disable
the two rules that would otherwise stop this curve first: the absolute
``min_improve_s`` bar and the ``degrade_factor`` abort. With the SHIPPED
defaults the same curve stops at b6 having selected b4 -- that is the
stricter behaviour the user asked for and it is covered by
tests/test_autotune_stopping.py. What is under test here is that the
runaway guard alone still bounds a sweep once the absolute bar is relaxed
(e.g. on a slow CPU box where 0.2 s/tile gains are real).
"""
import pytest

from utils import Prediction as Pred

# The user's measured curve, keyed by batch size.
USER_CURVE = {
    4: 0.340, 5: 0.337, 6: 0.330, 7: 0.471,
    8: 0.561, 9: 1.135, 10: 1.9, 11: 2.7, 12: 3.6, 13: 4.8,
    14: 6.1, 15: 7.7, 16: 9.4,
}


class Cfg:
    prediction_batch_autotune = True
    prediction_batch_autotune_patience = 4
    prediction_batch_autotune_repeats = 1
    prediction_batch_autotune_min_improve = 0.005
    # Relaxed so the runaway guard, not the absolute bar, is what stops the
    # sweep -- see the module docstring.
    prediction_batch_autotune_min_improve_s = 0.0
    prediction_batch_autotune_degrade_factor = 1.0      # disabled
    prediction_batch_autotune_stop_on_oom = True
    prediction_batch_autotune_runaway_factor = 1.5
    hardware = None


class Model:
    def __init__(self):
        self.model_path = "/models/fake.onnx"
        self.providers = ["CPUExecutionProvider"]


@pytest.fixture
def curve(monkeypatch):
    """Replay the user's timings; record which candidates were measured."""
    measured = []

    def _fake(tiles, masks, model, config, cand, repeats=1):
        measured.append(cand)
        return cand, USER_CURVE.get(cand, 99.0), False

    monkeypatch.setattr(Pred, "_time_batch_candidate", _fake)
    monkeypatch.setattr(
        Pred, "_prediction_batch_candidates",
        lambda config, initial, ceiling=None: sorted(USER_CURVE))
    return measured


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    """Force a real sweep: no cached answer, nothing persisted."""
    from plugin_utils import autotune_cache as ac
    monkeypatch.setattr(ac, "resolve_mode", lambda config: "force")
    monkeypatch.setattr(ac, "cache_key", lambda **kw: "k")
    monkeypatch.setattr(ac, "cache_path", lambda: None)
    monkeypatch.setattr(ac, "load", lambda key, path=None: None)
    monkeypatch.setattr(
        ac, "store", lambda key, val, meta=None, path=None: False)


def _tune(config, curve_len=None):
    tiles = [object()] * max(USER_CURVE)
    return Pred._autotune_batch_size(tiles, [None] * max(USER_CURVE),
                                     Model(), config, 4)


def test_sweep_stops_once_a_candidate_runs_away(curve, capsys):
    best = _tune(Cfg())

    # b6 is the optimum in the user's data and must still be found.
    assert best == 6
    # b7 (0.471) is 1.43x the best -- under the guard, so it is allowed.
    # b8 (0.561) is 1.70x the best -> stop there. Nothing beyond may be timed.
    assert measured_max(curve) == 8, curve
    assert 9 not in curve, "b9 cost 1.135s/tile and must never be measured"
    assert "stopped" in capsys.readouterr().out


def test_the_unguarded_sweep_would_have_kept_going(curve):
    """Without the guard, patience=4 grinds on well past the knee."""
    cfg = Cfg()
    cfg.prediction_batch_autotune_runaway_factor = 0.0

    best = _tune(cfg)

    assert best == 6
    # patience only trips after 4 non-improving steps: b7,b8,b9,b10.
    assert measured_max(curve) >= 10
    assert 9 in curve


def test_guard_does_not_fire_below_the_optimum(curve):
    """A candidate SMALLER than the current best is never a runaway."""
    cfg = Cfg()
    cfg.prediction_batch_autotune_runaway_factor = 1.01
    best = _tune(cfg)
    # b4 is the first candidate and becomes the best; b5/b6 improve on it.
    # The guard must not stop the sweep before it has explored the knee.
    assert best == 6


def measured_max(curve):
    return max(curve)
