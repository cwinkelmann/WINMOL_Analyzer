"""_autotune_batch_size must tune ONCE and then reuse the cached result.

The user's report: "when an autotune never happened before it should be run".
Before this change it was all-or-nothing — never (the shipped default) or on
every single prediction (a measured ~62 s stall each time).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("rasterio")
pytest.importorskip("skimage")

from plugin_utils import autotune_cache as ac        # noqa: E402
from utils import Prediction as Pred                 # noqa: E402


class Cfg:
    img_width = 512
    img_height = 512
    n_channels = 3
    prediction_batch_max_gpu = 4
    prediction_batch_autotune = "auto"
    prediction_batch_autotune_patience = 99
    prediction_batch_autotune_repeats = 1
    prediction_batch_autotune_min_improve = 0.0
    prediction_batch_autotune_stop_on_oom = True
    hardware = None


class Model:
    def __init__(self, path):
        self.model_path = str(path)
        self.providers = ["CPUExecutionProvider"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    model_file = tmp_path / "m.onnx"
    model_file.write_bytes(b"0" * 64)
    cache = tmp_path / "autotune.json"
    monkeypatch.setenv(ac.ENV_CACHE_PATH, str(cache))
    monkeypatch.delenv(ac.ENV_MODE, raising=False)
    # These tests are about the CACHE, not about the memory ceiling: pin the
    # free-memory probe so the sweep is bounded by the configured range and
    # not by whatever the machine running the suite happens to have free.
    monkeypatch.setattr(
        Pred, "_free_memory_bytes",
        lambda config: (64 * float(1024 ** 3), "stub"))
    return {"model": Model(model_file), "cache": cache}


@pytest.fixture
def timings(monkeypatch):
    """Stub the actual inference timing; record how often it is called."""
    calls = []

    def _fake(sample_tiles, sample_masks, model, config, cand, repeats=1):
        calls.append(cand)
        # 3 is the optimum: monotonically better up to 3, worse after. The
        # steps are 0.5 s so they clear the real 0.2 s/tile improvement bar
        # (tests/test_autotune_stopping.py owns the stop rules; these tests
        # must exercise the shipped defaults, not a threshold of their own).
        per_tile = abs(cand - 3) * 0.5 + 0.1
        return cand, per_tile, False

    monkeypatch.setattr(Pred, "_time_batch_candidate", _fake)
    return calls


TILES = [object()] * 4
MASKS = [None] * 4


def _tune(model, config):
    return Pred._autotune_batch_size(TILES, MASKS, model, config, 2)


def test_auto_tunes_on_the_first_run_and_persists(env, timings):
    cfg = Cfg()
    assert _tune(env["model"], cfg) == 3
    assert timings                              # it really ran
    assert env["cache"].exists()


def test_auto_reuses_the_cache_on_the_second_run(env, timings):
    cfg = Cfg()
    assert _tune(env["model"], cfg) == 3
    first = len(timings)
    assert first > 0

    assert _tune(env["model"], cfg) == 3
    assert len(timings) == first, "the second run must not re-tune"


def test_force_retunes_and_refreshes_the_cache(env, timings, monkeypatch):
    cfg = Cfg()
    _tune(env["model"], cfg)
    first = len(timings)
    cfg.prediction_batch_autotune = True         # legacy True == force
    assert _tune(env["model"], cfg) == 3
    assert len(timings) > first


def test_off_never_tunes_and_never_writes(env, timings):
    cfg = Cfg()
    cfg.prediction_batch_autotune = False
    assert _tune(env["model"], cfg) == 2         # the initial batch
    assert timings == []
    assert not env["cache"].exists()


def test_env_var_forces_off(env, timings, monkeypatch):
    monkeypatch.setenv(ac.ENV_MODE, "off")
    assert _tune(env["model"], Cfg()) == 2
    assert timings == []


def test_a_changed_model_invalidates_the_cache(env, timings):
    cfg = Cfg()
    _tune(env["model"], cfg)
    first = len(timings)
    with open(env["model"].model_path, "ab") as handle:
        handle.write(b"changed")
    _tune(env["model"], cfg)
    assert len(timings) > first


def test_a_changed_provider_invalidates_the_cache(env, timings):
    cfg = Cfg()
    _tune(env["model"], cfg)
    first = len(timings)
    env["model"].providers = ["CoreMLExecutionProvider"]
    _tune(env["model"], cfg)
    assert len(timings) > first


def test_a_corrupt_cache_degrades_to_a_retune(env, timings):
    cfg = Cfg()
    _tune(env["model"], cfg)
    first = len(timings)
    env["cache"].write_text("{ truncated", encoding="utf-8")
    assert _tune(env["model"], cfg) == 3         # no exception
    assert len(timings) > first


def test_an_out_of_range_cached_batch_is_ignored(env, timings, capsys):
    cfg = Cfg()
    key = ac.cache_key(env["model"], cfg, None)
    ac.store(key, 999, path=str(env["cache"]))
    assert _tune(env["model"], cfg) == 3
    assert timings, "an out-of-range entry must trigger a re-tune"


def test_an_unwritable_cache_is_not_fatal(env, timings, monkeypatch):
    monkeypatch.setattr(ac, "store", lambda *a, **k: False)
    assert _tune(env["model"], Cfg()) == 3


def test_a_degenerate_sample_never_poisons_the_cache(env, timings):
    cfg = Cfg()
    assert Pred._autotune_batch_size(
        [object()], [None], env["model"], cfg, 2) == 2
    assert not env["cache"].exists()


def test_the_tuning_loop_emits_parsable_progress(env, timings, capsys):
    from plugin_utils.run_progress import RunProgress

    _tune(env["model"], Cfg())
    out = capsys.readouterr().out
    progress = RunProgress("Trees")
    ticks = [progress.feed(line) for line in out.splitlines()]
    assert any(t is not None for t in ticks), (
        "the one-time autotune must tick the progress bar, not look like a "
        "hang:\n" + out)
