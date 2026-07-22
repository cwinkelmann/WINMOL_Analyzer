"""Every autotune skip/stop line must name the cap that actually bound.

The report that motivated this, from an M2 Mac running the shipped rc3:

    Prediction micro-batch autotune: memory ceiling b20 (4.4 GB free per
    unified memory (psutil available), 60% budget, ~128 MB/tile); nothing to
    tune, using b2.
    "It looks like the autotune is effectively disabled"

The user was right about the symptom and the message was why they had to ask:
the only limit it printed (memory, b20) was the one that did NOT bind, while
the one that did -- Config.prediction_batch_max_coreml = 2 -- was never
mentioned. Skipping the sweep is CORRECT there (measured fp32
Spruce_Deadwood on an M2: b1 184.0, b2 171.6, b4 182.8 ms/image, flat within
~7 %); only the explanation was wrong.

These tests stub the timing: no model, no inference, no GPU, no network.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("rasterio")
pytest.importorskip("skimage")

from classes.Config import Config                   # noqa: E402
from plugin_utils import autotune_cache as ac       # noqa: E402
from utils import Prediction as Pred                # noqa: E402

GB = float(1024 ** 3)


class Hardware:
    def __init__(self, accelerator="coreml"):
        self.accelerator = accelerator
        self.gpu_count = 1 if accelerator in ("cuda", "coreml") else 0
        self.gpu_names = ["stub"] if self.gpu_count else []
        self.gpu_memory_gb = [8.0] if self.gpu_count else []


class Model:
    def __init__(self, path):
        self.model_path = str(path)
        self.providers = ["CPUExecutionProvider"]


@pytest.fixture
def model(tmp_path):
    (tmp_path / "m.onnx").write_bytes(b"0" * 64)
    return Model(tmp_path / "m.onnx")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = tmp_path / "autotune.json"
    monkeypatch.setenv(ac.ENV_CACHE_PATH, str(path))
    monkeypatch.delenv(ac.ENV_MODE, raising=False)
    return path


@pytest.fixture
def memory(monkeypatch):
    def _set(free_bytes, source="stub"):
        monkeypatch.setattr(
            Pred, "_free_memory_bytes", lambda cfg: (free_bytes, source))

    _set(64 * GB)
    return _set


@pytest.fixture
def timed(monkeypatch):
    """Time every candidate as a clean win, so nothing else stops the sweep."""
    calls = []

    def _fake(tiles, masks, model, config, cand, repeats=1):
        calls.append(cand)
        return cand, 4.0 / cand, False

    monkeypatch.setattr(Pred, "_time_batch_candidate", _fake)
    return calls


def _config(accelerator="coreml", **kw):
    cfg = Config()
    cfg.hardware = Hardware(accelerator)
    # The stop rules are not what these tests are about: relax them so the
    # sweep ends on a CAP and the cap is what gets reported.
    cfg.prediction_batch_autotune_patience = 99
    cfg.prediction_batch_autotune_min_improve_s = 0.0
    cfg.prediction_batch_autotune_degrade_factor = 1.0
    cfg.prediction_batch_autotune_runaway_factor = 0.0
    for key, value in kw.items():
        setattr(cfg, key, value)
    return cfg


def _tune(model, cfg, initial, samples=64):
    return Pred._autotune_batch_size(
        [object()] * samples, [None] * samples, model, cfg, initial)


# --- the user's exact scenario ----------------------------------------------

def test_the_coreml_cap_is_named_when_it_skips_the_sweep(
    cache, model, memory, timed, capsys,
):
    """coreml, initial 2, cap 2, memory ceiling ~20: the CoreML cap bound."""
    memory(4.4 * GB, "unified memory (psutil available)")
    cfg = _config("coreml")
    assert cfg.prediction_batch_max_coreml == 2, "the cap is what binds here"

    assert _tune(model, cfg, 2, samples=182) == 2
    assert timed == [], "measurement says a sweep cannot help on CoreML"

    out = capsys.readouterr().out
    assert "CoreML cap b2" in out, (
        "the binding constraint must be named, not the one that did not "
        "bind:\n" + out)
    assert "prediction_batch_max_coreml" in out, "say where the cap lives"
    assert "not faster on this accelerator" in out, (
        "read as a measured decision, not a silent no-op")
    # The memory ceiling is context now, not the headline.
    assert "memory would allow b2" in out or "memory would allow b" in out
    assert "Nothing to tune, using b2." in out
    head, _, tail = out.partition("bound by")
    assert "memory would allow" in tail, "memory comes after the binding cap"


def test_the_coreml_cap_line_is_one_line(cache, model, memory, timed, capsys):
    memory(4.4 * GB, "unified memory (psutil available)")
    _tune(model, _config("coreml"), 2, samples=182)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 1, "one line per skipped sweep:\n" + "\n".join(lines)


# --- the other caps, same rule ----------------------------------------------

def test_the_memory_ceiling_is_named_when_it_bounds_the_sweep(
    cache, model, memory, timed, capsys,
):
    # 1.5 GB x 60 % / 128 MB per tile -> b7.
    memory(1.5 * GB, "nvidia-smi memory.free")
    assert _tune(model, _config("cuda"), 4) == 7
    out = capsys.readouterr().out
    assert "bound by the memory ceiling b7" in out
    assert "memory ceiling b7 reached" in out, "and again when the sweep ends"
    assert "the configured cap allows b16" in out, "the cap that did not bind"


def test_the_candidate_cap_is_named_when_it_ends_the_sweep(
    cache, model, memory, timed, capsys,
):
    cfg = _config("cuda")
    last = 4 + cfg.prediction_batch_autotune_max_candidates - 1
    assert _tune(model, cfg, 4) == last
    out = capsys.readouterr().out
    assert f"bound by the candidate cap b{last}" in out
    assert f"candidate cap b{last} reached" in out
    assert "prediction_batch_autotune_max_candidates" in out
    assert "memory would allow b" in out, "memory did not bind -- say so"


def test_the_user_pin_is_named_when_it_skips_the_sweep(
    cache, model, memory, timed, capsys,
):
    assert _tune(model, _config("cuda", prediction_batch_override=6), 4) == 6
    out = capsys.readouterr().out
    assert timed == [], "a pin is never measured"
    assert "bound by the user pin" in out
    assert "prediction_batch_override" in out
    assert "pinned to b6" in out


def test_the_sample_pool_is_named_when_it_bounds_the_sweep(
    cache, model, memory, timed, capsys,
):
    """Fewer sample tiles than the caps allow is its own reason."""
    assert _tune(model, _config("cpu"), 1, samples=3) == 3
    out = capsys.readouterr().out
    assert "bound by the sample pool b3" in out
    assert "sample pool b3 reached" in out


def test_the_configured_cap_is_named_when_it_ends_the_sweep(
    cache, model, memory, timed, capsys,
):
    cfg = _config("cuda", prediction_batch_max_gpu=6,
                  prediction_batch_autotune_max_candidates=99)
    assert _tune(model, cfg, 4) == 6
    out = capsys.readouterr().out
    assert "bound by the configured cap b6" in out
    assert "configured cap b6 reached" in out
    assert "prediction_batch_max_gpu" in out


# --- the stop rules say which rule stopped them -----------------------------

def test_each_stop_rule_names_itself(cache, model, memory, monkeypatch):
    """patience, degradation and runaway must be distinguishable in the log."""
    rendered = []

    def _capture(reason):
        rendered.append(reason)

    curve = {4: 0.34, 5: 0.337, 6: 0.33, 7: 0.471, 8: 0.561, 9: 1.135}
    monkeypatch.setattr(
        Pred, "_time_batch_candidate",
        lambda t, m, mo, c, cand, repeats=1: (cand, curve[cand], False))

    candidates = [4, 5, 6, 7, 8, 9]
    for kw, expected in (
        ({"prediction_batch_autotune_patience": 2,
          "prediction_batch_autotune_degrade_factor": 1.0,
          "prediction_batch_autotune_runaway_factor": 0.0},
         "patience rule"),
        ({"prediction_batch_autotune_patience": 99,
          "prediction_batch_autotune_min_improve_s": 0.0,
          "prediction_batch_autotune_degrade_factor": 1.25,
          "prediction_batch_autotune_runaway_factor": 0.0},
         "degradation guard"),
        ({"prediction_batch_autotune_patience": 99,
          "prediction_batch_autotune_min_improve_s": 0.0,
          "prediction_batch_autotune_degrade_factor": 1.0,
          "prediction_batch_autotune_runaway_factor": 1.5},
         "runaway guard"),
    ):
        cfg = _config("cuda", **kw)
        rules = Pred._autotune_rules(cfg)
        _, _, _, reason = Pred._sweep_batch_candidates(
            [object()] * 16, [None] * 16, model, cfg, candidates, rules,
            "Prediction micro-batch")
        _capture(reason)
        assert reason is not None, f"{expected} must stop this curve"
        assert expected in reason, f"expected {expected!r} in {reason!r}"

    assert len(set(rendered)) == 3, "each rule must read differently"


def test_an_oom_fallback_names_the_oom_guard(cache, model, memory):
    cfg = _config("cuda")
    rules = Pred._autotune_rules(cfg)

    def _oom(tiles, masks, model_, config, cand, repeats=1):
        return (cand, 0.2, False) if cand < 6 else (cand // 2, 0.9, True)

    Pred_time = Pred._time_batch_candidate
    try:
        Pred._time_batch_candidate = _oom
        _, _, _, reason = Pred._sweep_batch_candidates(
            [object()] * 16, [None] * 16, model, cfg, [4, 5, 6, 7], rules,
            "Prediction micro-batch")
    finally:
        Pred._time_batch_candidate = Pred_time

    assert reason is not None and "OOM guard" in reason, reason


# --- the search direction ---------------------------------------------------

def test_the_search_is_upward_only_from_the_planners_batch():
    """A KNOWN limitation, documented rather than fixed -- see the module
    docstring of utils/Prediction.py for why (CoreML recompiles per batch
    size, and every stop rule is defined as 'larger is more expensive').

    prediction_batch_override is the escape hatch for a smaller batch.
    """
    cfg = _config("cuda")
    candidates = Pred._prediction_batch_candidates(cfg, 4, ceiling=99)
    assert min(candidates) == 4, (
        "the planner's batch is a floor: an optimum below it is unreachable")
    assert candidates == sorted(candidates)


def test_the_upward_only_behaviour_is_documented():
    """If the behaviour is not fixed it has to be findable."""
    import utils.Prediction as module

    doc = (module.__doc__ or "").lower()
    assert "upward-only" in doc or "upward only" in doc
    assert "prediction_batch_override" in doc, "name the escape hatch"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "docs", "CONFIG.md"), encoding="utf-8") as fh:
        config_doc = fh.read().lower()
    assert "upward" in config_doc, "docs/CONFIG.md must say it too"


def test_the_sweep_announces_its_direction(cache, model, memory, timed,
                                           capsys):
    _tune(model, _config("cuda"), 4)
    assert "upward from b4 only" in capsys.readouterr().out


# --- the progress bar must keep parsing what we print -----------------------

def test_the_new_lines_still_drive_the_progress_bar(
    cache, model, memory, timed, capsys,
):
    """plugin_utils/run_progress.py parses 'autotune candidate i/n'.

    A mismatch freezes the QGIS bar with no other symptom, so assert both
    that the candidate lines still tick and that none of the new
    binding-constraint lines are mistaken for one.
    """
    from plugin_utils.run_progress import RunProgress

    _tune(model, _config("cuda"), 4)
    out = capsys.readouterr().out

    progress = RunProgress("Trees")
    ticks = [progress.feed(line) for line in out.splitlines()]
    assert any(t is not None for t in ticks), (
        "the autotune must tick the progress bar:\n" + out)

    for line in out.splitlines():
        if "autotune candidate " not in line:
            assert RunProgress("Trees").feed(line) is None, (
                f"a non-candidate line must not move the bar: {line}")
