"""The batch-size autotune must be bounded, it must stop, and it must say
which limit stopped it.

Field report from an RTX 4080 box: the sweep timed b4..b9 and kept going
(b7 was 38 % worse than the best, b9 was 3.3x worse), and on a Linux box the
same march past the memory cliff froze the whole machine. Host RAM
exhaustion raises nothing -- the OOM fallback in ``_predict_batch_adaptive``
cannot help -- so the only defence is refusing to try a batch that does not
fit in the first place.

These tests stub ``_time_batch_candidate``: no model, no inference, no GPU.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("rasterio")
pytest.importorskip("skimage")

from classes.Config import Config                   # noqa: E402
from classes.HardwareInfo import HardwareInfo       # noqa: E402
from plugin_utils import autotune_cache as ac       # noqa: E402
from utils import Prediction as Pred                # noqa: E402

TILES = [object()] * 16
MASKS = [None] * 16

#: The user's measured RTX-4080 sweep, verbatim.
USER_TIMINGS = {
    4: 0.340, 5: 0.337, 6: 0.330, 7: 0.471, 8: 0.561,
    9: 1.135, 10: 1.400, 11: 1.700, 12: 2.000,
}

GB = float(1024 ** 3)


class Hardware:
    def __init__(self, accelerator="cuda"):
        self.accelerator = accelerator
        self.gpu_count = 1 if accelerator in ("cuda", "coreml") else 0
        self.gpu_names = ["stub"] if self.gpu_count else []
        self.gpu_memory_gb = [16.0] if self.gpu_count else []


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = tmp_path / "autotune.json"
    monkeypatch.setenv(ac.ENV_CACHE_PATH, str(path))
    monkeypatch.delenv(ac.ENV_MODE, raising=False)
    return path


@pytest.fixture
def model(tmp_path):
    class Model:
        def __init__(self):
            self.model_path = str(tmp_path / "m.onnx")
            self.providers = ["CPUExecutionProvider"]

    (tmp_path / "m.onnx").write_bytes(b"0" * 64)
    return Model()


@pytest.fixture
def memory(monkeypatch):
    """Control the free-memory probe; default is 'plenty'."""
    def _set(free_bytes, source="stub"):
        monkeypatch.setattr(
            Pred, "_free_memory_bytes", lambda cfg: (free_bytes, source))

    _set(64 * GB)
    return _set


class _Calls(list):
    """A list of measured candidates that can re-arm the timing stub."""


@pytest.fixture
def timed(monkeypatch):
    """Record every candidate that was actually measured."""
    calls = _Calls()

    def _install(table):
        def _fake(tiles, masks, model, config, cand, repeats=1):
            calls.append(cand)
            return cand, table[cand], False
        monkeypatch.setattr(Pred, "_time_batch_candidate", _fake)

    _install(USER_TIMINGS)
    calls.install = _install
    return calls


def _tune(model, config, initial=4):
    return Pred._autotune_batch_size(TILES, MASKS, model, config, initial)


def _config(**kw):
    cfg = Config()
    cfg.hardware = Hardware()
    for key, value in kw.items():
        setattr(cfg, key, value)
    return cfg


# --- the regression the user reported ---------------------------------------

def test_the_users_sweep_stops_after_three_candidates(
    cache, model, memory, timed, capsys,
):
    """b4=0.340, b5=0.337, b6=0.330 -> stop. Gains of 3 and 10 ms are noise.

    Before: it timed b4..b10 and selected b6. The 3 % throughput given up
    here is the price of a bounded sweep, and the user asked for it.
    """
    assert _tune(model, _config()) == 4
    assert timed == [4, 5, 6], (
        "the sweep must stop once two candidates in a row fail to gain "
        "0.2 s/tile")
    out = capsys.readouterr().out
    assert "non-improving step(s)" in out
    assert "0.20s/tile gain" in out
    # The sweep also has to say which way it searches: the planner's batch
    # is a floor, so an optimum below b4 is unreachable by design.
    assert "upward from b4 only" in out


def test_a_dramatically_worse_candidate_aborts_the_sweep(
    cache, model, memory, timed, capsys,
):
    timed.install({1: 0.10, 2: 0.40, 3: 0.40, 4: 0.40, 5: 0.40, 6: 0.40})
    assert _tune(model, _config(), initial=1) == 1
    assert timed == [1, 2], "b2 was 4x the best; nothing larger may be tried"
    assert "degradation guard" in capsys.readouterr().out


def test_real_gains_are_not_mistaken_for_noise(cache, model, memory, timed):
    """Every step gains > 0.2 s/tile, so the sweep must run to the end."""
    timed.install({1: 1.00, 2: 0.70, 3: 0.45, 4: 0.20, 5: 0.19, 6: 0.19})
    assert _tune(model, _config(), initial=1) == 4
    assert timed == [1, 2, 3, 4, 5, 6]


# --- the memory ceiling -----------------------------------------------------

def test_free_memory_caps_the_candidate_list(
    cache, model, memory, timed, capsys,
):
    # 1.5 GB free x 60 % budget / 128 MB per tile -> b7.
    memory(1.5 * GB)
    timed.install({c: 2.0 - 0.25 * c for c in range(4, 13)})
    assert _tune(model, _config()) == 7
    assert timed == [4, 5, 6, 7], "nothing above the memory ceiling may run"
    out = capsys.readouterr().out
    assert "memory ceiling b7" in out
    assert "memory ceiling b7 reached" in out
    # ...and it names the cap that did NOT bind as context, never instead.
    assert "the configured cap allows b16" in out


def test_unknown_memory_falls_back_to_a_tiny_sweep(
    cache, model, memory, timed, capsys,
):
    memory(None, "psutil unavailable")
    timed.install({c: 2.0 - 0.25 * c for c in range(4, 13)})
    assert _tune(model, _config()) == 6
    assert timed == [4, 5, 6], (
        "with memory unknown the sweep may only step "
        f"{Pred.AUTOTUNE_BLIND_HEADROOM} above the planner's batch")
    assert "free memory unknown" in capsys.readouterr().out


def test_a_cached_batch_that_no_longer_fits_is_rejected(
    cache, model, memory, timed, capsys,
):
    """The box that had 24 GB free last week may have 2 GB free today."""
    cfg = _config()
    ac.store(ac.cache_key(model, cfg, cfg.hardware), 12, path=str(cache))
    memory(None, "psutil unavailable")
    timed.install({c: 2.0 - 0.25 * c for c in range(4, 13)})

    assert _tune(model, cfg) <= 6
    assert timed, "an unusable cached batch must trigger a re-tune"
    assert "out-of-range cached batch 12" in capsys.readouterr().out


def test_the_ceiling_is_computed_before_anything_is_timed(
    cache, model, monkeypatch,
):
    """The whole safety argument: never measure a batch that may not fit."""
    order = []

    def _probe(cfg):
        order.append("probe")
        return 64 * GB, "stub"

    def _fake(tiles, masks, model_, config, cand, repeats=1):
        order.append("time")
        return cand, USER_TIMINGS[cand], False

    monkeypatch.setattr(Pred, "_free_memory_bytes", _probe)
    monkeypatch.setattr(Pred, "_time_batch_candidate", _fake)
    _tune(model, _config())
    assert order and order[0] == "probe"
    assert "time" in order


def test_a_low_memory_box_never_sweeps_at_all(cache, model, memory, timed):
    memory(0.5 * GB)                      # room for ~2 tiles at 60 %
    assert _tune(model, _config()) == 4
    assert timed == [], "a box this tight must not time anything"


@pytest.mark.parametrize("accelerator,expected_attr", [
    ("cpu", "prediction_batch_max_cpu"),
    ("cuda", "prediction_batch_max_gpu"),
])
def test_the_cpu_ceiling_is_not_the_gpu_ceiling(accelerator, expected_attr):
    cfg = _config()
    cfg.hardware = Hardware(accelerator)
    assert Pred._batch_ceiling_attr(cfg) == expected_attr
    # b1..b16 on a 4-core CPU box was 16 candidates x 6 forward passes
    # before the first tile was written.
    assert cfg.prediction_batch_max_cpu < cfg.prediction_batch_max_gpu


def test_the_candidate_list_is_capped_in_length():
    cfg = _config()
    candidates = Pred._prediction_batch_candidates(cfg, 1, ceiling=999)
    assert len(candidates) == cfg.prediction_batch_autotune_max_candidates


def test_the_free_probe_prefers_nvidia_smi_on_cuda(monkeypatch):
    monkeypatch.setattr(
        HardwareInfo, "free_gpu_memory_gb", staticmethod(lambda: [11.0, 8.0]))
    monkeypatch.setattr(Pred, "_available_ram_bytes", lambda: 64 * GB)
    free, source = Pred._free_memory_bytes(_config())
    assert free == pytest.approx(8.0 * GB), "the smallest device bounds it"
    assert "memory.free" in source


def test_host_ram_still_bounds_a_cuda_run(monkeypatch):
    """The frozen box had VRAM free and was allocating on the host.

    A run planned for CUDA whose onnxruntime session silently falls back to
    the CPU provider spends host RAM, so the ceiling has to respect both.
    """
    monkeypatch.setattr(
        HardwareInfo, "free_gpu_memory_gb", staticmethod(lambda: [16.0]))
    monkeypatch.setattr(Pred, "_available_ram_bytes", lambda: 2 * GB)
    free, source = Pred._free_memory_bytes(_config())
    assert free == pytest.approx(2 * GB)
    assert "psutil" in source


def test_the_free_probe_degrades_without_psutil(monkeypatch):
    cfg = _config()
    cfg.hardware = Hardware("cpu")
    monkeypatch.setattr(Pred, "_available_ram_bytes", lambda: None)
    free, reason = Pred._free_memory_bytes(cfg)
    assert free is None and "psutil" in reason


# --- the manual pin ---------------------------------------------------------

def test_a_pinned_batch_skips_the_sweep_entirely(
    cache, model, memory, timed, capsys,
):
    assert _tune(model, _config(prediction_batch_override=6)) == 6
    assert timed == [], "a pinned batch must never be measured"
    assert not cache.exists(), "a pinned batch must not poison the cache"
    out = capsys.readouterr().out
    assert "pinned to b6" in out
    assert "bound by the user pin" in out
    assert "prediction_batch_override" in out, "say where the pin lives"


@pytest.mark.parametrize("value", [None, 0, "", "auto", -3])
def test_a_missing_or_junk_pin_means_auto(value):
    assert Pred._batch_override(_config(prediction_batch_override=value)) \
        is None


RASTER = {
    "width": 8000, "height": 8000, "bands": 3, "dtype": "uint8",
    "pixel_size_x": 0.02, "pixel_size_y": 0.02, "estimated_input_gb": 0.6,
}


@pytest.mark.parametrize("accelerator", ["cpu", "cuda"])
def test_the_planner_honours_the_pin(accelerator):
    """prediction_batch_size is planner-owned; the pin has to survive it."""
    from classes.ExecutionPlan import build_execution_plan

    hardware = HardwareInfo(
        cpu_count=8,
        total_ram_gb=32.0,
        gpu_count=1 if accelerator == "cuda" else 0,
        gpu_names=["stub"] if accelerator == "cuda" else [],
        gpu_memory_gb=[16.0] if accelerator == "cuda" else [],
        accelerator=accelerator,
    )
    cfg = Config()
    default = build_execution_plan(cfg, hardware, RASTER, "Trees")
    cfg.prediction_batch_override = 3
    pinned = build_execution_plan(cfg, hardware, RASTER, "Trees")

    assert pinned.prediction_batch_size == 3
    assert default.prediction_batch_size != 3, (
        "pick a pin the planner would not have chosen anyway")


# --- the binding constraint must be the one that is named -------------------
#
# The user's report, from an M2 Mac running the shipped rc3:
#
#     Prediction micro-batch autotune: memory ceiling b20 (4.4 GB free per
#     unified memory (psutil available), 60% budget, ~128 MB/tile); nothing
#     to tune, using b2.
#     "It looks like the autotune is effectively disabled"
#
# They were right about the symptom, and the message is why they had to
# ask: the only limit it printed (memory, b20) was the one that did NOT
# bind, while the one that did -- Config.prediction_batch_max_coreml = 2 --
# was never mentioned. Skipping the sweep is CORRECT there (measured fp32
# Spruce_Deadwood on an M2: b1 184.0, b2 171.6, b4 182.8 ms/image, flat
# within ~7 %); only the explanation was wrong. The other constraints get
# the same treatment; this pins the one that was actually complained about.

def test_the_coreml_cap_is_named_when_it_skips_the_sweep(
    cache, model, memory, timed, capsys,
):
    """coreml, initial 2, cap 2, memory ceiling ~20: the CoreML cap bound."""
    memory(4.4 * GB, "unified memory (psutil available)")
    cfg = _config()
    cfg.hardware = Hardware("coreml")
    # The stop rules are not what this is about: relax them so the sweep
    # can only end on a CAP, and the cap is what has to get reported.
    cfg.prediction_batch_autotune_patience = 99
    cfg.prediction_batch_autotune_min_improve_s = 0.0
    cfg.prediction_batch_autotune_degrade_factor = 1.0
    cfg.prediction_batch_autotune_runaway_factor = 0.0
    assert cfg.prediction_batch_max_coreml == 2, "the cap is what binds here"

    assert Pred._autotune_batch_size(
        [object()] * 182, [None] * 182, model, cfg, 2) == 2
    assert timed == [], "measurement says a sweep cannot help on CoreML"

    out = capsys.readouterr().out
    assert "CoreML cap b2" in out, (
        "the binding constraint must be named, not the one that did not "
        "bind:\n" + out)
    assert "prediction_batch_max_coreml" in out, "say where the cap lives"
    assert "not faster on this accelerator" in out, (
        "read as a measured decision, not a silent no-op")
    # The memory ceiling is context now, not the headline.
    assert "memory would allow b" in out
    _, _, tail = out.partition("bound by")
    assert "memory would allow" in tail, "memory comes after the binding cap"
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1, "one line per skipped sweep:\n" + "\n".join(lines)


# --- every stop rule must be distinguishable in the log ---------------------

def test_each_stop_rule_names_itself(cache, model, memory, monkeypatch):
    """patience, degradation and runaway must read differently."""
    rendered = []
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
        cfg = _config(**kw)
        rules = Pred._autotune_rules(cfg)
        _, _, _, reason = Pred._sweep_batch_candidates(
            TILES, MASKS, model, cfg, candidates, rules,
            "Prediction micro-batch")
        assert reason is not None, f"{expected} must stop this curve"
        assert expected in reason, f"expected {expected!r} in {reason!r}"
        rendered.append(reason)

    assert len(set(rendered)) == 3, "each rule must read differently"


def test_an_oom_fallback_names_the_oom_guard(cache, model, memory,
                                             monkeypatch):
    cfg = _config()
    rules = Pred._autotune_rules(cfg)

    def _oom(tiles, masks, model_, config, cand, repeats=1):
        return (cand, 0.2, False) if cand < 6 else (cand // 2, 0.9, True)

    monkeypatch.setattr(Pred, "_time_batch_candidate", _oom)
    _, _, _, reason = Pred._sweep_batch_candidates(
        TILES, MASKS, model, cfg, [4, 5, 6, 7], rules,
        "Prediction micro-batch")
    assert reason is not None and "OOM guard" in reason, reason


# --- the search direction ---------------------------------------------------

def test_the_search_is_upward_only_from_the_planners_batch():
    """A KNOWN limitation, documented rather than fixed -- see the module
    docstring of utils/Prediction.py for why (CoreML recompiles per batch
    size, and every stop rule is defined as "larger is more expensive").

    prediction_batch_override is the escape hatch for a smaller batch.
    """
    candidates = Pred._prediction_batch_candidates(_config(), 4, ceiling=99)
    assert min(candidates) == 4, (
        "the planner's batch is a floor: an optimum below it is unreachable")
    assert candidates == sorted(candidates)


# --- the progress bar must keep parsing what we print -----------------------

def test_the_autotune_lines_still_drive_the_progress_bar(
    cache, model, memory, timed, capsys,
):
    """plugin_utils/run_progress.py parses "autotune candidate i/n".

    A mismatch freezes the QGIS bar with no other symptom, so assert both
    that the candidate lines still tick and that none of the
    binding-constraint lines is mistaken for one.
    """
    from plugin_utils.run_progress import RunProgress

    _tune(model, _config())
    out = capsys.readouterr().out

    progress = RunProgress("Trees")
    ticks = [progress.feed(line) for line in out.splitlines()]
    assert any(t is not None for t in ticks), (
        "the autotune must tick the progress bar:\n" + out)

    for line in out.splitlines():
        if "autotune candidate " not in line:
            assert RunProgress("Trees").feed(line) is None, (
                f"a non-candidate line must not move the bar: {line}")
