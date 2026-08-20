"""Autotune safety: memory ceiling, absolute improvement bar, manual
override.

The sweep (``_autotune_batch_size``, ``utils/Prediction.py``) already has a
patience/relative-min_improve stop rule and an OOM-halving retry
(``_predict_batch_adaptive``). This file covers the additional safety set
ported (trimmed) from rr6:

1. ``_memory_batch_ceiling`` / the free-memory helpers -- a candidate batch
   is never even timed once it is past what FREE memory (host RAM, or free
   VRAM bounded by host RAM when CUDA is the active provider) can hold.
2. An absolute ``prediction_batch_autotune_min_improve_s`` bar in addition
   to the existing relative ``min_improve`` -- a candidate only counts as
   progress if it clears BOTH, which kills jitter-chasing.
3. A post-OOM working ceiling -- once a candidate OOMs, nothing at or above
   it is tried again in the same sweep.
4. ``Config.prediction_batch_override`` -- pins the batch verbatim and
   skips the sweep (and the memory-ceiling computation) entirely.

Everything here monkeypatches ``Pred._free_memory_bytes`` /
``Pred._time_batch_candidate`` (or the lower-level
``_free_gpu_memory_gb`` / ``_available_ram_bytes``) directly: no GPU, no
real model, no real timing.
"""
from classes.Config import Config
from utils import Prediction as Pred


def _fake_model(accelerator="cpu"):
    return type("FakeModel", (), {"accelerator": accelerator})()


# --- 1. memory ceiling applied BEFORE timing -------------------------------

def test_ceiling_applied_before_timing_caps_candidates(monkeypatch):
    config = Config()
    config.prediction_batch_max_gpu = 20
    # High patience so the sweep only stops by running out of candidates --
    # isolates the memory ceiling as the thing that bounds it.
    config.prediction_batch_autotune_patience = 100
    initial = 4

    per_tile = Pred._estimated_bytes_per_tile(config)
    fraction = config.prediction_batch_autotune_memory_fraction
    # Sized so (free * fraction) // per_tile == 6, i.e. below the
    # configured cap of 20.
    free_bytes = (per_tile * 6.5) / fraction
    monkeypatch.setattr(
        Pred, "_free_memory_bytes",
        lambda model, cfg: (free_bytes, "mocked-low-memory"))

    attempted = []

    def fake_time(sample_tiles, sample_masks, model, cfg, cand, repeats=1):
        attempted.append(cand)
        return cand, 1.0 / cand, False

    monkeypatch.setattr(Pred, "_time_batch_candidate", fake_time)

    result = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), _fake_model(), config, initial)

    # The sweep now starts at b1 (a machine that cannot fit the planner's
    # batch must be able to find the low end); the memory ceiling still
    # caps the TOP at 6, which is what this test is about.
    assert attempted == [1, 2, 3, 4, 5, 6]
    assert 7 not in attempted and 20 not in attempted
    # fake_time makes every bigger batch genuinely faster (1.0/cand), so the
    # sweep now settles on the ceiling instead of returning `initial`
    # unchanged -- the absolute 0.2 s/tile bar used to make every
    # candidate after the first look like jitter.
    assert result == 6


def test_memory_batch_ceiling_blind_headroom_when_memory_unknown(
    monkeypatch,
):
    config = Config()
    monkeypatch.setattr(
        Pred, "_free_memory_bytes", lambda m, c: (None, "psutil unavailable"))

    budget = Pred._memory_batch_ceiling(_fake_model(), config, 5)

    assert budget["blind"] is True
    assert budget["ceiling"] == 5 + Pred.AUTOTUNE_BLIND_HEADROOM


# --- 2. low free memory: no sweep, batch floored at 1 ----------------------

def test_low_memory_never_sweeps_batch_is_ceiling_floored_at_one(
    monkeypatch,
):
    config = Config()
    config.prediction_batch_max_gpu = 20
    initial = 8

    monkeypatch.setattr(
        Pred, "_free_memory_bytes",
        lambda model, cfg: (1.0, "mocked-near-zero-memory"))

    def fake_time(*args, **kwargs):
        raise AssertionError(
            "timing must not run when the memory ceiling is below initial")

    monkeypatch.setattr(Pred, "_time_batch_candidate", fake_time)

    result = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), _fake_model(), config, initial)

    assert result == 1


# --- 3. absolute improvement bar: jitter is not progress --------------------

def test_jitter_not_progress_absolute_bar_stops_via_patience(monkeypatch):
    config = Config()
    config.prediction_batch_max_gpu = 20
    config.prediction_batch_autotune_patience = 2
    config.prediction_batch_autotune_min_improve = 0.0  # relative bar off
    config.prediction_batch_autotune_min_improve_s = 0.2
    config.prediction_batch_autotune_stop_on_oom = False
    initial = 2

    # Generous memory so it never binds here.
    monkeypatch.setattr(
        Pred, "_free_memory_bytes",
        lambda model, cfg: (1024.0 ** 4, "mocked-plenty"))

    # Each step is ~0.001s faster than the last -- real, but far under the
    # noise floor, so jitter. b6 would be a genuine 0.999s win but must
    # never be reached: patience stops the sweep first.
    #
    # The floor is now RELATIVE (5% of the measured baseline, capped by
    # min_improve_s) rather than a flat 0.2s. At a ~1s baseline that is
    # ~0.05s, so these 0.001s steps are still correctly rejected -- the
    # property this test guards is unchanged.
    timings = {1: 1.001, 2: 1.000, 3: 0.999, 4: 0.998, 5: 0.997, 6: 0.001}
    attempted = []

    def fake_time(sample_tiles, sample_masks, model, cfg, cand, repeats=1):
        attempted.append(cand)
        return cand, timings[cand], False

    monkeypatch.setattr(Pred, "_time_batch_candidate", fake_time)

    result = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), _fake_model(), config, initial)

    assert attempted == [1, 2, 3]
    assert 5 not in attempted and 6 not in attempted
    assert result == 1


# --- post-OOM working ceiling ----------------------------------------------

def test_post_oom_working_ceiling_never_retries_at_or_above(monkeypatch):
    config = Config()
    config.prediction_batch_max_gpu = 20
    config.prediction_batch_autotune_stop_on_oom = False
    config.prediction_batch_autotune_patience = 100
    initial = 2

    monkeypatch.setattr(
        Pred, "_free_memory_bytes",
        lambda model, cfg: (1024.0 ** 4, "mocked-plenty"))

    attempted = []

    def fake_time(sample_tiles, sample_masks, model, cfg, cand, repeats=1):
        attempted.append(cand)
        if cand == 5:
            # OOM'd at b5, _predict_batch_adaptive fell back to b3.
            return 3, 0.4, True
        return cand, 1.0 / cand, False

    monkeypatch.setattr(Pred, "_time_batch_candidate", fake_time)

    Pred._autotune_batch_size(
        list(range(20)), list(range(20)), _fake_model(), config, initial)

    assert attempted == [1, 2, 3, 4, 5]
    assert 6 not in attempted and 7 not in attempted


# --- 4. manual override: pins the batch, skips everything ------------------

def test_override_pins_skips_sweep_entirely(monkeypatch):
    config = Config()
    config.prediction_batch_override = 4

    def fake_time(*args, **kwargs):
        raise AssertionError("timing must not run when override is set")

    def fake_free_memory(*args, **kwargs):
        raise AssertionError(
            "memory ceiling must not be computed when override is set")

    monkeypatch.setattr(Pred, "_time_batch_candidate", fake_time)
    monkeypatch.setattr(Pred, "_free_memory_bytes", fake_free_memory)

    result = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), _fake_model("cuda"), config,
        initial_batch=1,
    )

    assert result == 4


def test_batch_override_helper_parses_and_rejects_values():
    config = Config()
    assert Pred._batch_override(config) is None

    config.prediction_batch_override = 0
    assert Pred._batch_override(config) is None

    config.prediction_batch_override = -1
    assert Pred._batch_override(config) is None

    config.prediction_batch_override = 4
    assert Pred._batch_override(config) == 4

    config.prediction_batch_override = "6"
    assert Pred._batch_override(config) == 6

    config.prediction_batch_override = "not-a-number"
    assert Pred._batch_override(config) is None


def test_prediction_batch_override_default_and_setattr_wiring():
    """Config default is off, and the attribute exists so
    ``WINMOL_CONFIG_OVERRIDES_JSON`` (hasattr-gated setattr in
    ``winmol_run.py``) can pick it up without any extra plumbing."""
    config = Config()
    assert config.prediction_batch_override is None
    assert hasattr(config, "prediction_batch_override")

    setattr(config, "prediction_batch_override", 4)
    assert config.prediction_batch_override == 4


# --- free-memory helpers: CUDA bounded by min(free VRAM, host free) --------

def test_free_memory_bytes_cuda_bound_by_host_when_host_lower(monkeypatch):
    config = Config()
    monkeypatch.setattr(Pred, "_free_gpu_memory_gb", lambda: [8.0])
    monkeypatch.setattr(
        Pred, "_available_ram_bytes", lambda: 2.0 * Pred._GB)

    free, source = Pred._free_memory_bytes(_fake_model("cuda"), config)

    assert free == 2.0 * Pred._GB
    assert "RAM" in source


def test_free_memory_bytes_cuda_uses_vram_when_host_is_higher(monkeypatch):
    config = Config()
    monkeypatch.setattr(Pred, "_free_gpu_memory_gb", lambda: [4.0])
    monkeypatch.setattr(
        Pred, "_available_ram_bytes", lambda: 64.0 * Pred._GB)

    free, source = Pred._free_memory_bytes(_fake_model("cuda"), config)

    assert free == 4.0 * Pred._GB
    assert "nvidia-smi" in source


def test_free_memory_bytes_cuda_falls_back_when_nvidia_smi_empty(
    monkeypatch,
):
    config = Config()
    monkeypatch.setattr(Pred, "_free_gpu_memory_gb", lambda: [])

    free, source = Pred._free_memory_bytes(_fake_model("cuda"), config)

    assert free is None
    assert "nvidia-smi" in source


def test_free_memory_bytes_cpu_uses_host_ram(monkeypatch):
    config = Config()
    monkeypatch.setattr(
        Pred, "_available_ram_bytes", lambda: 16.0 * Pred._GB)

    free, source = Pred._free_memory_bytes(_fake_model("cpu"), config)

    assert free == 16.0 * Pred._GB
    assert "psutil" in source


def test_free_memory_bytes_cpu_unavailable_returns_none(monkeypatch):
    config = Config()
    monkeypatch.setattr(Pred, "_available_ram_bytes", lambda: None)

    free, source = Pred._free_memory_bytes(_fake_model("cpu"), config)

    assert free is None
    assert source == "psutil unavailable"
