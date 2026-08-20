"""Persistent autotune cache: tune once per (hardware, model, execution
provider, tile geometry), reuse forever after.

Two layers are covered:

1. ``plugin_utils.autotune_cache`` in isolation -- mode resolution, key
   derivation and its sensitivity to each of the dimensions it is built
   from, load/store/clear, and tolerance of a missing/corrupt/wrong-version
   file.
2. Its wiring into ``utils.Prediction._autotune_batch_size`` -- the
   precedence chain (override > $WINMOL_BATCH_AUTOTUNE=off > a cache hit
   in range > sweep, then persist) proven with a fake timing function that
   raises if it is ever called when it should not run.

Every test gets an isolated cache file via the autouse fixture in
tests/conftest.py (``WINMOL_AUTOTUNE_CACHE`` pointed at ``tmp_path``), so
none of this touches the real per-user cache or leaks a batch size between
tests.
"""
import json

from classes.Config import Config
from plugin_utils import autotune_cache
from utils import Prediction as Pred


def _fake_model(accelerator="cpu", model_path=None, providers=None):
    return type("FakeModel", (), {
        "accelerator": accelerator,
        "model_path": model_path,
        "providers": providers or ["CPUExecutionProvider"],
    })()


def _config(**overrides):
    config = Config()
    config.prediction_batch_max_gpu = overrides.pop(
        "prediction_batch_max_gpu", 20)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _plenty_memory(monkeypatch):
    monkeypatch.setattr(
        Pred, "_free_memory_bytes",
        lambda model, cfg: (1024.0 ** 4, "mocked-plenty"))


def _must_not_be_called(reason):
    def _fail(*args, **kwargs):
        raise AssertionError(reason)
    return _fail


def _counting_timer(monkeypatch, timings):
    """``timings``: ``{batch: per_tile_seconds}``, missing candidates fall
    back to ``1/batch``. Returns the list of attempted candidates."""
    attempted = []

    def fake_time(sample_tiles, sample_masks, model, cfg, cand, repeats=1):
        attempted.append(cand)
        return cand, timings.get(cand, 1.0 / cand), False

    monkeypatch.setattr(Pred, "_time_batch_candidate", fake_time)
    return attempted


# --- mode resolution ---------------------------------------------------

def test_resolve_mode_env_wins_over_config(monkeypatch):
    monkeypatch.setenv("WINMOL_BATCH_AUTOTUNE", "off")
    config = _config(prediction_batch_autotune="force")
    assert autotune_cache.resolve_mode(config) == "off"


def test_resolve_mode_legacy_booleans(monkeypatch):
    monkeypatch.delenv("WINMOL_BATCH_AUTOTUNE", raising=False)
    assert autotune_cache.resolve_mode(
        _config(prediction_batch_autotune=True)) == "force"
    assert autotune_cache.resolve_mode(
        _config(prediction_batch_autotune=False)) == "off"


def test_resolve_mode_default_is_auto(monkeypatch):
    monkeypatch.delenv("WINMOL_BATCH_AUTOTUNE", raising=False)
    assert autotune_cache.resolve_mode(_config()) == "auto"
    assert autotune_cache.resolve_mode(None) == "auto"


def test_resolve_mode_unknown_string_falls_back_to_auto(monkeypatch):
    monkeypatch.delenv("WINMOL_BATCH_AUTOTUNE", raising=False)
    assert autotune_cache.resolve_mode(
        _config(prediction_batch_autotune="banana")) == "auto"


# --- cache key sensitivity -----------------------------------------------

def test_cache_key_stable_for_identical_inputs():
    model = _fake_model()
    config = _config()
    assert (autotune_cache.cache_key(model, config)
            == autotune_cache.cache_key(model, config))


def test_cache_key_changes_with_tile_geometry():
    model = _fake_model()
    key_a = autotune_cache.cache_key(
        model, _config(img_width=512, img_height=512))
    key_b = autotune_cache.cache_key(
        model, _config(img_width=256, img_height=256))
    assert key_a != key_b


def test_cache_key_changes_with_execution_provider():
    config = _config()
    key_a = autotune_cache.cache_key(
        _fake_model(providers=["CPUExecutionProvider"]), config)
    key_b = autotune_cache.cache_key(
        _fake_model(providers=["CUDAExecutionProvider"]), config)
    assert key_a != key_b


def test_cache_key_changes_with_model_identity(tmp_path):
    config = _config()
    model_a_path = tmp_path / "model_a.onnx"
    model_a_path.write_bytes(b"0" * 128)
    model_b_path = tmp_path / "model_b.onnx"
    model_b_path.write_bytes(b"0" * 256)          # different size

    key_a = autotune_cache.cache_key(
        _fake_model(model_path=str(model_a_path)), config)
    key_b = autotune_cache.cache_key(
        _fake_model(model_path=str(model_b_path)), config)
    assert key_a != key_b


def test_cache_key_changes_with_hardware():
    model = _fake_model()
    config = _config()
    hw_a = type("HW", (), {
        "gpu_names": ["RTX 3090"], "gpu_memory_gb": [24.0]})()
    hw_b = type("HW", (), {
        "gpu_names": ["RTX 4090"], "gpu_memory_gb": [24.0]})()
    assert (autotune_cache.cache_key(model, config, hw_a)
            != autotune_cache.cache_key(model, config, hw_b))


# --- load / store / clear -------------------------------------------------

def test_store_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "autotune.json")
    assert autotune_cache.store("k1", 6, meta={"note": "test"}, path=path)
    assert autotune_cache.load("k1", path=path) == 6


def test_load_missing_file_returns_none(tmp_path):
    path = str(tmp_path / "does-not-exist.json")
    assert autotune_cache.load("k1", path=path) is None


def test_load_corrupt_json_ignored_not_fatal(tmp_path):
    path = tmp_path / "autotune.json"
    path.write_text("{ this is not json")
    assert autotune_cache.load("k1", path=str(path)) is None


def test_load_wrong_schema_version_ignored(tmp_path):
    path = tmp_path / "autotune.json"
    path.write_text(json.dumps({
        "version": autotune_cache.SCHEMA_VERSION - 1,
        "entries": {"k1": {"batch": 6}},
    }))
    assert autotune_cache.load("k1", path=str(path)) is None


def test_load_rejects_batch_outside_max_sane_range(tmp_path):
    path = tmp_path / "autotune.json"
    path.write_text(json.dumps({
        "version": autotune_cache.SCHEMA_VERSION,
        "entries": {"k1": {"batch": autotune_cache.MAX_SANE_BATCH + 1}},
    }))
    assert autotune_cache.load("k1", path=str(path)) is None


def test_store_rejects_bool_zero_and_negative(tmp_path):
    path = str(tmp_path / "autotune.json")
    assert autotune_cache.store("k1", True, path=path) is False
    assert autotune_cache.store("k1", 0, path=path) is False
    assert autotune_cache.store("k1", -3, path=path) is False


def test_clear_removes_file_and_reports_true_once(tmp_path):
    path = str(tmp_path / "autotune.json")
    autotune_cache.store("k1", 4, path=path)
    assert autotune_cache.clear(path=path) is True
    assert not (tmp_path / "autotune.json").exists()
    assert autotune_cache.clear(path=path) is False    # nothing left


def test_cache_path_honors_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom" / "autotune.json"
    monkeypatch.setenv("WINMOL_AUTOTUNE_CACHE", str(target))
    assert autotune_cache.cache_path() == str(target)


# --- wiring into _autotune_batch_size: the precedence chain --------------
#
# override > $WINMOL_BATCH_AUTOTUNE=off > a cache hit in range > sweep,
# then persist. See utils/Prediction.py::_autotune_batch_size.

def test_cache_hit_skips_the_sweep(monkeypatch):
    config = _config()
    model = _fake_model()
    _plenty_memory(monkeypatch)
    key = autotune_cache.cache_key(model, config, None)
    autotune_cache.store(key, 5, path=autotune_cache.cache_path())

    monkeypatch.setattr(
        Pred, "_time_batch_candidate",
        _must_not_be_called("a cache hit must skip the sweep entirely"))

    result = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), model, config, initial_batch=2)

    assert result == 5


def test_cache_miss_sweeps_and_persists_then_next_call_hits(monkeypatch):
    config = _config()
    model = _fake_model()
    _plenty_memory(monkeypatch)
    timings = {2: 1.0, 3: 0.5, 4: 0.4, 5: 0.39, 6: 0.389}
    attempted = _counting_timer(monkeypatch, timings)

    first = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), model, config, initial_batch=2)
    assert attempted, "a cache miss must sweep"

    key = autotune_cache.cache_key(model, config, None)
    assert autotune_cache.load(
        key, path=autotune_cache.cache_path()) == first

    attempted.clear()
    monkeypatch.setattr(
        Pred, "_time_batch_candidate",
        _must_not_be_called("the persisted result must be reused"))
    second = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), model, config, initial_batch=2)

    assert second == first


def test_different_model_identity_misses_cache(monkeypatch, tmp_path):
    config = _config()
    model_a_path = tmp_path / "a.onnx"
    model_a_path.write_bytes(b"x" * 64)
    model_b_path = tmp_path / "b.onnx"
    model_b_path.write_bytes(b"x" * 128)
    model_a = _fake_model(model_path=str(model_a_path))
    model_b = _fake_model(model_path=str(model_b_path))
    _plenty_memory(monkeypatch)

    autotune_cache.store(
        autotune_cache.cache_key(model_a, config, None), 5,
        path=autotune_cache.cache_path())

    attempted = _counting_timer(monkeypatch, {})
    Pred._autotune_batch_size(
        list(range(20)), list(range(20)), model_b, config, initial_batch=2)
    assert attempted, "a different model must miss the cache and sweep"


def test_different_tile_geometry_misses_cache(monkeypatch):
    model = _fake_model()
    config_a = _config(img_width=512, img_height=512)
    config_b = _config(img_width=256, img_height=256)
    _plenty_memory(monkeypatch)

    autotune_cache.store(
        autotune_cache.cache_key(model, config_a, None), 5,
        path=autotune_cache.cache_path())

    attempted = _counting_timer(monkeypatch, {})
    Pred._autotune_batch_size(
        list(range(20)), list(range(20)), model, config_b, initial_batch=2)
    assert attempted, "different tile geometry must miss the cache and sweep"


def test_out_of_range_cached_batch_is_re_swept_not_clamped(monkeypatch):
    """A cached batch above what THIS run can reach (bounded here by the
    6-tile sample, standing in for a tighter memory ceiling) is
    re-measured rather than blindly clamped and returned: the clamped
    value was never actually timed under the current constraint."""
    config = _config(prediction_batch_max_gpu=20)
    model = _fake_model()
    _plenty_memory(monkeypatch)

    key = autotune_cache.cache_key(model, config, None)
    autotune_cache.store(key, 50, path=autotune_cache.cache_path())

    attempted = _counting_timer(monkeypatch, {2: 1.0, 3: 0.9, 4: 0.89})
    result = Pred._autotune_batch_size(
        list(range(6)), list(range(6)), model, config, initial_batch=2)

    assert attempted, "an out-of-range cached value must be re-measured"
    assert result != 50


def test_autotune_off_env_skips_sweep_and_cache_entirely(monkeypatch):
    monkeypatch.setenv("WINMOL_BATCH_AUTOTUNE", "off")
    config = _config()
    model = _fake_model()

    monkeypatch.setattr(
        Pred, "_free_memory_bytes",
        _must_not_be_called(
            "off must return before the memory ceiling is computed"))
    monkeypatch.setattr(
        Pred, "_time_batch_candidate",
        _must_not_be_called("off must never sweep"))
    monkeypatch.setattr(
        autotune_cache, "load",
        _must_not_be_called("off must never read the cache"))
    monkeypatch.setattr(
        autotune_cache, "store",
        _must_not_be_called("off must never write the cache"))

    result = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), model, config, initial_batch=3)

    assert result == 3


def test_legacy_false_also_skips_sweep_and_cache(monkeypatch):
    monkeypatch.delenv("WINMOL_BATCH_AUTOTUNE", raising=False)
    config = _config(prediction_batch_autotune=False)
    model = _fake_model()
    monkeypatch.setattr(
        Pred, "_time_batch_candidate",
        _must_not_be_called("False must never sweep"))
    monkeypatch.setattr(
        autotune_cache, "load",
        _must_not_be_called("False must never read the cache"))

    result = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), model, config, initial_batch=3)
    assert result == 3


def test_override_pin_never_touches_the_cache(monkeypatch):
    config = _config(prediction_batch_override=4)
    model = _fake_model()
    monkeypatch.setattr(
        autotune_cache, "cache_key",
        _must_not_be_called("an override must never consult the cache"))
    monkeypatch.setattr(
        Pred, "_time_batch_candidate",
        _must_not_be_called("an override must never sweep"))

    result = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), model, config, initial_batch=1)

    assert result == 4


def test_force_mode_ignores_cache_but_still_refreshes_it(monkeypatch):
    monkeypatch.setenv("WINMOL_BATCH_AUTOTUNE", "force")
    config = _config()
    model = _fake_model()
    _plenty_memory(monkeypatch)

    key = autotune_cache.cache_key(model, config, None)
    path = autotune_cache.cache_path()
    autotune_cache.store(key, 5, path=path)      # stale would-be cache hit

    attempted = _counting_timer(monkeypatch, {2: 1.0, 3: 0.5, 4: 0.49})
    result = Pred._autotune_batch_size(
        list(range(20)), list(range(20)), model, config, initial_batch=2)

    assert attempted, "force must re-sweep even with a cache entry present"
    assert autotune_cache.load(key, path=path) == result, (
        "force must refresh the cache with the freshly measured value")
