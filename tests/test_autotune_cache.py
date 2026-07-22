"""The batch-size autotune must run once, persist, and reuse.

Covers plugin_utils/autotune_cache.py: mode resolution, key invalidation,
round-trip, and — the important half — that every corrupt/hostile cache
degrades to "no entry" instead of raising.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_utils import autotune_cache as ac  # noqa: E402


class FakeConfig:
    img_width = 512
    img_height = 512
    n_channels = 3
    prediction_batch_max_gpu = 16
    prediction_batch_autotune = "auto"


class FakeModel:
    def __init__(self, path, providers=("CPUExecutionProvider",)):
        self.model_path = str(path)
        self.providers = list(providers)


class FakeHardware:
    def __init__(self, gpu_names=(), gpu_memory_gb=()):
        self.gpu_names = list(gpu_names)
        self.gpu_memory_gb = list(gpu_memory_gb)


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "General.onnx"
    path.write_bytes(b"x" * 1024)
    return FakeModel(path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ac.ENV_MODE, raising=False)
    monkeypatch.delenv(ac.ENV_CACHE_PATH, raising=False)


# --- mode -------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (False, "off"), ("off", "off"), ("0", "off"), ("false", "off"),
    (True, "force"), ("force", "force"), ("1", "force"),
    ("auto", "auto"), ("AUTO", "auto"),
    ("nonsense", "auto"), (None, "off"),
])
def test_resolve_mode_from_config(value, expected):
    cfg = FakeConfig()
    cfg.prediction_batch_autotune = value
    assert ac.resolve_mode(cfg) == expected


def test_config_default_is_auto():
    from classes.Config import Config
    assert ac.resolve_mode(Config()) == "auto"


def test_env_overrides_config(monkeypatch):
    cfg = FakeConfig()
    cfg.prediction_batch_autotune = "auto"
    monkeypatch.setenv(ac.ENV_MODE, "off")
    assert ac.resolve_mode(cfg) == "off"
    monkeypatch.setenv(ac.ENV_MODE, "force")
    assert ac.resolve_mode(cfg) == "force"
    monkeypatch.setenv(ac.ENV_MODE, "   ")     # blank falls back to config
    assert ac.resolve_mode(cfg) == "auto"


# --- path -------------------------------------------------------------------

def test_cache_path_honours_the_env_override(monkeypatch, tmp_path):
    target = tmp_path / "managed" / "autotune.json"
    monkeypatch.setenv(ac.ENV_CACHE_PATH, str(target))
    assert ac.cache_path() == str(target)


def test_default_cache_path_is_per_user(monkeypatch):
    monkeypatch.delenv(ac.ENV_CACHE_PATH, raising=False)
    path = ac.cache_path()
    assert path.endswith(ac.CACHE_FILENAME)
    assert "winmol" in path


# --- key --------------------------------------------------------------------

def test_key_is_stable(model):
    cfg = FakeConfig()
    assert ac.cache_key(model, cfg) == ac.cache_key(model, cfg)


def test_key_changes_with_provider(model, tmp_path):
    cfg = FakeConfig()
    other = FakeModel(model.model_path, providers=("CoreMLExecutionProvider",))
    assert ac.cache_key(model, cfg) != ac.cache_key(other, cfg)


def test_key_changes_with_model_content(model, tmp_path):
    cfg = FakeConfig()
    before = ac.cache_key(model, cfg)
    with open(model.model_path, "ab") as handle:
        handle.write(b"more")
    assert ac.cache_key(model, cfg) != before


def test_key_changes_with_tile_size_and_max_batch(model):
    cfg = FakeConfig()
    before = ac.cache_key(model, cfg)
    cfg.img_width = 256
    assert ac.cache_key(model, cfg) != before
    cfg.img_width = 512
    cfg.prediction_batch_max_gpu = 32
    assert ac.cache_key(model, cfg) != before


def test_key_changes_with_hardware(model):
    cfg = FakeConfig()
    one = ac.cache_key(model, cfg, FakeHardware(["A100"], [40.0]))
    two = ac.cache_key(model, cfg, FakeHardware(["RTX 4090"], [24.0]))
    assert one != two


def test_key_survives_a_missing_model_file(tmp_path):
    model = FakeModel(tmp_path / "gone.onnx")
    assert isinstance(ac.cache_key(model, FakeConfig()), str)


# --- round trip -------------------------------------------------------------

def test_store_then_load(tmp_path):
    path = tmp_path / "sub" / "autotune.json"
    assert ac.store("k1", 5, meta={"per_tile_s": 0.159}, path=str(path))
    assert ac.load("k1", path=str(path)) == 5
    assert ac.load("other", path=str(path)) is None


def test_store_keeps_other_entries(tmp_path):
    path = str(tmp_path / "autotune.json")
    ac.store("a", 4, path=path)
    ac.store("b", 8, path=path)
    assert ac.load("a", path=path) == 4
    assert ac.load("b", path=path) == 8


def test_store_overwrites_the_same_key(tmp_path):
    path = str(tmp_path / "autotune.json")
    ac.store("a", 4, path=path)
    ac.store("a", 9, path=path)
    assert ac.load("a", path=path) == 9


def test_clear_removes_the_file(tmp_path):
    path = str(tmp_path / "autotune.json")
    ac.store("a", 4, path=path)
    assert ac.clear(path) is True
    assert ac.load("a", path=path) is None
    assert ac.clear(path) is False          # already gone: no raise


# --- degrade-safe -----------------------------------------------------------

@pytest.mark.parametrize("blob", [
    "",                                     # empty file
    "{",                                    # truncated JSON
    "null",
    "[]",
    "{}",                                   # no version / entries
    '{"version": 999, "entries": {"k": {"batch": 5}}}',
    '{"version": 1, "entries": []}',
    '{"version": 1, "entries": {"k": 5}}',           # entry not a dict
    '{"version": 1, "entries": {"k": {"batch": "5"}}}',
    '{"version": 1, "entries": {"k": {"batch": true}}}',
    '{"version": 1, "entries": {"k": {"batch": 0}}}',
    '{"version": 1, "entries": {"k": {"batch": 9999}}}',
    '{"version": 1, "entries": {"k": {"batch": -3}}}',
    '{"version": 1, "entries": {"k": {"batch": 1.5}}}',
    '{"version": 1, "entries": {"k": {}}}',
])
def test_corrupt_cache_degrades_to_no_entry(tmp_path, blob):
    path = tmp_path / "autotune.json"
    path.write_text(blob, encoding="utf-8")
    assert ac.load("k", path=str(path)) is None


def test_load_of_a_missing_file_is_none(tmp_path):
    assert ac.load("k", path=str(tmp_path / "nope.json")) is None


def test_load_of_a_directory_is_none(tmp_path):
    assert ac.load("k", path=str(tmp_path)) is None


def test_store_over_a_corrupt_file_repairs_it(tmp_path):
    path = tmp_path / "autotune.json"
    path.write_text("not json at all", encoding="utf-8")
    assert ac.store("k", 6, path=str(path))
    assert ac.load("k", path=str(path)) == 6


def test_store_rejects_a_nonsense_batch(tmp_path):
    path = str(tmp_path / "autotune.json")
    assert ac.store("k", 0, path=path) is False
    assert ac.store("k", -1, path=path) is False
    assert ac.store("k", 99999, path=path) is False
    assert ac.store("k", "five", path=path) is False
    assert not os.path.exists(path)


def test_store_failure_is_not_fatal(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    # mkdirs under a regular file must fail -> False, no exception.
    assert ac.store("k", 4, path=str(blocker / "deep" / "a.json")) is False


def test_store_leaves_no_temp_files_behind(tmp_path):
    path = str(tmp_path / "autotune.json")
    ac.store("k", 4, path=path)
    leftovers = [n for n in os.listdir(tmp_path) if n.startswith(".autotune-")]
    assert leftovers == []


def test_written_file_is_valid_json_with_a_version(tmp_path):
    path = tmp_path / "autotune.json"
    ac.store("k", 4, meta={"note": "x"}, path=str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == ac.SCHEMA_VERSION
    assert data["entries"]["k"]["batch"] == 4
    assert data["entries"]["k"]["meta"] == {"note": "x"}
