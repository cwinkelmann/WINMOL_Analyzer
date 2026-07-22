"""The plugin side of the autotune cache.

The cache crosses a process boundary into a foreign interpreter, so the
path has to be handed over explicitly (``$WINMOL_AUTOTUNE_CACHE``) rather
than derived from a HOME the QGIS/Docker user may not own. These are
source/AST-level checks plus real calls into the QGIS-free helpers — no
live QGIS, as the suite requires.
"""

import ast
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_utils import autotune_cache, config_overrides, installer  # noqa: E402,E501
from plugin_utils.childenv import STRIPPED, child_env  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source(relpath):
    with open(os.path.join(REPO, relpath), encoding="utf-8") as handle:
        return handle.read()


# --- the handover -----------------------------------------------------------

def test_the_cache_var_is_not_stripped_from_the_child_environment():
    assert autotune_cache.ENV_CACHE_PATH not in STRIPPED


def test_child_env_passes_the_cache_path_through(tmp_path):
    target = str(tmp_path / "autotune.json")
    env = child_env({autotune_cache.ENV_CACHE_PATH: target})
    assert env[autotune_cache.ENV_CACHE_PATH] == target


def test_the_mode_override_survives_child_env(monkeypatch):
    monkeypatch.setenv(autotune_cache.ENV_MODE, "off")
    assert child_env()[autotune_cache.ENV_MODE] == "off"


def test_the_worker_forwards_env_extra_to_child_env():
    tree = ast.parse(_source("tasks_threads.py"))
    worker = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and n.name == "Worker")
    dumped = ast.dump(worker)
    assert "env_extra" in dumped, "Worker takes no extra child environment"
    assert "child_env" in dumped


def test_the_dialog_hands_the_managed_cache_path_to_the_worker():
    source = _source("winmol_analyzer_dialog.py")
    assert "WINMOL_AUTOTUNE_CACHE" in source
    assert "autotune_cache_location" in source


# --- location + lifecycle ---------------------------------------------------

def test_the_cache_lives_under_the_managed_root(tmp_path):
    path = installer.autotune_cache_location(str(tmp_path))
    assert path.startswith(installer.managed_root(str(tmp_path)))
    assert path.endswith(autotune_cache.CACHE_FILENAME)


def test_deleting_the_environment_removes_the_cache(tmp_path):
    plugin_dir = tmp_path / "plugin"
    (plugin_dir / "models").mkdir(parents=True)
    path = installer.autotune_cache_location(str(plugin_dir))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    autotune_cache.store("k", 4, path=path)
    assert os.path.exists(path)

    installer.remove_environment(str(plugin_dir), remove_venv=True)
    assert not os.path.exists(path), (
        "the cache describes a batch measured against the deleted "
        "environment's onnxruntime; it must go with it")


def test_a_dry_run_deletion_keeps_the_cache(tmp_path):
    plugin_dir = tmp_path / "plugin"
    (plugin_dir / "models").mkdir(parents=True)
    path = installer.autotune_cache_location(str(plugin_dir))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    autotune_cache.store("k", 4, path=path)
    installer.remove_environment(str(plugin_dir), remove_venv=True,
                                 dry_run=True)
    assert os.path.exists(path)


# --- the Setup tab action ---------------------------------------------------

def test_the_clear_action_is_wired_and_guarded():
    source = _source("winmol_analyzer_dialog.py")
    tree = ast.parse(source)
    slot = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and n.name == "_setup_clear_autotune"), None)
    assert slot is not None, "the Setup tab has no clear-cache slot"
    assert "autotune_clear_button" in source

    body = [s for s in slot.body
            if not (isinstance(s, ast.Expr)
                    and isinstance(s.value, ast.Constant))]
    assert isinstance(body[0], ast.If) and "_busy" in ast.dump(body[0].test)


def test_the_clear_button_exists_once_in_the_ui():
    ui = _source("winmol_analyzer_dialog_base.ui")
    assert ui.count('name="autotune_clear_button"') == 1


@pytest.mark.parametrize("path_only", [True])
def test_clearing_twice_is_not_an_error(tmp_path, path_only):
    path = str(tmp_path / "autotune.json")
    autotune_cache.store("k", 4, path=path)
    assert autotune_cache.clear(path) is True
    assert autotune_cache.clear(path) is False


# --- the manual batch-size pin (Detection tab) ------------------------------
#
# "It should be possible to set the batch size manually in the GUI." The
# Detection tab's other config widgets are dead ends -- self.config is never
# serialised to the child -- so the pin travels the one channel that works:
# Worker(env_extra=...) -> child_env() -> WINMOL_CONFIG_OVERRIDES_JSON.

def test_the_batch_spin_box_exists_once_in_the_ui():
    ui = _source("winmol_analyzer_dialog_base.ui")
    assert ui.count('name="batchsize_spinBox"') == 1
    assert 'name="batchsize_label"' in ui
    # 0 must read as "Auto", not as a batch of zero tiles.
    assert "<string>Auto</string>" in ui


def test_the_dialog_sends_the_pin_through_the_worker_environment():
    source = _source("winmol_analyzer_dialog.py")
    tree = ast.parse(source)
    slot = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and n.name == "_batch_override_env"), None)
    assert slot is not None, "the dialog cannot pin the batch size"
    assert "batchsize_spinBox" in ast.dump(slot)
    assert "env_extra.update(self._batch_override_env())" in source


def test_a_pin_becomes_a_config_override_for_the_child():
    env = config_overrides.batch_override_env(6)
    payload = json.loads(env[config_overrides.ENV_VAR])
    assert payload == {"prediction_batch_override": 6}
    assert config_overrides.ENV_VAR not in STRIPPED
    assert child_env(env)[config_overrides.ENV_VAR] == \
        env[config_overrides.ENV_VAR]


@pytest.mark.parametrize("value", [0, None, "", "auto", -1])
def test_auto_injects_nothing(value):
    assert config_overrides.batch_override_env(value) == {}


def test_an_existing_override_is_merged_not_clobbered():
    existing = json.dumps({"tile_inner_px": 2048})
    env = config_overrides.batch_override_env(4, existing)
    payload = json.loads(env[config_overrides.ENV_VAR])
    assert payload == {"tile_inner_px": 2048, "prediction_batch_override": 4}


def test_an_unparsable_existing_override_is_dropped():
    env = config_overrides.batch_override_env(4, "{ truncated")
    assert json.loads(env[config_overrides.ENV_VAR]) == {
        "prediction_batch_override": 4}


def test_the_pinned_key_is_a_real_config_attribute():
    from classes.Config import Config
    assert hasattr(Config(), "prediction_batch_override"), (
        "winmol_run skips unknown keys, so a typo would be silently ignored")
