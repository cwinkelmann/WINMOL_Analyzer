"""Verbosity control for runtime logging (utils/Log.py).

The merge stage used to print per-tile diagnostics (MERGE DISCOVERY /
MERGE INPUT / MERGE READ OK ...) unconditionally, flooding a normal run.
They are still available, but only at debug level.
"""
import os

import pytest

from classes.Config import Config
from utils import Log


@pytest.fixture(autouse=True)
def _restore_log_level():
    previous = Log.current_level()
    yield
    Log.set_level(previous)


def _clear_env(monkeypatch):
    for var in ("WINMOL_LOG_LEVEL", "WINMOL_VERBOSE"):
        monkeypatch.delenv(var, raising=False)
    Log._exported = None


def test_default_level_is_normal(monkeypatch):
    _clear_env(monkeypatch)
    Log.configure_from_config(Config())
    assert Log.current_level() == Log.NORMAL


def test_debug_is_silent_at_normal(monkeypatch, capsys):
    _clear_env(monkeypatch)
    Log.configure_from_config(Config())
    Log.debug("MERGE DISCOVERY | root /tmp | gpkg_files 3")
    assert capsys.readouterr().out == ""


def test_debug_is_visible_at_debug(monkeypatch, capsys):
    _clear_env(monkeypatch)
    cfg = Config()
    cfg.log_level = "debug"
    Log.configure_from_config(cfg)
    Log.debug("MERGE DISCOVERY | root /tmp | gpkg_files 3")
    assert "MERGE DISCOVERY" in capsys.readouterr().out


def test_info_visible_at_normal_silent_at_quiet(monkeypatch, capsys):
    _clear_env(monkeypatch)
    Log.configure_from_config(Config())
    Log.info("MERGE SUMMARY")
    assert "MERGE SUMMARY" in capsys.readouterr().out

    cfg = Config()
    cfg.log_level = "quiet"
    Log.configure_from_config(cfg)
    Log.info("MERGE SUMMARY")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("level", ["quiet", "normal", "debug"])
def test_warn_and_error_always_print(monkeypatch, capsys, level):
    _clear_env(monkeypatch)
    cfg = Config()
    cfg.log_level = level
    Log.configure_from_config(cfg)
    Log.warn("MERGE READ FAIL | broken.gpkg")
    Log.error("MERGE VERIFY FAIL | 0 features")
    out = capsys.readouterr().out
    assert "WARNING: MERGE READ FAIL | broken.gpkg" in out
    assert "ERROR: MERGE VERIFY FAIL | 0 features" in out


def test_env_overrides_config(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WINMOL_LOG_LEVEL", "debug")
    cfg = Config()
    cfg.log_level = "normal"
    Log.configure_from_config(cfg)
    assert Log.current_level() == Log.DEBUG


def test_winmol_verbose_env_implies_debug(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WINMOL_VERBOSE", "1")
    Log.configure_from_config(Config())
    assert Log.current_level() == Log.DEBUG


def test_legacy_vector_debug_implies_debug(monkeypatch):
    _clear_env(monkeypatch)
    cfg = Config()
    cfg.vector_debug = True
    Log.configure_from_config(cfg)
    assert Log.current_level() == Log.DEBUG


def test_unknown_level_falls_back_to_normal(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("WINMOL_LOG_LEVEL", "chatty")
    Log.configure_from_config(Config())
    assert Log.current_level() == Log.NORMAL


def test_configure_exports_env_for_worker_processes(monkeypatch):
    """Vector tiles run in a spawned pool: the level must travel by env."""
    _clear_env(monkeypatch)
    cfg = Config()
    cfg.log_level = "debug"
    Log.configure_from_config(cfg)
    assert os.environ["WINMOL_LOG_LEVEL"] == "debug"
