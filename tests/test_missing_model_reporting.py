"""The plugin must say so when it could not fetch its models.

download_models() is deliberately tolerant — a hosting gap must not brick the
plugin — but the result was computed and then dropped: `missing_models` was
returned by installer and referenced nowhere else, so a user whose download
failed got "WINMOL environment installed." and an empty model list, with no
indication anything had gone wrong.
"""
import os

import pytest

from plugin_utils import installer


@pytest.fixture
def fresh_setup(monkeypatch):
    """Force resolve_environment down the 'build a new environment' path."""
    monkeypatch.setattr(installer, "configured_python_executable", lambda: None)
    monkeypatch.setattr(installer, "is_ready", lambda _p: False)
    monkeypatch.setattr(installer, "_confirm_setup", lambda: True)


def _stub_setup(missing):
    def _setup(*_a, **_k):
        return {"python": "/tmp/venv/bin/python", "missing_models": missing}
    return _setup


def test_message_names_the_models_that_could_not_be_downloaded(
        fresh_setup, monkeypatch, tmp_path):
    monkeypatch.setattr(installer, "setup_environment",
                        _stub_setup(["General", "Beech"]))
    result = installer.resolve_environment(str(tmp_path), prompt=True,
                                           build=True)
    assert result["status"] == "installed"
    assert result["missing_models"] == ["General", "Beech"]
    assert "General" in result["message"], result["message"]
    assert "Beech" in result["message"], result["message"]


def test_message_tells_the_user_what_to_do_about_it(
        fresh_setup, monkeypatch, tmp_path):
    monkeypatch.setattr(installer, "setup_environment",
                        _stub_setup(["General"]))
    msg = installer.resolve_environment(str(tmp_path), prompt=True,
                                        build=True)["message"].lower()
    # Must point at the escape hatch: selecting a local .onnx by hand.
    assert "browse" in msg or "select" in msg or ".onnx" in msg, msg


def test_no_warning_when_every_model_downloaded(
        fresh_setup, monkeypatch, tmp_path):
    monkeypatch.setattr(installer, "setup_environment", _stub_setup([]))
    result = installer.resolve_environment(str(tmp_path), prompt=True,
                                           build=True)
    assert result["status"] == "installed"
    assert "could not" not in result["message"].lower(), result["message"]


# --- the schema-v2 hazard ---------------------------------------------------

SHIPPED_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config.json")

#: The top-level keys of the schema-v2 registry. The v1 loop iterated
#: config.json's keys directly, so every one of these was reported as a
#: model that could not be downloaded — a fabricated failure at the end
#: of the very first environment build.
V2_TOP_LEVEL_KEYS = ("_comment", "schema", "tile_px", "gui_default",
                     "preload", "families", "models", "recommended")


def test_download_models_never_reports_registry_keys_as_models(
        tmp_path, monkeypatch):
    def _no_network(*_a, **_k):
        raise AssertionError("startup must perform zero network I/O")
    monkeypatch.setattr(installer.urllib.request, "urlretrieve", _no_network)
    from plugin_utils import model_registry
    monkeypatch.setattr(model_registry, "_DEFAULT_FETCHER", _no_network)

    missing = installer.download_models(str(tmp_path),
                                        config_path=SHIPPED_CONFIG)
    assert missing == []
    assert not set(missing) & set(V2_TOP_LEVEL_KEYS)


def test_the_environment_build_does_not_fetch_models(monkeypatch):
    """The Setup tab owns every byte of model traffic: an environment
    build must not end with a model download failure the user can
    neither retry nor understand."""
    import ast
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tasks_threads.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "setup_environment"]
    assert calls, "EnvSetupWorker no longer calls setup_environment"
    for call in calls:
        flags = {kw.arg: kw.value for kw in call.keywords}
        assert "download" in flags, (
            "setup_environment must be called with download=False")
        assert flags["download"].value is False
