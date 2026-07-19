"""The plugin must say so when it could not fetch its models.

download_models() is deliberately tolerant — a hosting gap must not brick the
plugin — but the result was computed and then dropped: `missing_models` was
returned by installer and referenced nowhere else, so a user whose download
failed got "WINMOL environment installed." and an empty model list, with no
indication anything had gone wrong.
"""
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
