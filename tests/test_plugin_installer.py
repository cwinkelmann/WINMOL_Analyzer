"""Unit tests for plugin_utils/installer.py — the QGIS plugin environment
setup. Pure logic only (no QGIS/PyQt): the module is import-safe off-QGIS, so
these exercise the sentinel gate, requirements selection, base-python choice,
and the tolerant model download without a running QGIS.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import plugin_utils.installer as inst   # noqa: E402


def test_module_imports_without_qgis_or_pyqt():
    # nothing Qt/QGIS at import time
    assert not any(m.startswith(("qgis", "PyQt5", "PyQt6"))
                   for m in sys.modules
                   if m in ("qgis", "PyQt5", "PyQt6"))
    assert inst.plugin_requirements_path().name in ("plugin.txt", "base.txt")


def test_venv_python_path_platform():
    p = inst.get_venv_python_path("/x/v")
    if sys.platform == "win32":
        assert p.endswith(os.path.join("Scripts", "python.exe"))
    else:
        assert "/bin/python" in p


def test_sentinel_gate_roundtrip(tmp_path, monkeypatch):
    venv = tmp_path / "winmol_venv"
    # fake an existing venv python so is_ready checks the marker, not the exe
    pybin = venv / ("Scripts" if sys.platform == "win32" else "bin")
    pybin.mkdir(parents=True)
    exe = "python.exe" if sys.platform == "win32" else "python"
    (pybin / exe).write_text("")
    # the fake python can't report a version; pretend it's supported
    monkeypatch.setattr(inst, "_python_version", lambda p: (3, 11))

    assert inst.is_ready(str(venv)) is False      # no marker yet
    inst._write_marker(str(venv))
    assert inst.is_ready(str(venv)) is True        # marker matches reqs hash

    # a requirements change invalidates the marker
    monkeypatch.setattr(inst, "_requirements_hash", lambda: "different")
    assert inst.is_ready(str(venv)) is False


def test_download_models_is_tolerant(tmp_path):
    plugin_dir = tmp_path
    # one already-present file (skipped), one bad/placeholder URL (reported)
    (plugin_dir / "models").mkdir()
    (plugin_dir / "models" / "Have.onnx").write_bytes(b"x")
    cfg = plugin_dir / "config.json"
    cfg.write_text(json.dumps({
        "Have": "https://example.invalid/Have.onnx",
        "Missing": "https://winmol.invalid.nonexistent/Missing.onnx",
    }))
    missing = inst.download_models(str(plugin_dir), str(cfg))
    assert "Missing" in missing        # unreachable host -> reported not raised
    assert "Have" not in missing       # existing file left alone
    assert (plugin_dir / "models" / "Have.onnx").exists()


def test_resolve_environment_never_raises_and_reports(tmp_path, monkeypatch):
    # no BYO env, no venv, prompt declined -> a clean 'declined' status
    monkeypatch.setattr(inst, "configured_python_executable", lambda: None)
    monkeypatch.setattr(inst, "_confirm_setup", lambda: False)
    res = inst.resolve_environment(str(tmp_path), prompt=True)
    assert res["status"] == "declined"
    assert res["python"] is None
    assert "venv_path" in res


def test_resolve_environment_uses_byo_when_deps_present(tmp_path, monkeypatch):
    monkeypatch.setattr(inst, "configured_python_executable",
                        lambda: "/opt/env/bin/python")
    monkeypatch.setattr(inst, "_python_version", lambda e: (3, 11))
    monkeypatch.setattr(inst, "_has_compute_deps", lambda exe: True)
    res = inst.resolve_environment(str(tmp_path))
    assert res["status"] == "byo"
    assert res["python"] == "/opt/env/bin/python"


def test_resolve_environment_flags_byo_missing_deps(tmp_path, monkeypatch):
    monkeypatch.setattr(inst, "configured_python_executable",
                        lambda: "/opt/env/bin/python")
    monkeypatch.setattr(inst, "_python_version", lambda e: (3, 11))
    monkeypatch.setattr(inst, "_has_compute_deps", lambda exe: False)
    res = inst.resolve_environment(str(tmp_path))
    assert res["status"] == "error"
    assert "onnxruntime" in res["message"]


def test_resolve_environment_rejects_too_old_byo(tmp_path, monkeypatch):
    # a Python 3.9 interpreter (e.g. macOS /usr/bin/python3) is rejected:
    # the code uses PEP 604 unions and needs >= 3.10.
    monkeypatch.setattr(inst, "configured_python_executable",
                        lambda: "/usr/bin/python3")
    monkeypatch.setattr(inst, "_python_version", lambda e: (3, 9))
    res = inst.resolve_environment(str(tmp_path))
    assert res["status"] == "error"
    assert "3.9" in res["message"]
