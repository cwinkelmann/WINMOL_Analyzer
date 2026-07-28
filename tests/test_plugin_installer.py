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
    assert inst.plugin_requirements_path().name == "cpu.txt"


def test_venv_python_path_platform():
    p = inst.get_venv_python_path("/x/v")
    if sys.platform == "win32":
        assert p.endswith(os.path.join("Scripts", "python.exe"))
    else:
        assert "/bin/python" in p


def test_venv_location_falls_back_inside_plugin_off_qgis(tmp_path):
    # Off-QGIS (no profile dir) the venv lives inside the plugin dir, which is
    # what the tests and headless runs assume.
    loc = inst.venv_location(str(tmp_path))
    assert loc == os.path.join(str(tmp_path), inst.WINMOL_VENV_NAME)


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


def test_resolve_environment_ignores_stale_managed_interpreter(
        tmp_path, monkeypatch):
    # The user deleted the managed venv folder by hand (Open folder + rm -rf)
    # but the QgsSettings key still points at <venv>/bin/python, which no
    # longer exists. That must NOT dead-end as a "Python 0.0" error: it is
    # WINMOL's own vanished interpreter, so ignore it and fall through so the
    # env can be rebuilt. Regression for docs/BUGS.md "Reinstalling the plugin
    # didn't work on the GPU machine" after a manual delete.
    venv = inst.venv_location(str(tmp_path))
    stale_exe = inst.get_venv_python_path(venv)   # inside the venv, absent
    assert not os.path.isfile(stale_exe)
    monkeypatch.setattr(inst, "configured_python_executable",
                        lambda: stale_exe)
    res = inst.resolve_environment(str(tmp_path), prompt=True, build=False)
    assert res["status"] == "needs_setup"
    assert res["python"] is None


def test_resolve_environment_still_errors_on_missing_external_byo(
        tmp_path, monkeypatch):
    # A genuine BYO interpreter the user chose that is now gone is NOT our
    # managed venv, so keep the clear error rather than silently swallowing it
    # -- the boundary that keeps the stale-managed fix from over-reaching.
    external = os.path.join(os.sep, "opt", "gone", "bin", "python")
    assert not inst.path_is_inside(external, inst.venv_location(str(tmp_path)))
    monkeypatch.setattr(inst, "configured_python_executable",
                        lambda: external)
    monkeypatch.setattr(inst, "_python_version", lambda e: (0, 0))
    res = inst.resolve_environment(str(tmp_path), prompt=True, build=False)
    assert res["status"] == "error"
    assert "0.0" in res["message"]
