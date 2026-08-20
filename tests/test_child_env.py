"""Unit tests for the child-process environment sanitizer.

QGIS exports PYTHONHOME/PYTHONPATH pointing at its own interpreter, and
GDAL_DATA/PROJ_LIB pointing at its own geo data. Both are poison for the
child processes WINMOL spawns, which run a *different* Python with its
own vendored GDAL.
"""
import os
import sys

import pytest

from plugin_utils.childenv import (child_env, safe_child_cwd,
                                   sanitize_windows_path)


@pytest.fixture
def polluted(monkeypatch):
    """os.environ as QGIS leaves it."""
    monkeypatch.setenv("PYTHONHOME", r"C:\PROGRA~1\QGIS34~1.12\apps\qgis-ltr")
    monkeypatch.setenv("PYTHONPATH",
                       r"C:\PROGRA~1\QGIS34~1.12\apps\qgis-ltr\python")
    monkeypatch.setenv("PYTHONSTARTUP", "/tmp/startup.py")
    monkeypatch.setenv("PYTHONEXECUTABLE", "/qgis/python.exe")
    monkeypatch.setenv("__PYVENV_LAUNCHER__", "/qgis/python.exe")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/other/venv")
    monkeypatch.setenv("GDAL_DATA", "/qgis/share/gdal")
    monkeypatch.setenv("PROJ_LIB", "/qgis/share/proj")
    monkeypatch.setenv("PROJ_DATA", "/qgis/share/proj")


def test_strips_interpreter_vars_that_break_a_foreign_python(polluted):
    env = child_env()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP",
                "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__", "VIRTUAL_ENV"):
        assert var not in env, f"{var} leaked into the child environment"


def test_strips_geo_data_vars_that_break_a_vendored_gdal(polluted):
    env = child_env()
    for var in ("GDAL_DATA", "PROJ_LIB", "PROJ_DATA"):
        assert var not in env, f"{var} leaked into the child environment"


def test_preserves_vars_the_child_needs(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = child_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert env["WINMOL_ONNX_FORCE_CPU"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_pins_hash_seed_so_stem_counts_stay_deterministic(polluted):
    # Same guarantee winmol_run.py's re-exec guard provides (PR #4).
    assert child_env()["PYTHONHASHSEED"] == "0"


def test_disables_user_site_packages(polluted):
    assert child_env()["PYTHONNOUSERSITE"] == "1"


def test_extra_is_applied_after_stripping(polluted):
    # tests/test_plugin_compute_contract.py injects a PYTHONPATH
    # deliberately; an explicit extra must win over the strip list.
    env = child_env(extra={"PYTHONPATH": "/block/tf"})
    assert env["PYTHONPATH"] == "/block/tf"


def test_does_not_mutate_the_parent_environment(polluted):
    child_env()
    assert "PYTHONHOME" in os.environ, "child_env mutated os.environ"


# --- Windows DLL-shadowing: sanitizing the child's PATH -------------------
# QGIS's own MSVC/Qt runtime DLLs on PATH can shadow what an onnxruntime
# native extension binds (the QGIS-3.28 "DLL init routine failed" bug),
# since PATH doubles as the Windows DLL search path. These run on
# macOS/Linux with stubbed Windows-style inputs -- no Windows required.

_QGIS_PATH = ";".join([
    r"C:\Program Files\QGIS 3.28\bin",
    r"C:\Program Files\QGIS 3.28\apps\qgis\bin",
    r"C:\OSGeo4W\bin",
    r"C:\OSGeo4W\apps\Python39",
    r"C:\Windows\System32",
    r"C:\Windows",
    r"C:\Windows\System32\WindowsPowerShell\v1.0",
    (r"C:\Users\me\AppData\Roaming\QGIS\QGIS3\profiles\default\winmol"
     r"\winmol_venv\Scripts"),
])

_QGIS_MARKERS = {
    "OSGEO4W_ROOT": r"C:\OSGeo4W",
    "QGIS_PREFIX_PATH": r"C:\Program Files\QGIS 3.28\apps\qgis",
    "GDAL_DATA": r"C:\Program Files\QGIS 3.28\apps\gdal\share\gdal",
    "PROJ_LIB": r"C:\Program Files\QGIS 3.28\apps\proj\share",
}

_VENV_ROOT = (r"C:\Users\me\AppData\Roaming\QGIS\QGIS3\profiles\default"
              r"\winmol\winmol_venv")


def test_sanitizer_drops_qgis_and_osgeo_but_keeps_system_and_venv():
    out = sanitize_windows_path(_QGIS_PATH, _QGIS_MARKERS,
                                keep_roots=(_VENV_ROOT,))
    entries = out.split(";")
    low = out.lower()
    # QGIS/OSGeo directories -- the source of the shadowing DLLs -- gone.
    assert r"C:\Program Files\QGIS 3.28\bin" not in entries
    assert r"C:\Program Files\QGIS 3.28\apps\qgis\bin" not in entries
    assert r"C:\OSGeo4W\bin" not in entries
    assert r"C:\OSGeo4W\apps\Python39" not in entries
    assert "osgeo4w" not in low
    # System32 / %SystemRoot% survive -- dropping them breaks the child.
    assert r"C:\Windows\System32" in entries
    assert r"C:\Windows" in entries
    assert r"C:\Windows\System32\WindowsPowerShell\v1.0" in entries
    # The venv survives even though it sits under a \QGIS\ profile path.
    assert any(e.endswith(r"winmol_venv\Scripts") for e in entries)


def test_keep_roots_is_load_bearing_for_the_venv_under_a_qgis_path():
    """Without the keep_roots exemption the venv's own Scripts dir --
    under ...\\QGIS\\QGIS3\\... -- would be dropped by the heuristic."""
    without = sanitize_windows_path(_QGIS_PATH, _QGIS_MARKERS).split(";")
    assert not any(e.endswith(r"winmol_venv\Scripts") for e in without)


def test_sanitizer_is_a_noop_without_qgis_markers():
    plain = r"C:\Windows\System32;C:\Windows;C:\tools\bin"
    assert sanitize_windows_path(plain, {}) == plain


def test_sanitizer_drops_qgis_by_heuristic_without_any_marker():
    """A stray QGIS PATH entry is dropped with no marker naming its root."""
    path = r"C:\Program Files\QGIS 3.40\apps\qgis-ltr\bin;C:\Windows\System32"
    out = sanitize_windows_path(path, {})
    assert out == r"C:\Windows\System32"


def test_child_env_sanitizes_path_on_windows(monkeypatch):
    monkeypatch.setattr("plugin_utils.childenv.sys.platform", "win32")
    monkeypatch.setenv("PATH", _QGIS_PATH)
    for var, val in _QGIS_MARKERS.items():
        monkeypatch.setenv(var, val)
    path = child_env()["PATH"]
    assert "osgeo4w" not in path.lower()
    assert r"C:\Windows\System32" in path.split(";")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="off-Windows no-op; the win32 path is covered by "
           "test_child_env_sanitizes_path_on_windows")
def test_child_env_leaves_path_untouched_off_windows(monkeypatch):
    """No-op on macOS/Linux: the loader reads DYLD/LD paths, not PATH,
    and a POSIX PATH is never a Windows DLL search path."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/opt/qgis/bin")
    assert child_env()["PATH"] == "/usr/bin:/bin:/opt/qgis/bin"


# --- Windows DLL-shadowing: a neutral working directory --------------------
# The process cwd is also on the Windows DLL search order, so a child
# inheriting a QGIS cwd is a second shadowing vector.

def test_safe_child_cwd_uses_the_interpreters_venv_root(tmp_path):
    exe = tmp_path / "bin" / "python"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert safe_child_cwd(str(exe)) == str(tmp_path)


def test_safe_child_cwd_falls_back_to_a_temp_dir(tmp_path):
    import tempfile
    assert safe_child_cwd(None) == tempfile.gettempdir()
    assert safe_child_cwd("/definitely/not/a/python") == tempfile.gettempdir()
