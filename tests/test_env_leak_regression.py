"""Regression tests for the QGIS environment leak.

The plugin runs inside QGIS's Python but spawns a DIFFERENT one. Until these
tests existed, every spawn site inherited QGIS's PYTHONHOME/PYTHONPATH, which
on Windows killed environment creation with

    Could not import runpy module

and on macOS/Linux with "No module named 'encodings'". The symptom differs by
platform only because of stdlib layout; the bug is identical. These tests
poison os.environ the way QGIS does and assert the production helpers still
work, so the class of bug is caught on any OS.
"""
import subprocess
import sys

import pytest

from plugin_utils import installer
from plugin_utils.childenv import child_env

REAL_VERSION = sys.version_info[:2]


@pytest.fixture
def qgis_style_pollution(monkeypatch, tmp_path):
    """PYTHONHOME pointing at a directory that is not this interpreter's home.

    This is what QGIS exports; for any interpreter but its own it is fatal.
    """
    fake_home = tmp_path / "qgis_python"
    (fake_home / "lib").mkdir(parents=True)
    monkeypatch.setenv("PYTHONHOME", str(fake_home))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "qgis_site_packages"))
    return fake_home


def test_the_pollution_really_does_break_a_child(qgis_style_pollution):
    """Guard against a vacuous suite: if this passes, the tests below prove
    nothing. A child spawned with the inherited environment MUST fail."""
    proc = subprocess.run([sys.executable, "-c", "import sys"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0, (
        "the poisoned environment no longer breaks a child process, so these "
        "regression tests would pass vacuously")


def test_child_env_neutralises_the_pollution(qgis_style_pollution):
    proc = subprocess.run([sys.executable, "-c", "import sys"],
                          capture_output=True, text=True, timeout=60,
                          env=child_env())
    assert proc.returncode == 0, (
        f"child_env() did not rescue the child: {proc.stderr[-400:]}")


def test_python_version_probe_survives_qgis_pollution(qgis_style_pollution):
    """installer._python_version must report the real version, not (0, 0).

    Returning (0, 0) is what makes is_ready() consider a perfectly good venv
    unusable, which sends the plugin into a rebuild loop.
    """
    assert installer._python_version(sys.executable) == REAL_VERSION


def test_is_ready_does_not_reject_a_good_venv_under_pollution(
        qgis_style_pollution, monkeypatch, tmp_path):
    """The rebuild-loop guard: a venv that IS ready must still look ready when
    QGIS's environment is leaking."""
    venv_path = tmp_path / "venv"
    venv_path.mkdir()
    monkeypatch.setattr(installer, "get_venv_python_path",
                        lambda _p: sys.executable)
    monkeypatch.setattr(installer, "MIN_PY", REAL_VERSION)
    monkeypatch.setattr(installer, "MAX_PY", REAL_VERSION)
    monkeypatch.setattr(installer, "_requirements_hash", lambda: "deadbeef")
    with open(installer._marker_path(str(venv_path)), "w") as fh:
        fh.write('{"req_hash": "deadbeef"}')

    assert installer.is_ready(str(venv_path)) is True


# --- the leak class extended to PATH (Windows DLL shadowing) ----------------
#
# The original leak was PYTHONHOME/PYTHONPATH/GDAL_DATA. The same failure of
# nerve — inheriting the parent's environment wholesale — also leaked QGIS's
# program directory on PATH into the child, and on Windows PATH is the DLL
# search path: an onnxruntime native extension then bound QGIS 3.28's 2022-era
# MSVC/Qt runtime and died with "DLL initialization routine failed". child_env
# must strip QGIS/OSGeo directories from PATH on Windows and leave it alone
# everywhere else.

_WIN_QGIS_PATH = (
    r"C:\Program Files\QGIS 3.28\bin;"
    r"C:\Program Files\QGIS 3.28\apps\qgis\bin;"
    r"C:\OSGeo4W\bin;"
    r"C:\Windows\System32;C:\Windows"
)


def test_child_env_strips_qgis_from_path_on_windows(monkeypatch):
    monkeypatch.setattr("plugin_utils.childenv.sys.platform", "win32")
    monkeypatch.setenv("PATH", _WIN_QGIS_PATH)
    monkeypatch.setenv("OSGEO4W_ROOT", r"C:\OSGeo4W")
    monkeypatch.setenv("QGIS_PREFIX_PATH",
                       r"C:\Program Files\QGIS 3.28\apps\qgis")
    monkeypatch.setenv("GDAL_DATA",
                       r"C:\Program Files\QGIS 3.28\apps\gdal\share\gdal")
    entries = child_env()["PATH"].split(";")
    assert r"C:\OSGeo4W\bin" not in entries
    assert r"C:\Program Files\QGIS 3.28\bin" not in entries
    assert r"C:\Program Files\QGIS 3.28\apps\qgis\bin" not in entries
    # System32 must survive — the child needs the core Windows runtime.
    assert r"C:\Windows\System32" in entries
    assert r"C:\Windows" in entries


def test_child_env_does_not_touch_path_off_windows(monkeypatch):
    """No-op on this macOS/Linux host: PATH comes back verbatim."""
    assert sys.platform != "win32"
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/opt/qgis/bin")
    monkeypatch.setenv("OSGEO4W_ROOT", "/opt/osgeo4w")
    assert child_env()["PATH"] == "/usr/bin:/bin:/opt/qgis/bin"
