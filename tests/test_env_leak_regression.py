"""Live-child regression tests for the QGIS-environment-leak fix.

QGIS exports PYTHONHOME/PYTHONPATH describing ITS OWN interpreter.
Handing those to a child that runs a DIFFERENT interpreter (or even the
SAME interpreter, via a stale/foreign value) breaks it outright -- see
plugin_utils/childenv.py's module docstring. test_child_env.py already
checks that child_env()'s output dict has the poisoned keys stripped;
this file goes one step further and actually SPAWNS the child, so a
future change that stops sanitizing (or sanitizes the wrong variable)
fails on a real process instead of merely looking correct in isolation.

POSIX only: PYTHONHOME on Windows needs a different (path-shaped) poison
value than the bogus directory used here, and that side is already
covered by the Windows PATH-sanitizer tests in test_child_env.py.
"""
import os
import subprocess
import sys

import pytest

from plugin_utils import installer
from plugin_utils.childenv import child_env

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-specific pollution values; see test_child_env.py for "
           "Windows PATH-sanitizer coverage")


@pytest.fixture
def polluted(monkeypatch, tmp_path):
    """os.environ as QGIS leaves it: PYTHONHOME/PYTHONPATH pointing at a
    directory that is not sys.executable's own stdlib -- poison for any
    interpreter, per childenv.py's docstring."""
    bogus = str(tmp_path / "winmol-bogus-stdlib")
    os.mkdir(bogus)
    monkeypatch.setenv("PYTHONHOME", bogus)
    monkeypatch.setenv("PYTHONPATH", os.path.join(bogus, "lib"))
    return bogus


def test_anti_vacuity_polluted_child_actually_fails(polluted):
    """Proves the pollution used below is real poison, not a no-op.

    Without this, (b) and (c) passing would prove nothing -- they could
    just as well be passing because the pollution never mattered.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import sys"],
        env=dict(os.environ), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, (
        "expected the polluted child to fail to start; it did not, so "
        "this fixture is not valid poison and the tests below are moot")


def test_child_env_recovers_a_polluted_interpreter(polluted):
    """The actual regression guard: the same pollution, run through
    child_env(), and the child now starts cleanly."""
    result = subprocess.run(
        [sys.executable, "-c", "import sys"],
        env=child_env(), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (
        f"child_env() failed to recover a polluted interpreter: "
        f"{result.stderr}")


def test_python_version_survives_a_polluted_os_environ(polluted):
    """installer._python_version() must keep working even when the
    CALLING process's os.environ is polluted -- it is what
    choose_base_python()/is_ready() poll repeatedly, and a version probe
    that silently returns (0, 0) under pollution is exactly the
    rebuild-loop bug this fix closes (setup would think no Python is
    ever ready and reinstall on every plugin load)."""
    major, minor = installer._python_version(sys.executable)
    assert (major, minor) == sys.version_info[:2]
