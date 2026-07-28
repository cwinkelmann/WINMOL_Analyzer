"""Environment setup for the WINMOL QGIS plugin.

Inference runs in a separate Python env (requirements/cpu.txt —
onnxruntime + geo stack): either a user-configured interpreter
(QgsSettings ``winmol/python_executable``) or a venv this module builds
once, blessed by a ``.winmol_ready`` sentinel keyed to the requirements
hash. Import-safe without QGIS/PyQt (lazy imports); every child process
runs with ``child_env()`` because QGIS exports PYTHONHOME/PYTHONPATH/
GDAL vars that break any foreign interpreter (see childenv.py).
"""
import collections
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .childenv import child_env, safe_child_cwd

WINMOL_VENV_NAME = "winmol_venv"
READY_MARKER = ".winmol_ready"
QSETTINGS_PYTHON_KEY = "winmol/python_executable"
CPU_REQUIREMENTS = "cpu.txt"

# WINMOL is validated on Python 3.11 only; MIN==MAX pins it exactly.
MIN_PY = (3, 11)
MAX_PY = (3, 11)

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAX_LINE = 300
_TAIL_LINES = 80


# --- paths ------------------------------------------------------------------

def repo_requirements_dir() -> Path:
    return Path(_PLUGIN_DIR, "requirements")


def plugin_requirements_path() -> Path:
    """The requirements file the compute environment is built from."""
    return repo_requirements_dir().joinpath(CPU_REQUIREMENTS)


def managed_root(plugin_dir) -> str:
    """Root for WINMOL's managed venv: under the QGIS profile dir, NOT
    the plugin dir — uninstall is rmtree(plugin_dir), and deleting a
    symlink-filled venv is what QGIS reports as "uninstall failed".
    Off-QGIS (no profile) falls back to ``plugin_dir``."""
    try:
        from qgis.core import QgsApplication
        base = QgsApplication.qgisSettingsDirPath()
        if base:
            return os.path.join(base, "winmol")
    except Exception:
        pass
    return plugin_dir


def venv_location(plugin_dir) -> str:
    """Absolute path of the managed venv (under managed_root)."""
    return os.path.join(managed_root(plugin_dir), WINMOL_VENV_NAME)


def get_venv_python_path(venv_path) -> str:
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    # venv always installs a 'python' symlink; prefer it.
    py = os.path.join(venv_path, "bin", "python")
    return py if os.path.exists(py) else os.path.join(
        venv_path, "bin", "python3")


# --- base interpreter selection ---------------------------------------------

def _python_version(executable) -> tuple:
    """(major, minor) of ``executable``, or (0, 0) when unusable.
    ``-I`` + ``child_env()``: QGIS's PYTHONHOME/PYTHONPATH would point
    the child at QGIS's stdlib and stop it starting at all."""
    try:
        out = subprocess.run(
            [executable, "-I", "-c",
             "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=30, env=child_env())
        if out.returncode == 0:
            major, minor = out.stdout.strip().split(".")
            return (int(major), int(minor))
    except Exception:
        pass
    return (0, 0)


def choose_base_python() -> str:
    """A system python (3.11) to build the venv from, or RuntimeError
    with an actionable message."""
    candidates = []
    for name in ("python3.11", "python3", "python"):
        exe = shutil.which(name)
        if exe and exe not in candidates:
            candidates.append(exe)
    for exe in candidates:
        if MIN_PY <= _python_version(exe) <= MAX_PY:
            return exe
    raise RuntimeError(
        "No suitable Python found to build the WINMOL environment "
        f"(need {MIN_PY[0]}.{MIN_PY[1]}). Install one, or set an "
        "existing interpreter in the plugin settings "
        f"({QSETTINGS_PYTHON_KEY}).")


def _has_compute_deps(executable) -> bool:
    try:
        out = subprocess.run(
            [executable, "-I", "-c",
             "import onnxruntime, rasterio, geopandas"],
            capture_output=True, timeout=60, env=child_env())
        return out.returncode == 0
    except Exception:
        return False


def configured_python_executable():
    """The user-provided interpreter from QgsSettings, or None."""
    try:
        from qgis.core import QgsSettings
        val = QgsSettings().value(QSETTINGS_PYTHON_KEY, "")
    except Exception:
        return None
    return str(val).strip() or None


def path_is_inside(path, root) -> bool:
    """True when ``path`` is ``root`` or a descendant. Only the ROOT is
    realpath'd unconditionally — a venv's bin/python is an absolute
    symlink, so realpath'ing the leaf would relocate WINMOL's own
    interpreter outside its venv. The realpath'd candidate is a
    fallback (macOS /var -> /private/var); normcase for Windows."""
    if not path or not root:
        return False
    b = os.path.normcase(os.path.realpath(root))
    for candidate in (os.path.abspath(path), os.path.realpath(path)):
        a = os.path.normcase(candidate)
        if a == b or a.startswith(b + os.sep):
            return True
    return False


# --- sentinel (install once) ------------------------------------------------

def _file_hash(path) -> str:
    try:
        data = Path(path).read_bytes()
    except Exception:
        data = b""
    return hashlib.sha256(data).hexdigest()[:16]


def _marker_path(venv_path) -> str:
    return os.path.join(venv_path, READY_MARKER)


def marker_matches(venv_path) -> bool:
    """Sentinel matches the CURRENT requirements hash. Pure file I/O —
    no interpreter spawned, so it is safe on a GUI thread."""
    try:
        with open(_marker_path(venv_path)) as f:
            stored = json.load(f).get("req_hash")
    except Exception:
        return False
    return stored == _file_hash(plugin_requirements_path())


def is_ready(venv_path) -> bool:
    """The venv exists, runs a supported Python, and matches the
    current requirements."""
    py = get_venv_python_path(venv_path)
    if not os.path.exists(py):
        return False
    if not (MIN_PY <= _python_version(py) <= MAX_PY):
        return False
    return marker_matches(venv_path)


def _write_marker(venv_path) -> None:
    req = plugin_requirements_path()
    with open(_marker_path(venv_path), "w") as f:
        json.dump({"req_hash": _file_hash(req),
                   "requirements": str(req)}, f)


# --- streamed child processes -----------------------------------------------

class _Progress:
    """Timestamped status sink; a raising sink is ignored (reporting
    must never fail the install)."""

    def __init__(self, sink=None):
        self._sink = sink
        self._t0 = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def __call__(self, message) -> None:
        if self._sink is None:
            return
        try:
            self._sink(f"[{self.elapsed():>4.0f}s] {message}")
        except Exception:
            pass


def _as_progress(progress) -> _Progress:
    if isinstance(progress, _Progress):
        return progress
    return _Progress(progress)


def _clean_line(line) -> str:
    """Drop the newline; keep only the last \\r segment (progress bars
    redraw with carriage returns)."""
    return line.rstrip("\r\n").split("\r")[-1].rstrip()


def _kill(proc) -> None:
    try:
        proc.kill()
    except Exception:
        pass


def _tail_text(tail) -> str:
    return ("\n" + "\n".join(tail)) if tail else ""


def _run_streamed(cmd, progress=None, label="command", timeout=3600,
                  heartbeat=15.0) -> None:
    """Run ``cmd``, streaming merged stdout+stderr through ``progress``.
    A reader thread feeds a queue so the loop wakes on ``heartbeat``
    (liveness); ``timeout`` is a hard deadline (child killed). Runs
    with ``child_env()`` from a neutral cwd (a QGIS cwd contributes
    shadowing DLLs on Windows). RuntimeError with tail on failure."""
    progress = _as_progress(progress)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=child_env({"PYTHONUNBUFFERED": "1"}),
        cwd=safe_child_cwd(cmd[0] if cmd else None))
    lines = queue.Queue()
    tail = collections.deque(maxlen=_TAIL_LINES)

    def _pump(stream):
        try:
            for raw in stream:
                lines.put(raw)
        except Exception:
            pass
        finally:
            lines.put(None)
            try:
                stream.close()
            except Exception:
                pass

    threading.Thread(target=_pump, args=(proc.stdout,),
                     daemon=True).start()

    deadline = time.monotonic() + timeout
    started = time.monotonic()
    while True:
        if time.monotonic() > deadline:
            _kill(proc)
            raise RuntimeError(
                f"{label} timed out after {timeout:.0f}s and was "
                "stopped." + _tail_text(tail))
        try:
            raw = lines.get(timeout=heartbeat)
        except queue.Empty:
            progress(f"{label} — still working "
                     f"({time.monotonic() - started:.0f}s)…")
            continue
        if raw is None:
            break
        text = _clean_line(raw)
        if text:
            tail.append(text)
            progress(text[:_MAX_LINE])

    try:
        rc = proc.wait(timeout=30)
    except Exception:
        _kill(proc)
        rc = proc.poll() or 1
    if rc != 0:
        for line in tail:   # the dialog may truncate the exception
            progress(line[:_MAX_LINE])
        raise RuntimeError(f"{label} failed (exit {rc})."
                           + _tail_text(tail))


# --- venv creation + install ------------------------------------------------

def _looks_like_missing_venv_package(exc) -> bool:
    """Debian/Ubuntu's "install the python3-venv package" failure."""
    text = str(exc).lower()
    return ("python3-venv" in text
            or "ensurepip is not available" in text)


def create_venv(venv_path, base_python=None, progress=None) -> None:
    """Build the venv. No ``--copies`` (macOS CLT python can't); the
    ``child_env()`` inside _run_streamed is load-bearing — with QGIS's
    PYTHONHOME inherited this exact call died on Windows."""
    base_python = base_python or choose_base_python()
    progress = _as_progress(progress)
    progress(f"Creating the virtual environment with {base_python} …")
    try:
        _run_streamed([base_python, "-m", "venv", venv_path],
                      progress=progress, label="venv creation",
                      timeout=300, heartbeat=10.0)
    except RuntimeError as exc:
        hint = ""
        if _looks_like_missing_venv_package(exc):
            hint = (" On Debian/Ubuntu install the matching "
                    "python3-venv package (e.g. `sudo apt install "
                    "python3.11-venv`), or point WINMOL at an existing "
                    "interpreter in the plugin settings.")
        raise RuntimeError(
            f"venv creation failed with {base_python}: {exc}{hint}"
        ) from exc


def ensure_pip(venv_path, progress=None) -> None:
    py = get_venv_python_path(venv_path)
    progress = _as_progress(progress)
    if subprocess.run([py, "-I", "-c", "import pip"],
                      capture_output=True, timeout=120,
                      env=child_env()).returncode == 0:
        progress("pip is available.")
        return
    progress("pip missing — bootstrapping it with ensurepip …")
    _run_streamed([py, "-m", "ensurepip", "--upgrade"],
                  progress=progress, label="ensurepip", timeout=300,
                  heartbeat=10.0)


def install_requirements(venv_path, progress=None) -> None:
    """pip-install requirements/cpu.txt into the venv, streaming pip's
    output (``--no-input`` prevents a hidden prompt; ``--progress-bar
    off`` stops \\r spam a QPlainTextEdit can't render)."""
    py = get_venv_python_path(venv_path)
    progress = _as_progress(progress)
    req = str(plugin_requirements_path())
    progress(f"Installing packages from {os.path.basename(req)} — the "
             "first run downloads a few hundred MB and can take "
             "several minutes …")
    _run_streamed(
        [py, "-u", "-m", "pip", "install", "--upgrade", "--no-input",
         "--progress-bar", "off", "-r", req],
        progress=progress,
        label=f"pip install -r {os.path.basename(req)}", timeout=3600)


def setup_environment(venv_path, base_python=None, progress=None) -> dict:
    """Create the venv + install deps (idempotent via the sentinel).
    Returns ``{'python': <exe>}``; raises only on a real venv/pip
    failure, which callers convert to a retry."""
    report = _as_progress(progress)
    if not is_ready(venv_path):
        report("Setting up the WINMOL environment …")
        if not os.path.exists(get_venv_python_path(venv_path)):
            create_venv(venv_path, base_python, progress=report)
        ensure_pip(venv_path, progress=report)
        install_requirements(venv_path, progress=report)
        _write_marker(venv_path)
    report(f"Environment ready in {report.elapsed():.0f}s: "
           f"{get_venv_python_path(venv_path)}")
    return {"python": get_venv_python_path(venv_path)}


# --- top-level resolution used by classFactory ------------------------------

def resolve_environment(plugin_dir, prompt=False, build=False) -> dict:
    """Decide which Python runs winmol_run.py. Returns {'status':
    'byo'|'ready'|'installed'|'needs_setup'|'error', 'python',
    'venv_path', 'message'}; never raises, so QGIS keeps loading.
    ``build=False`` (plugin load) never does heavy work — 'needs_setup'
    instead. ``prompt`` is accepted for call-site compat, ignored."""
    del prompt
    venv_path = venv_location(plugin_dir)
    result = {"venv_path": venv_path, "python": None, "message": ""}

    byo = configured_python_executable()
    if byo and path_is_inside(byo, venv_path) and not os.path.isfile(byo):
        # Stale pointer at WINMOL's OWN managed venv, deleted by hand:
        # not user-chosen, so ignore it and fall through to
        # is_ready/needs_setup so the env can be rebuilt. A missing
        # EXTERNAL interpreter stays an error below.
        byo = None
    if byo:
        ver = _python_version(byo)
        if not (MIN_PY <= ver <= MAX_PY):
            result.update(
                status="error",
                message=(f"Configured interpreter {byo} is Python "
                         f"{ver[0]}.{ver[1]}; WINMOL needs "
                         f"{MIN_PY[0]}.{MIN_PY[1]}."))
        elif _has_compute_deps(byo):
            result.update(status="byo", python=byo,
                          message=f"Using configured interpreter: {byo}")
        else:
            result.update(
                status="error", python=byo,
                message=(f"Configured interpreter {byo} is missing "
                         "WINMOL deps (onnxruntime/rasterio/"
                         "geopandas)."))
        return result

    if is_ready(venv_path):
        result.update(status="ready",
                      python=get_venv_python_path(venv_path),
                      message="WINMOL environment ready.")
        return result

    if not build:
        result.update(
            status="needs_setup",
            message="WINMOL environment not set up yet. Open the "
                    "plugin dialog to create it (Python 3.11 + "
                    "onnxruntime).")
        return result

    try:
        info = setup_environment(venv_path)
        result.update(status="installed", python=info["python"],
                      message="WINMOL environment installed.")
    except Exception as exc:
        result.update(status="error",
                      message=f"WINMOL setup failed: {exc}")
    return result
