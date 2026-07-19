"""Environment setup for the WINMOL QGIS plugin.

The plugin never runs inference in QGIS's own interpreter; it shells out to
``winmol_run.py`` in a separate Python environment that has the compute deps
(requirements/plugin.txt — onnxruntime + the geo stack, NO TensorFlow).

Two ways to get that environment:
  1. Point the plugin at an EXISTING interpreter (QgsSettings key
     ``winmol/python_executable``) — a conda env, the CI docker image's python,
     any venv with the deps. Most robust.
  2. Let the plugin create a venv next to itself (``winmol_venv``) and
     pip-install requirements/plugin.txt once.

Design notes (fixing the old flaky installer):
- Import-safe without QGIS/PyQt: Qt and QgsSettings are imported lazily inside
  the functions that need them, so this module is unit-testable off-QGIS.
- One-time install, gated by a ``.winmol_ready`` sentinel keyed to the
  requirements hash — no pip on every QGIS start.
- Never bricks the plugin: setup returns a status; callers handle it and offer a
  retry action instead of raising out of classFactory.
- No pkg_resources, no ``sudo apt``/``brew``; the get-pip fallback is fixed.
"""
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .childenv import child_env

WINMOL_VENV_NAME = "winmol_venv"
MODELS_PATH = "models"
READY_MARKER = ".winmol_ready"
QSETTINGS_PYTHON_KEY = "winmol/python_executable"

# WINMOL standardises on Python 3.11 everywhere: the code is validated only on
# 3.11 and the managed environment is always built as 3.11 (downloaded via
# plugin_utils/py311.py when the host has none). MIN==MAX pins it exactly.
MIN_PY = (3, 11)
MAX_PY = (3, 11)

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- paths ------------------------------------------------------------------

def repo_requirements_dir() -> Path:
    return Path(_PLUGIN_DIR, "requirements")


def plugin_requirements_path() -> Path:
    path = repo_requirements_dir().joinpath("plugin.txt")
    if not path.exists():   # fall back to base if plugin.txt is absent
        path = repo_requirements_dir().joinpath("base.txt")
    return path


def managed_root(plugin_dir) -> str:
    """Directory holding WINMOL's managed artifacts (the venv and, when
    downloaded, the Python 3.11 runtime).

    Prefer a location OUTSIDE the plugin directory (under the QGIS profile dir)
    so uninstalling the plugin -- a recursive delete of ``plugin_dir`` -- never
    has to remove thousands of venv/runtime files and symlinks. That deletion
    is exactly what QGIS reports as "plugin uninstall failed" when a half-built
    or symlinked venv lives inside the plugin folder. Off-QGIS (unit tests,
    headless) there is no profile, so fall back to ``plugin_dir``.
    """
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


def managed_base_python(plugin_dir, progress=None) -> str:
    """A Python 3.11 interpreter to build the venv from.

    Prefer a 3.11 already on PATH (no download); otherwise download a
    relocatable python-build-standalone 3.11 into ``managed_root/py311`` — so a
    bare machine with only QGIS (fresh Windows, macOS system 3.9, no conda)
    still gets a working 3.11. Raises RuntimeError only if no 3.11 is on PATH
    AND the download/extract fails.
    """
    import shutil
    for name in ("python3.11", "python3.11.exe", "python3", "python"):
        exe = shutil.which(name)
        if exe and _python_version(exe) == (3, 11):
            return exe
    from . import py311
    dest = os.path.join(managed_root(plugin_dir), "py311")
    return py311.ensure_python311(dest, progress=progress)


def get_python_command() -> str:
    import shutil
    return "python3" if shutil.which("python3") else "python"


def get_venv_python_path(venv_path) -> str:
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    # venv always installs a 'python' symlink; prefer it (no version guessing)
    py = os.path.join(venv_path, "bin", "python")
    return py if os.path.exists(py) else os.path.join(venv_path, "bin",
                                                      get_python_command())


# --- base interpreter selection --------------------------------------------

def _python_version(executable) -> tuple:
    try:
        # -I isolates the child from user site-packages and env vars; child_env
        # additionally strips the PYTHONHOME/PYTHONPATH QGIS exports, which
        # would otherwise point this interpreter at QGIS's stdlib and stop it
        # starting at all (see plugin_utils/childenv.py).
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
    """A system python suitable (version-wise) to build the venv from.

    Returns the executable path, or raises RuntimeError with an actionable
    message if none in the supported range is found.
    """
    import shutil
    candidates = []
    for name in ("python3.11", "python3.10", "python3.12", "python3.9",
                 "python3", "python"):
        exe = shutil.which(name)
        if exe and exe not in candidates:
            candidates.append(exe)
    for exe in candidates:
        ver = _python_version(exe)
        if MIN_PY <= ver <= MAX_PY:
            return exe
    raise RuntimeError(
        "No suitable Python found to build the WINMOL environment (need "
        f"{MIN_PY[0]}.{MIN_PY[1]}-{MAX_PY[0]}.{MAX_PY[1]}). Install one, or "
        "set an existing interpreter in the plugin settings "
        f"({QSETTINGS_PYTHON_KEY}).")


# --- bring-your-own environment --------------------------------------------

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
    except Exception:
        return None
    val = QgsSettings().value(QSETTINGS_PYTHON_KEY, "")
    return str(val).strip() or None


# --- sentinel (install once) -----------------------------------------------

def _requirements_hash() -> str:
    try:
        data = plugin_requirements_path().read_bytes()
    except Exception:
        data = b""
    return hashlib.sha256(data).hexdigest()[:16]


def _marker_path(venv_path) -> str:
    return os.path.join(venv_path, READY_MARKER)


def is_ready(venv_path) -> bool:
    """True when the venv exists, runs a supported Python, and was installed
    against the current requirements."""
    py = get_venv_python_path(venv_path)
    if not os.path.exists(py):
        return False
    if not (MIN_PY <= _python_version(py) <= MAX_PY):
        return False   # e.g. a stale 3.9 venv the code can't run
    try:
        with open(_marker_path(venv_path)) as f:
            return json.load(f).get("req_hash") == _requirements_hash()
    except Exception:
        return False


def _write_marker(venv_path) -> None:
    with open(_marker_path(venv_path), "w") as f:
        json.dump({"req_hash": _requirements_hash(),
                   "requirements": str(plugin_requirements_path())}, f)


# --- venv creation + install -----------------------------------------------

def create_venv(venv_path, base_python=None) -> None:
    base_python = base_python or choose_base_python()
    # No --copies: the macOS Command Line Tools python (3.9) cannot create
    # venvs without symlinks ("This build of python cannot create venvs without
    # using symlinks"). The symlinked default works on all platforms.
    # env=child_env() is load-bearing: with QGIS's PYTHONHOME inherited, this
    # exact call is what died on Windows with "could not import runpy module"
    # (-m is handled by runpy, which cannot load from a foreign stdlib).
    r = subprocess.run([base_python, "-m", "venv", venv_path],
                       capture_output=True, text=True, timeout=300,
                       env=child_env())
    if r.returncode != 0:
        raise RuntimeError(
            f"venv creation failed with {base_python} (exit {r.returncode}): "
            f"{(r.stderr or r.stdout).strip()[:600]}")


def ensure_pip(venv_path) -> None:
    py = get_venv_python_path(venv_path)
    if subprocess.run([py, "-I", "-c", "import pip"], capture_output=True,
                      timeout=120, env=child_env()).returncode == 0:
        return
    if subprocess.run([py, "-m", "ensurepip", "--upgrade"],
                      capture_output=True, timeout=300,
                      env=child_env()).returncode == 0:
        return
    # last resort: bootstrap pip from the network
    get_pip = Path(_PLUGIN_DIR, "plugin_utils", "get-pip.py")
    if not get_pip.exists():
        urllib.request.urlretrieve(
            "https://bootstrap.pypa.io/get-pip.py", str(get_pip))
    r = subprocess.run([py, str(get_pip)], capture_output=True, timeout=600,
                       env=child_env())
    if r.returncode != 0:
        raise RuntimeError(
            "Could not bootstrap pip in the WINMOL venv. On Debian/Ubuntu "
            "install the 'python3-venv' package for your Python; then retry. "
            f"pip error: {r.stderr.decode(errors='replace')[:500]}")


def install_requirements(venv_path) -> None:
    py = get_venv_python_path(venv_path)
    req = str(plugin_requirements_path())
    r = subprocess.run(
        [py, "-m", "pip", "install", "--upgrade", "-r", req],
        capture_output=True, timeout=3600, env=child_env())
    if r.returncode != 0:
        raise RuntimeError(
            f"pip failed installing {req} (exit {r.returncode}). "
            f"{r.stderr.decode(errors='replace')[-800:]}")


def download_models(plugin_dir, config_path=None) -> list:
    """Download configured models into <plugin>/models. Tolerant: skips files
    that exist, and returns the list of models that could NOT be fetched
    (missing URL / download error) instead of raising, so a hosting gap never
    bricks the plugin."""
    models_dir = os.path.join(plugin_dir, MODELS_PATH)
    os.makedirs(models_dir, exist_ok=True)
    config_path = config_path or os.path.join(plugin_dir, "config.json")
    missing = []
    try:
        with open(config_path) as f:
            entries = json.load(f)
    except Exception:
        return missing
    for name, url in entries.items():
        if not isinstance(url, str) or not url.lower().startswith("http"):
            missing.append(name)
            continue
        ext = ".onnx" if url.lower().split("?")[0].endswith(".onnx") else \
            os.path.splitext(url.split("?")[0])[1] or ".onnx"
        dest = os.path.join(models_dir, f"{name}{ext}")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        try:
            urllib.request.urlretrieve(url, dest)
        except (urllib.error.URLError, OSError):
            if os.path.exists(dest):
                os.remove(dest)   # drop truncated file
            missing.append(name)
    return missing


def installed_message(missing_models) -> str:
    """The message shown after a successful environment build.

    download_models() never raises, so a hosting gap or an offline machine
    leaves the plugin installed but model-less. Say so explicitly and name the
    escape hatch, rather than reporting plain success and letting the user
    discover an empty model list on their own.
    """
    if not missing_models:
        return "WINMOL environment installed."
    names = ", ".join(sorted(missing_models))
    return ("WINMOL environment installed, but these models could not be "
            f"downloaded: {names}. Check your internet connection and use the "
            "Environment button to retry, or Browse to select a local .onnx "
            "model file.")


def setup_environment(venv_path, base_python=None, download=True,
                      plugin_dir=None, progress=None) -> dict:
    """Create the venv and install deps (idempotent via the sentinel).
    Returns {'python': <exe>, 'missing_models': [...]}. Raises only on a real
    environment failure (venv/pip/deps); callers convert that to a retry.

    The venv is always built on Python 3.11: ``base_python`` is used if given,
    else a 3.11 is resolved (PATH or a downloaded PBS build) via
    ``managed_base_python``. ``plugin_dir`` is where models are downloaded
    (``<plugin_dir>/models``) and where the runtime is cached; it must be
    passed explicitly now that the venv lives outside the plugin directory.
    ``progress`` (callable taking a status string) surfaces download/build
    steps to the UI."""
    if plugin_dir is None:
        plugin_dir = os.path.dirname(venv_path)
    if not is_ready(venv_path):
        if not os.path.exists(get_venv_python_path(venv_path)):
            base = base_python or managed_base_python(plugin_dir,
                                                      progress=progress)
            if progress:
                progress("Creating the WINMOL environment…")
            create_venv(venv_path, base)
        ensure_pip(venv_path)
        if progress:
            progress("Installing dependencies (onnxruntime + geo stack)…")
        install_requirements(venv_path)
        _write_marker(venv_path)
    missing = download_models(plugin_dir) if download else []
    return {"python": get_venv_python_path(venv_path),
            "missing_models": missing}


# --- top-level resolution used by classFactory -----------------------------

def resolve_environment(plugin_dir, prompt=True, build=True) -> dict:
    """Decide which Python runs winmol_run.py, setting one up if needed.

    Returns a status dict:
      {'status': 'byo'|'ready'|'installed'|'needs_setup'|'declined'|'error',
       'python': <exe or None>, 'venv_path': <path>, 'message': <str>,
       'missing_models': [...]}
    Never raises — a bad state is reported, not thrown, so QGIS keeps loading.

    With ``build=False`` (used at plugin load) it never does heavy work: if no
    ready env exists it returns 'needs_setup' instead of downloading Python /
    building the venv, so QGIS startup can't block. The dialog then builds the
    environment asynchronously (with progress) on first open/run.
    """
    venv_path = venv_location(plugin_dir)
    result = {"venv_path": venv_path, "python": None, "missing_models": []}

    byo = configured_python_executable()
    if byo:
        ver = _python_version(byo)
        if not (MIN_PY <= ver <= MAX_PY):
            result.update(
                status="error", python=None,
                message=(f"Configured interpreter {byo} is Python "
                         f"{ver[0]}.{ver[1]}; WINMOL needs "
                         f"{MIN_PY[0]}.{MIN_PY[1]}-{MAX_PY[0]}.{MAX_PY[1]}."))
        elif _has_compute_deps(byo):
            result.update(status="byo", python=byo,
                          message=f"Using configured interpreter: {byo}")
        else:
            result.update(
                status="error", python=byo,
                message=(f"Configured interpreter {byo} is missing WINMOL "
                         "deps (onnxruntime/rasterio/geopandas)."))
        return result

    if is_ready(venv_path):
        result.update(status="ready", python=get_venv_python_path(venv_path),
                      message="WINMOL environment ready.")
        return result

    if not build:
        result.update(
            status="needs_setup",
            message="WINMOL environment not set up yet. Open the plugin and "
                    "use the Environment button to create it (Python 3.11 + "
                    "onnxruntime).")
        return result

    if prompt and not _confirm_setup():
        result.update(status="declined",
                      message="WINMOL setup declined; run it later from the "
                              "plugin dialog.")
        return result

    try:
        info = setup_environment(venv_path, plugin_dir=plugin_dir)
        result.update(status="installed", python=info["python"],
                      missing_models=info["missing_models"],
                      message=installed_message(info["missing_models"]))
    except Exception as exc:   # pragma: no cover - environment dependent
        result.update(status="error",
                      message=f"WINMOL setup failed: {exc}")
    return result


def _confirm_setup() -> bool:
    """Ask the user (Qt); default Yes. True if unavailable (headless)."""
    try:
        from qgis.PyQt.QtWidgets import QMessageBox
    except Exception:
        return True
    reply = QMessageBox.question(
        None, "Set up WINMOL environment",
        "The WINMOL Analyzer needs a small Python environment (onnxruntime + "
        "geo libraries) and model files. Create it now?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    return reply == QMessageBox.Yes
