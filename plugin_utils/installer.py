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
- Verbose by design: the build takes minutes, so every step streams its child
  process output line by line through the optional ``progress`` callback (the
  dialog wires it to the same log panel the prediction run writes to) and a
  heartbeat proves liveness while a single step is quiet. ``progress=None``
  stays the default, so batch/CLI/headless callers are unaffected.
"""
import collections
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
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


# --- progress reporting -----------------------------------------------------

# Longest line pushed into the log widget; pip can emit very long paths.
_MAX_LINE = 300
# How many lines of child output are kept for the failure message.
_TAIL_LINES = 80


class _Progress:
    """Timestamped status sink.

    Wraps the caller's ``progress`` callable (or nothing at all) so every call
    site can report unconditionally. Messages are prefixed with the seconds
    since the build started, which is what turns "it looks stuck" into "it is
    120 s in and still downloading". A sink that raises (a Qt widget destroyed
    mid-build) is ignored: reporting must never fail the install.
    """

    def __init__(self, sink=None):
        self._sink = sink
        self._t0 = time.monotonic()
        self._phase_t0 = self._t0

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def phase_elapsed(self) -> float:
        return time.monotonic() - self._phase_t0

    def phase(self, message) -> None:
        """Start a new phase (resets the per-phase clock) and announce it."""
        self._phase_t0 = time.monotonic()
        self(message)

    def __call__(self, message) -> None:
        if self._sink is None:
            return
        try:
            self._sink(f"[{self.elapsed():>4.0f}s] {message}")
        except Exception:
            pass


def _as_progress(progress) -> _Progress:
    """Normalise a callback / None / an existing _Progress into a _Progress."""
    if isinstance(progress, _Progress):
        return progress
    return _Progress(progress)


def _clean_line(line) -> str:
    """One printable line: drop the trailing newline and keep only the last
    carriage-return segment (progress bars redraw with \\r)."""
    return line.rstrip("\r\n").split("\r")[-1].rstrip()


def _run_streamed(cmd, progress=None, label="command", timeout=3600,
                  heartbeat=15.0, line_filter=None) -> None:
    """Run ``cmd``, streaming its output through ``progress`` as it arrives.

    stderr is merged into stdout: it avoids the two-pipe deadlock and, more
    importantly, pip writes most resolution/build failure detail to STDOUT, so
    a stderr-only tail is usually empty. A daemon reader thread feeds a queue
    so the main loop can wake on ``heartbeat`` and prove liveness instead of
    blocking in readline. ``timeout`` is enforced as a hard deadline (the
    child is killed), so this can never hang longer than the old
    ``subprocess.run(timeout=...)`` did.

    Raises RuntimeError with the tail of the real output on failure.
    """
    progress = _as_progress(progress)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=child_env({"PYTHONUNBUFFERED": "1"}))
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

    reader = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
    reader.start()

    deadline = time.monotonic() + timeout
    started = time.monotonic()
    while True:
        if time.monotonic() > deadline:
            _kill(proc)
            raise RuntimeError(
                f"{label} timed out after {timeout:.0f}s and was stopped."
                + _tail_text(tail))
        try:
            raw = lines.get(timeout=heartbeat)
        except queue.Empty:
            progress(f"{label} — still working "
                     f"({time.monotonic() - started:.0f}s)…")
            continue
        if raw is None:
            break
        text = _clean_line(raw)
        if not text:
            continue
        tail.append(text)
        shown = line_filter(text) if line_filter else text
        if shown:
            progress(shown[:_MAX_LINE])

    try:
        rc = proc.wait(timeout=30)
    except Exception:
        _kill(proc)
        rc = proc.poll() or 1
    if rc != 0:
        # Push the tail to the log too: the dialog may truncate the exception.
        for line in tail:
            progress(line[:_MAX_LINE])
        raise RuntimeError(f"{label} failed (exit {rc})." + _tail_text(tail))


def _kill(proc) -> None:
    try:
        proc.kill()
    except Exception:
        pass


def _tail_text(tail) -> str:
    return ("\n" + "\n".join(tail)) if tail else ""


def _requirement_names(path) -> list:
    """Package names in a requirements file, following ``-r`` includes.

    Purely cosmetic (it feeds the "N of M" counter), so anything unparseable
    is skipped rather than raised on.
    """
    names, seen = [], set()

    def _walk(current):
        current = Path(current)
        key = str(current)
        if key in seen:
            return
        seen.add(key)
        try:
            content = current.read_text()
        except OSError:
            return
        for raw in content.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith(("-r", "--requirement")):
                ref = line[2:] if line.startswith("-r") else line[13:]
                ref = ref.lstrip("=").strip()
                if ref:
                    _walk(current.parent / ref)
                continue
            if line.startswith("-"):
                continue
            name = re.split(r"[\s<>=!~;\[]", line, 1)[0].strip()
            if name and name not in names:
                names.append(name)

    _walk(path)
    return names


def _pip_line_filter(total):
    """Rewrite pip's ``Collecting <pkg>`` into ``Installing <pkg> (k of M)``.

    Cosmetic only — unrecognised lines pass through verbatim, and nothing here
    influences control flow or the exit-code check.
    """
    state = {"n": 0}

    def _filter(line):
        if line.startswith("Collecting "):
            state["n"] += 1
            pkg = line[len("Collecting "):].strip()
            if total:
                return f"Installing {pkg} ({state['n']} of {total})"
            return f"Installing {pkg}"
        return line

    return _filter


# --- venv creation + install -----------------------------------------------

def create_venv(venv_path, base_python=None, progress=None) -> None:
    base_python = base_python or choose_base_python()
    progress = _as_progress(progress)
    progress.phase(f"Creating the virtual environment with {base_python} …")
    # No --copies: the macOS Command Line Tools python (3.9) cannot create
    # venvs without symlinks ("This build of python cannot create venvs without
    # using symlinks"). The symlinked default works on all platforms.
    # env=child_env() is load-bearing: with QGIS's PYTHONHOME inherited, this
    # exact call is what died on Windows with "could not import runpy module"
    # (-m is handled by runpy, which cannot load from a foreign stdlib).
    try:
        _run_streamed([base_python, "-m", "venv", venv_path],
                      progress=progress, label="venv creation", timeout=300,
                      heartbeat=10.0)
    except RuntimeError as exc:
        raise RuntimeError(
            f"venv creation failed with {base_python}: {exc}") from exc
    progress(f"Virtual environment created in "
             f"{progress.phase_elapsed():.0f}s.")


def ensure_pip(venv_path, progress=None) -> None:
    py = get_venv_python_path(venv_path)
    progress = _as_progress(progress)
    progress.phase("Checking pip …")
    if subprocess.run([py, "-I", "-c", "import pip"], capture_output=True,
                      timeout=120, env=child_env()).returncode == 0:
        progress("pip is available.")
        return
    progress("pip missing — bootstrapping it with ensurepip …")
    try:
        _run_streamed([py, "-m", "ensurepip", "--upgrade"], progress=progress,
                      label="ensurepip", timeout=300, heartbeat=10.0)
        return
    except RuntimeError as exc:
        progress(f"ensurepip failed ({exc}); falling back to get-pip.py …")
    # last resort: bootstrap pip from the network
    get_pip = Path(_PLUGIN_DIR, "plugin_utils", "get-pip.py")
    if not get_pip.exists():
        progress("Downloading get-pip.py …")
        urllib.request.urlretrieve(
            "https://bootstrap.pypa.io/get-pip.py", str(get_pip))
    try:
        _run_streamed([py, str(get_pip)], progress=progress, label="get-pip",
                      timeout=600, heartbeat=10.0)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not bootstrap pip in the WINMOL venv. On Debian/Ubuntu "
            "install the 'python3-venv' package for your Python; then retry. "
            f"pip error: {exc}") from exc


def install_requirements(venv_path, progress=None) -> None:
    install_requirements_into(get_venv_python_path(venv_path),
                              progress=progress)


def install_requirements_into(python_exe, progress=None) -> None:
    """pip-install requirements/plugin.txt into an arbitrary interpreter.

    Used both for the managed venv and for an interpreter the user picked in
    the dialog. Streams pip's own output so a multi-minute install visibly
    progresses; ``--no-input`` prevents a hidden prompt that would be
    indistinguishable from a hang, and ``--progress-bar off`` stops the \\r
    bar spam that a QPlainTextEdit cannot render usefully.
    """
    progress = _as_progress(progress)
    req = str(plugin_requirements_path())
    total = len(_requirement_names(req))
    progress.phase(
        f"Installing {total} packages from {os.path.basename(req)} into "
        f"{python_exe} — the first run downloads a few hundred MB and can "
        "take several minutes …")
    _run_streamed(
        [python_exe, "-u", "-m", "pip", "install", "--upgrade", "--no-input",
         "--progress-bar", "off", "-r", req],
        progress=progress, label=f"pip install -r {os.path.basename(req)}",
        timeout=3600, line_filter=_pip_line_filter(total))
    progress(f"Dependencies installed in {progress.phase_elapsed():.0f}s.")


def _model_reporthook(progress, label):
    """urlretrieve reporthook throttled to ~1 message/s or every 5 MB, so a
    fast link cannot flood the log widget."""
    state = {"at": 0.0, "mb": -5}

    def _hook(blocks, block_size, total_size):
        read_mb = (blocks * block_size) // (1 << 20)
        now = time.monotonic()
        if now - state["at"] < 1.0 and read_mb - state["mb"] < 5:
            return
        state["at"], state["mb"] = now, read_mb
        if total_size and total_size > 0:
            progress(f"{label} — {read_mb}/{total_size // (1 << 20)} MB")
        else:
            progress(f"{label} — {read_mb} MB")

    return _hook


def download_models(plugin_dir, config_path=None, progress=None) -> list:
    """Download configured models into <plugin>/models. Tolerant: skips files
    that exist (verified against the registry checksum where one is pinned),
    and returns the list of models that could NOT be fetched (missing URL /
    download error / checksum mismatch) instead of raising, so a hosting gap
    never bricks the plugin.

    Backed by plugin_utils/model_registry. A legacy flat {name: url} config
    keeps the historical fetch-everything behavior (key-based
    ``models/<Name>.onnx`` naming, skip existing non-empty files). A schema-v2
    registry fetches ONLY the entries named in its ``preload`` list — the
    shipped registry has none, so plugin startup performs zero network I/O
    and models download on demand when selected in the dialog."""
    models_dir = os.path.join(plugin_dir, MODELS_PATH)
    os.makedirs(models_dir, exist_ok=True)
    config_path = config_path or os.path.join(plugin_dir, "config.json")
    progress = _as_progress(progress)
    missing = []
    try:
        from . import model_registry
        registry = model_registry.load_registry(config_path)
    except Exception:
        return missing
    if registry.schema >= 2:
        for name in registry.preload:
            entry = registry.entries.get(name)
            progress.phase(f"Downloading model {name} …")
            try:
                if entry is None:
                    raise KeyError(name)
                model_registry.ensure_model(entry, models_dir)
                progress(f"Model {name} downloaded in "
                         f"{progress.phase_elapsed():.0f}s.")
            except Exception as exc:
                progress(f"Model {name} could not be downloaded: {exc}")
                missing.append(name)
        return missing
    total = len(registry.entries)
    if total:
        progress.phase(f"Downloading {total} model file(s) …")
    for index, entry in enumerate(registry.entries.values(), start=1):
        name, url = entry.id, entry.url
        if not isinstance(url, str) or not url.lower().startswith("http"):
            missing.append(name)
            continue
        dest = os.path.join(models_dir, entry.file)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            progress(f"Model {name} ({index} of {total}) already present.")
            continue
        label = f"Downloading model {name} ({index} of {total})"
        progress.phase(label + " …")
        try:
            urllib.request.urlretrieve(
                url, dest, reporthook=_model_reporthook(progress, label))
            progress(f"Model {name} downloaded in "
                     f"{progress.phase_elapsed():.0f}s.")
        except (urllib.error.URLError, OSError) as exc:
            if os.path.exists(dest):
                os.remove(dest)   # drop truncated file
            progress(f"Model {name} could not be downloaded: {exc}")
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
    report = _as_progress(progress)
    if not is_ready(venv_path):
        report.phase("Setting up the WINMOL environment …")
        if not os.path.exists(get_venv_python_path(venv_path)):
            report("Locating a Python 3.11 interpreter …")
            base = base_python or managed_base_python(plugin_dir,
                                                      progress=report)
            create_venv(venv_path, base, progress=report)
        ensure_pip(venv_path, progress=report)
        install_requirements(venv_path, progress=report)
        _write_marker(venv_path)
    else:
        report("WINMOL environment already built; checking models …")
    missing = download_models(plugin_dir, progress=report) if download else []
    report(f"Environment ready in {report.elapsed():.0f}s: "
           f"{get_venv_python_path(venv_path)}")
    if missing:
        report(installed_message(missing))
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
