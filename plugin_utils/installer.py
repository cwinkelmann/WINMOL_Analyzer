"""Environment setup for the WINMOL QGIS plugin.

Inference runs in a separate Python env (requirements/cpu.txt —
onnxruntime + geo stack — or, opted in via WINMOL_GPU=1, gpu.txt with
onnxruntime-gpu): either a user-configured interpreter (QgsSettings
``winmol/python_executable``) or a venv this module builds once,
blessed by a ``.winmol_ready`` sentinel keyed to the requirements hash
and variant. Import-safe without QGIS/PyQt (lazy imports); every child
process runs with ``child_env()`` because QGIS exports PYTHONHOME/
PYTHONPATH/GDAL vars that break any foreign interpreter (childenv.py).
"""
import collections
import hashlib
import json
import os
import queue
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

from .childenv import (
    PY_VERSION_PROBE,
    child_env,
    run_isolated,
    safe_child_cwd,
)
from .gpu_probe import verify_gpu_providers

WINMOL_VENV_NAME = "winmol_venv"
MODELS_DIR_NAME = "models"
READY_MARKER = ".winmol_ready"
QSETTINGS_PYTHON_KEY = "winmol/python_executable"
#: Any value here silences the dialog's one-line GPU offer.
QSETTINGS_GPU_PROMPT_KEY = "winmol/gpu_offer_dismissed"
CPU_REQUIREMENTS = "cpu.txt"
GPU_REQUIREMENTS = "gpu.txt"
#: Environment variable that opts a rebuild into the GPU runtime.
GPU_ENV_VAR = "WINMOL_GPU"

#: The two inference runtimes, and the rule about them: both provide
#: the ``onnxruntime`` module, so exactly ONE may be installed. Every
#: code path that installs one uninstalls the other first — pip never
#: will (the two are unrelated distribution names to it), and with
#: both present the loser's dangling shared libraries produce import
#: errors that read like a broken CUDA install.
CPU_RUNTIME_DIST = "onnxruntime"
GPU_RUNTIME_DIST = "onnxruntime-gpu"

# WINMOL is validated on Python 3.11 only; MIN==MAX pins it exactly.
MIN_PY = (3, 11)
MAX_PY = (3, 11)

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAX_LINE = 300
_TAIL_LINES = 80


# --- paths ------------------------------------------------------------------

def repo_requirements_dir() -> Path:
    return Path(_PLUGIN_DIR, "requirements")


def plugin_requirements_path(gpu=False) -> Path:
    """The requirements file the compute environment is built from.
    ``gpu=True`` selects the CUDA twin (onnxruntime-gpu), falling back
    to cpu.txt when gpu.txt is missing from an incomplete checkout."""
    if gpu:
        path = repo_requirements_dir().joinpath(GPU_REQUIREMENTS)
        if path.exists():
            return path
    return repo_requirements_dir().joinpath(CPU_REQUIREMENTS)


def gpu_requested() -> bool:
    """True when WINMOL_GPU=1 (or true/yes/on) asks for the CUDA
    runtime. The only opt-in switch until a Setup tab exists."""
    value = os.environ.get(GPU_ENV_VAR, "").strip().lower()
    return value in ("1", "true", "yes", "on")


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


def models_location(plugin_dir) -> str:
    """The downloaded-models directory (a sibling of the venv, so it
    also survives a QGIS plugin uninstall)."""
    return os.path.join(managed_root(plugin_dir), MODELS_DIR_NAME)


def autotune_cache_location(plugin_dir) -> str:
    """Absolute path of the prediction batch-size autotune cache.

    Lives beside the venv under ``managed_root`` so it is part of WINMOL's
    managed state: the dialog hands it to the compute child through
    ``$WINMOL_AUTOTUNE_CACHE``. See plugin_utils/autotune_cache.py.
    """
    from .autotune_cache import CACHE_FILENAME
    return os.path.join(managed_root(plugin_dir), CACHE_FILENAME)


def get_venv_python_path(venv_path) -> str:
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    # venv always installs a 'python' symlink; prefer it.
    py = os.path.join(venv_path, "bin", "python")
    return py if os.path.exists(py) else os.path.join(
        venv_path, "bin", "python3")


# --- base interpreter selection ---------------------------------------------

def managed_base_python(plugin_dir, progress=None) -> str:
    """A Python 3.11 interpreter to build the venv from.

    Prefer a 3.11 already on PATH (no download); otherwise download a
    relocatable python-build-standalone 3.11 into ``managed_root/py311`` —
    so a bare machine with only QGIS (fresh Windows, macOS system 3.9, no
    conda) still gets a working 3.11. Raises RuntimeError only if no 3.11
    is on PATH AND the download/extract fails.
    """
    for name in ("python3.11", "python3.11.exe", "python3", "python"):
        exe = shutil.which(name)
        if exe and _python_version(exe) == MIN_PY:
            return exe
    from . import py311
    dest = os.path.join(managed_root(plugin_dir), "py311")
    return py311.ensure_python311(dest, progress=progress)


def _python_version(executable) -> tuple:
    """(major, minor) of ``executable``, or (0, 0) when unusable.
    ``run_isolated`` (``-I`` + ``child_env()``): QGIS's PYTHONHOME/
    PYTHONPATH would point the child at QGIS's stdlib and stop it
    starting at all."""
    try:
        out = run_isolated(executable, PY_VERSION_PROBE, timeout=30)
        if out.returncode == 0:
            major, minor = out.stdout.strip().split(".")
            return (int(major), int(minor))
    except Exception:
        pass
    return (0, 0)


def choose_base_python(progress=None) -> str:
    """A system python (3.11) to build the venv from.

    Prefers PATH, but never dead-ends there: with none found it falls back
    to :func:`managed_base_python`, which downloads a relocatable Python
    3.11 (see plugin_utils/py311.py) — so a bare machine with only QGIS
    (fresh Windows, macOS system 3.9, no terminal) still gets a working
    venv. ``progress`` is forwarded to that download so its "Downloading
    Python 3.11 …" lines reach the caller's log. Only raises RuntimeError
    when even the download fails (unsupported platform or network error);
    that error names the manual fallback.
    """
    candidates = []
    for name in ("python3.11", "python3", "python"):
        exe = shutil.which(name)
        if exe and exe not in candidates:
            candidates.append(exe)
    for exe in candidates:
        if MIN_PY <= _python_version(exe) <= MAX_PY:
            return exe
    return managed_base_python(_PLUGIN_DIR, progress=progress)


#: What a compute environment must be able to import for a real run.
#: Single source of truth: the bring-your-own-interpreter probe below,
#: the messages that name the deps, and the requirements guard in
#: tests/test_plugin_installer.py all read it. `onnx` was declared in
#: core.txt but not here, so a BYO interpreter passed this gate and then
#: died at model load -- the exact failure the declaration fixed.
REQUIRED_RUNTIME_MODULES = ("onnxruntime", "onnx", "rasterio", "geopandas")


def _has_compute_deps(executable) -> bool:
    try:
        out = run_isolated(
            executable, "import " + ", ".join(REQUIRED_RUNTIME_MODULES),
            timeout=60)
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

def _requirements_closure(path, _seen=None):
    """``path`` and every file it pulls in with ``-r``, in stable order.

    cpu.txt and gpu.txt are thin: both are little more than ``-r
    core.txt`` plus a runtime. Hashing only the named file therefore
    misses every change to the shared stack -- a dependency added to
    core.txt would leave the sentinel matching, and no existing install
    would ever rebuild its venv. That is how the `onnx` requirement
    reached users as a ModuleNotFoundError instead of a reinstall.
    """
    path = Path(path)
    _seen = set() if _seen is None else _seen
    if path in _seen:
        return []
    _seen.add(path)
    found = [path]
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return found
    for line in lines:
        line = line.strip()
        if line.startswith("-r "):
            found += _requirements_closure(path.parent / line[3:].strip(),
                                           _seen)
    return found


def _file_hash(path) -> str:
    """Hash of the whole requirements closure, not just the entry file."""
    digest = hashlib.sha256()
    for part in _requirements_closure(path):
        try:
            digest.update(part.read_bytes())
        except Exception:
            digest.update(b"")
    return digest.hexdigest()[:16]


def _marker_path(venv_path) -> str:
    return os.path.join(venv_path, READY_MARKER)


def _read_marker(venv_path) -> dict:
    try:
        with open(_marker_path(venv_path)) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def installed_variant(venv_path):
    """'cpu' | 'gpu' as recorded by the sentinel, or None without one.
    Markers from before the variant field are cpu.txt installs."""
    marker = _read_marker(venv_path)
    if not marker:
        return None
    return marker.get("variant", "cpu")


def marker_matches(venv_path, gpu=None) -> bool:
    """Sentinel matches the CURRENT requirements hash of the venv's own
    variant; a bool ``gpu`` additionally requires that variant. Pure
    file I/O — no interpreter spawned, so safe on a GUI thread."""
    marker = _read_marker(venv_path)
    if not marker:
        return False
    variant = marker.get("variant", "cpu")
    if gpu is not None and variant != ("gpu" if gpu else "cpu"):
        return False
    req = plugin_requirements_path(gpu=(variant == "gpu"))
    return marker.get("req_hash") == _file_hash(req)


def _marker_python_version(venv_path):
    """(major, minor) as recorded by :func:`_write_marker`, or None
    when the marker predates the field or it is unreadable."""
    recorded = _read_marker(venv_path).get("python_version")
    if isinstance(recorded, (list, tuple)) and len(recorded) == 2:
        try:
            return (int(recorded[0]), int(recorded[1]))
        except (TypeError, ValueError):
            pass
    return None


def is_ready(venv_path, gpu=None) -> bool:
    """The venv exists, runs a supported Python, and matches the
    current requirements (of the required variant when ``gpu`` is a
    bool; of whichever variant it was built as when None). The Python
    version recorded in the marker is trusted when present — one
    subprocess saved on every QGIS start — with a live probe as the
    back-compat fallback for markers from older builds."""
    py = get_venv_python_path(venv_path)
    if not os.path.exists(py):
        return False
    version = _marker_python_version(venv_path)
    if version is None:
        version = _python_version(py)
    if not (MIN_PY <= version <= MAX_PY):
        return False
    return marker_matches(venv_path, gpu=gpu)


def _write_marker(venv_path, gpu=False) -> None:
    """Bless the venv, recording the venv python's version so the
    ready-path never has to spawn it again (see :func:`is_ready`). An
    unusable probe result is simply not recorded — is_ready then falls
    back to probing live."""
    req = plugin_requirements_path(gpu=gpu)
    data = {"req_hash": _file_hash(req),
            "variant": "gpu" if gpu else "cpu",
            "requirements": str(req)}
    version = _python_version(get_venv_python_path(venv_path))
    if version > (0, 0):
        data["python_version"] = list(version)
    with open(_marker_path(venv_path), "w") as f:
        json.dump(data, f)


def invalidate_marker(venv_path) -> bool:
    """Drop the ``.winmol_ready`` sentinel so :func:`is_ready` reports
    the environment as needing a rebuild. First step of both "Reinstall
    dependencies" and "Delete environment" — ``setup_environment``
    short-circuits on a valid marker, and a half-deleted venv must
    never keep being blessed as ready. True when one was removed."""
    try:
        os.remove(_marker_path(venv_path))
        return True
    except OSError:
        return False


# --- disk usage / removal ---------------------------------------------------

def directory_size(path) -> int:
    """Total size in bytes of the files under ``path`` (0 when absent).
    Symlinks are not followed and unreadable entries are skipped: this
    feeds a human-readable "frees N GB" figure, so it must never
    raise."""
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            full = os.path.join(root, name)
            try:
                if not os.path.islink(full):
                    total += os.path.getsize(full)
            except OSError:
                pass
    return total


def _managed_roots(plugin_dir) -> tuple:
    return (managed_root(plugin_dir), plugin_dir)


def _is_managed_path(path, plugin_dir) -> bool:
    """A path may only be deleted when it is a STRICT descendant of the
    managed root or of the plugin directory."""
    for root in _managed_roots(plugin_dir):
        if not root:
            continue
        if path_is_inside(path, root) and not path_is_inside(root, path):
            return True
    return False


def _removal_refusal(path, plugin_dir):
    """The shared refusal rule for anything remove_environment would
    delete: the ``(path, message)`` failure tuple, or None when the
    path may be removed. Two refusals exist. A symlink — deleting
    through one reaches (and, via the chmod-retry handler, could
    mutate) the TARGET, so the link itself must survive. And a path
    outside the managed tree (:func:`_is_managed_path`)."""
    if os.path.islink(path):
        return (path, "refused: the path is a symlink")
    if not _is_managed_path(path, plugin_dir):
        return (path, "refused: outside the managed tree")
    return None


def _chmod_retry(func, path):
    """rmtree error handler: clear the read-only bit (Windows venvs
    ship read-only files) and retry once; re-raise if it still fails.
    When the failing func is ``os.path.islink`` — rmtree refusing to
    operate on a symlink — re-raise immediately: chmod would follow
    the link and mutate the TARGET, and retrying islink() returns a
    bool instead of raising, silently masking the failure."""
    if func is os.path.islink:
        raise OSError(f"refusing to rmtree the symlink {path}")
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _rmtree(path) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=lambda f, p, _e: _chmod_retry(f, p))
    else:
        shutil.rmtree(path, onerror=lambda f, p, _i: _chmod_retry(f, p))


def remove_environment(plugin_dir, remove_venv=True, remove_runtime=False,
                       remove_models=False, configured_exe=None,
                       dry_run=False, progress=None) -> dict:
    """Delete WINMOL's managed artifacts. Never raises. REFUSES any
    path outside the managed tree (:func:`_is_managed_path`).

    Returns ``{'planned': [(path, bytes)], 'removed': [path],
    'failed': [(path, message)], 'freed_bytes': int,
    'clear_setting': bool}``. With ``dry_run=True`` the SAME dict shape
    comes back without anything being deleted — the confirmation text
    and its "frees N GB" figure are therefore produced by the exact
    code path that performs the deletion, so the two can never drift.

    ``clear_setting`` is True only when ``configured_exe`` resolves
    INSIDE a tree that is actually being removed; a bring-your-own
    conda interpreter is never touched and never un-configured. The
    caller (the dialog) performs the QgsSettings write — this module
    stays importable without QGIS.
    """
    report = _as_progress(progress)
    if not plugin_dir:
        # Never raises: a None/"" plugin_dir cannot resolve to a
        # managed tree, so there is nothing that may be deleted.
        return {"planned": [], "removed": [],
                "failed": [("", "refused: no plugin directory")],
                "freed_bytes": 0, "clear_setting": False}
    venv = venv_location(plugin_dir)
    runtime = os.path.join(managed_root(plugin_dir), "py311")
    models_dir = models_location(plugin_dir)
    result = {"planned": [], "removed": [], "failed": [],
              "freed_bytes": 0, "clear_setting": False}

    result["clear_setting"] = bool(
        remove_venv and configured_exe
        and path_is_inside(configured_exe, venv))

    trees = []
    if remove_venv:
        trees.append(venv)
    if remove_runtime:
        trees.append(runtime)
    for path in trees:
        refusal = _removal_refusal(path, plugin_dir)
        if refusal is not None:
            result["failed"].append(refusal)
            continue
        if os.path.isdir(path):
            result["planned"].append((path, directory_size(path)))

    model_plan = None
    if remove_models:
        refusal = _removal_refusal(models_dir, plugin_dir)
        if refusal is not None:
            result["failed"].append(refusal)
        else:
            model_plan = _plan_models(plugin_dir, models_dir, dry_run=True)
            if model_plan["freed_bytes"] or model_plan["planned"]:
                result["planned"].append(
                    (models_dir, model_plan["freed_bytes"]))

    if dry_run:
        if model_plan is not None:
            result["failed"].extend(model_plan["failed"])
        result["freed_bytes"] = sum(size for _p, size in result["planned"])
        return result

    # A half-deleted venv must degrade to "needs rebuild", never stay
    # blessed by is_ready(); do this BEFORE the first rmtree. Gated on
    # the same managed-tree and not-a-symlink checks as the rmtree
    # itself (through a symlinked venv it would reach the TARGET).
    if (remove_venv and not os.path.islink(venv)
            and _is_managed_path(venv, plugin_dir)):
        invalidate_marker(venv)

    for path, size in list(result["planned"]):
        if path == models_dir:
            continue
        report(f"Removing {path} …")
        try:
            _rmtree(path)
            result["removed"].append(path)
            result["freed_bytes"] += size
        except Exception as exc:
            result["failed"].append((path, str(exc)))
    if model_plan is not None:
        done = _plan_models(plugin_dir, models_dir, dry_run=False,
                            progress=report)
        result["removed"].extend(done["removed"])
        result["failed"].extend(done["failed"])
        result["freed_bytes"] += done["freed_bytes"]
    return result


def remove_model_files(entry, models_dir, dry_run=False) -> dict:
    """Delete (or, with ``dry_run``, price) ONE registry entry's model
    file and its ``.part`` leftover under ``models_dir``. THE single
    owner of model-file deletion — :func:`remove_environment`'s model
    phase and the Setup tab's per-model delete
    (tasks_threads.ModelMaintenanceWorker) both go through it, so the
    safety rule cannot drift: a path that resolves outside the models
    directory (an entry ``file`` of "../x", or an absolute path) is
    refused per victim, never removed.

    Returns ``{'planned': [(path, bytes)], 'removed': [path],
    'failed': [(path, message)], 'freed_bytes': int}``.
    """
    from . import model_registry
    result = {"planned": [], "removed": [], "failed": [], "freed_bytes": 0}
    base = model_registry.local_path(entry, models_dir)
    for victim in (base, base + ".part"):
        if not path_is_inside(victim, models_dir):
            result["failed"].append(
                (victim, "refused: outside the models directory"))
            continue
        try:
            size = os.path.getsize(victim)
        except OSError:
            continue
        result["planned"].append((victim, size))
        if dry_run:
            result["freed_bytes"] += size
            continue
        try:
            os.remove(victim)
            result["removed"].append(victim)
            result["freed_bytes"] += size
        except OSError as exc:
            result["failed"].append((victim, str(exc)))
    return result


def _plan_models(plugin_dir, models_dir, dry_run, progress=None):
    """Price/remove registry-known model files (and their ``.part``
    leftovers) under ``models_dir`` — the registry owns the on-disk
    naming rule, so unknown files are never touched. Both phases run
    through :func:`remove_model_files`, the one owner of the deletion
    (and its containment refusal)."""
    result = {"planned": [], "removed": [], "failed": [], "freed_bytes": 0}
    try:
        from . import model_registry
        registry = model_registry.load_registry(
            os.path.join(plugin_dir, "config.json"))
    except Exception as exc:
        if not dry_run and progress is not None:
            progress(f"Could not read the model registry: {exc}")
        return result
    for entry in registry.entries.values():
        done = remove_model_files(entry, models_dir, dry_run=dry_run)
        for key in ("planned", "removed", "failed"):
            result[key].extend(done[key])
        result["freed_bytes"] += done["freed_bytes"]
    return result


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
    progress = _as_progress(progress)
    base_python = base_python or choose_base_python(progress=progress)
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
    if run_isolated(py, "import pip", timeout=120).returncode == 0:
        progress("pip is available.")
        return
    progress("pip missing — bootstrapping it with ensurepip …")
    _run_streamed([py, "-m", "ensurepip", "--upgrade"],
                  progress=progress, label="ensurepip", timeout=300,
                  heartbeat=10.0)


def distribution_installed(python_exe, dist, timeout=60) -> bool:
    """True when ``dist`` is installed in ``python_exe``. Asks
    importlib.metadata, not an import: importing ``onnxruntime`` cannot
    tell the two distributions apart, which is the entire problem."""
    try:
        out = run_isolated(
            python_exe,
            "import importlib.metadata as m, sys;"
            "sys.exit(0 if m.distribution(sys.argv[1]) else 1)",
            timeout=timeout, args=(dist,))
        return out.returncode == 0
    except Exception:
        return False


def uninstall_conflicting_runtime(python_exe, gpu, progress=None) -> bool:
    """Remove the runtime that must not coexist with the one we
    install (see CPU_RUNTIME_DIST above — both ship the ``onnxruntime``
    module and pip will never resolve the conflict itself). Returns
    True when something was removed; failures are logged, not raised —
    the install that follows fails loudly on its own if this mattered."""
    progress = _as_progress(progress)
    doomed = CPU_RUNTIME_DIST if gpu else GPU_RUNTIME_DIST
    if not distribution_installed(python_exe, doomed):
        return False
    progress(f"Removing {doomed}: it cannot be installed alongside "
             f"{GPU_RUNTIME_DIST if gpu else CPU_RUNTIME_DIST} — both "
             "provide the 'onnxruntime' module.")
    try:
        _run_streamed(
            [python_exe, "-u", "-m", "pip", "uninstall", "-y", doomed],
            progress=progress, label=f"pip uninstall {doomed}",
            timeout=600)
        return True
    except Exception as exc:
        progress(f"Could not remove {doomed}: {exc}")
        return False


def install_requirements(venv_path, progress=None, gpu=False) -> None:
    """pip-install requirements/cpu.txt (or gpu.txt) into the venv,
    streaming pip's output (``--no-input`` prevents a hidden prompt;
    ``--progress-bar off`` stops \\r spam a QPlainTextEdit can't
    render). Uninstalls the conflicting runtime FIRST — a swap, never
    an addition."""
    py = get_venv_python_path(venv_path)
    progress = _as_progress(progress)
    uninstall_conflicting_runtime(py, gpu, progress=progress)
    req = str(plugin_requirements_path(gpu=gpu))
    progress(f"Installing packages from {os.path.basename(req)} — the "
             "first run downloads a few hundred MB and can take "
             "several minutes …")
    _run_streamed(
        [py, "-u", "-m", "pip", "install", "--upgrade", "--no-input",
         "--progress-bar", "off", "-r", req],
        progress=progress,
        label=f"pip install -r {os.path.basename(req)}", timeout=3600)


def setup_environment(venv_path, base_python=None, progress=None,
                      gpu=False) -> dict:
    """Create the venv + install deps (idempotent via the sentinel).
    ``gpu=True`` builds/rebuilds the onnxruntime-gpu variant. Returns
    ``{'python': <exe>}``; raises only on a real venv/pip failure,
    which callers convert to a retry."""
    report = _as_progress(progress)
    if not is_ready(venv_path, gpu=gpu):
        report("Setting up the WINMOL environment "
               f"({'GPU' if gpu else 'CPU'} runtime) …")
        if not os.path.exists(get_venv_python_path(venv_path)):
            create_venv(venv_path, base_python, progress=report)
        ensure_pip(venv_path, progress=report)
        install_requirements(venv_path, progress=report, gpu=gpu)
        if gpu:
            # Visible install-time verdict for the variant that can
            # silently fail to be what it claims (a CUDA generation
            # mismatch installs fine and only fails at runtime); the CPU
            # variant has nothing GPU-shaped to verify.
            report(verify_gpu_providers(get_venv_python_path(venv_path)))
        _write_marker(venv_path, gpu=gpu)
    report(f"Environment ready in {report.elapsed():.0f}s: "
           f"{get_venv_python_path(venv_path)}")
    return {"python": get_venv_python_path(venv_path)}


# --- top-level resolution used by classFactory ------------------------------

def resolve_environment(plugin_dir) -> dict:
    """Decide which Python runs winmol_run.py. Returns {'status':
    'byo'|'ready'|'needs_setup'|'error', 'python', 'venv_path',
    'message'}; never raises, so QGIS keeps loading. Never does heavy
    work — a missing environment reports 'needs_setup', and the actual
    build runs through setup_environment on a worker thread."""
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
                         "WINMOL deps ("
                         + "/".join(REQUIRED_RUNTIME_MODULES) + ")."))
        return result

    # WINMOL_GPU=1 requires the gpu variant; otherwise any variant the
    # venv was built as is ready (a GPU venv must not rebuild as CPU
    # just because the env var is unset today).
    want_gpu = gpu_requested()
    if is_ready(venv_path, gpu=True if want_gpu else None):
        result.update(status="ready",
                      python=get_venv_python_path(venv_path),
                      message="WINMOL environment ready.")
        return result

    message = ("WINMOL environment not set up yet. Open the "
               "plugin dialog to create it (Python 3.11 + "
               "onnxruntime).")
    if want_gpu:
        message = ("WINMOL_GPU=1: the environment will be "
                   "(re)built with the GPU runtime "
                   "(onnxruntime-gpu) on the next Run.")
    result.update(status="needs_setup", message=message)
    return result
