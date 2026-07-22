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
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import gpu_probe
from .childenv import child_env

WINMOL_VENV_NAME = "winmol_venv"
MODELS_PATH = "models"
READY_MARKER = ".winmol_ready"
QSETTINGS_PYTHON_KEY = "winmol/python_executable"

#: The two inference runtimes, and the rule about them: they both provide
#: the ``onnxruntime`` module, so exactly one may be installed. Every code
#: path that installs one uninstalls the other first.
CPU_RUNTIME_DIST = "onnxruntime"
GPU_RUNTIME_DIST = "onnxruntime-gpu"

#: The execution provider that proves the CUDA build is not just present
#: but loadable. Its ABSENCE after a GPU install is the failure this whole
#: feature exists to report instead of silently running on the CPU.
CUDA_PROVIDER = "CUDAExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"

# WINMOL standardises on Python 3.11 everywhere: the code is validated only on
# 3.11 and the managed environment is always built as 3.11 (downloaded via
# plugin_utils/py311.py when the host has none). MIN==MAX pins it exactly.
MIN_PY = (3, 11)
MAX_PY = (3, 11)

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- paths ------------------------------------------------------------------

def repo_requirements_dir() -> Path:
    return Path(_PLUGIN_DIR, "requirements")


def plugin_requirements_path(gpu=False) -> Path:
    """The requirements file the compute environment is built from.

    ``gpu=True`` selects the CUDA twin (onnxruntime-gpu). It falls back to
    the CPU file when plugin-gpu.txt is missing, so an incomplete checkout
    installs a working CPU environment rather than nothing at all.
    """
    if gpu:
        path = repo_requirements_dir().joinpath("plugin-gpu.txt")
        if path.exists():
            return path
    path = repo_requirements_dir().joinpath("plugin.txt")
    if not path.exists():   # fall back to base if plugin.txt is absent
        path = repo_requirements_dir().joinpath("base.txt")
    return path


# --- which runtime does this machine want? ----------------------------------

def detect_gpu(timeout=gpu_probe.NVIDIA_SMI_TIMEOUT):
    """:class:`gpu_probe.GpuProbe` for this machine. Never raises.

    Shells out to ``nvidia-smi`` (with a timeout — see gpu_probe), so it
    belongs on a worker thread, never on a repaint path.
    """
    try:
        return gpu_probe.probe(timeout=timeout)
    except Exception:                       # pragma: no cover - defensive
        return gpu_probe.GpuProbe(status=gpu_probe.STATUS_NO_DRIVER,
                                  detail="GPU detection failed.")


def wants_gpu_runtime(probe=None) -> bool:
    """True when this machine should get onnxruntime-gpu."""
    return (probe if probe is not None else detect_gpu()).present


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


def autotune_cache_location(plugin_dir) -> str:
    """Absolute path of the prediction batch-size autotune cache.

    Lives beside the venv under ``managed_root`` so it is part of WINMOL's
    managed state: the dialog hands it to the compute child through
    ``$WINMOL_AUTOTUNE_CACHE`` and "Delete environment" removes it. See
    plugin_utils/autotune_cache.py.
    """
    from .autotune_cache import CACHE_FILENAME
    return os.path.join(managed_root(plugin_dir), CACHE_FILENAME)


def managed_base_python(plugin_dir, progress=None) -> str:
    """A Python 3.11 interpreter to build the venv from.

    Prefer a 3.11 already on PATH (no download); otherwise download a
    relocatable python-build-standalone 3.11 into ``managed_root/py311`` — so a
    bare machine with only QGIS (fresh Windows, macOS system 3.9, no conda)
    still gets a working 3.11. Raises RuntimeError only if no 3.11 is on PATH
    AND the download/extract fails.
    """
    for name in ("python3.11", "python3.11.exe", "python3", "python"):
        exe = shutil.which(name)
        if exe and _python_version(exe) == (3, 11):
            return exe
    from . import py311
    dest = os.path.join(managed_root(plugin_dir), "py311")
    return py311.ensure_python311(dest, progress=progress)


def get_python_command() -> str:
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


def _nvidia_gpu_present() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=30, text=True)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def _onnx_providers(executable) -> list:
    """Execution providers the given interpreter's onnxruntime offers."""
    try:
        out = subprocess.run(
            [executable, "-I", "-c",
             "import onnxruntime as ort; "
             "print(','.join(ort.get_available_providers()))"],
            capture_output=True, timeout=60, text=True, env=child_env())
        if out.returncode != 0:
            return []
        return [p for p in out.stdout.strip().split(",") if p]
    except Exception:
        return []


def cuda_capability_hint(executable) -> str:
    """Warn when an NVIDIA GPU is present but unusable by this environment.

    The default environment installs the CPU-only 'onnxruntime' wheel
    (requirements/plugin.txt -> base.txt), so CUDA is simply not available --
    no amount of driver or CUDA toolkit fixes that, because 'onnxruntime-gpu'
    is a different package. Returns "" when there is nothing to say.

    Deliberately only a HINT: the default install is left alone rather than
    silently pulling a large CUDA dependency into every plugin environment.
    """
    if not executable or not _nvidia_gpu_present():
        return ""
    providers = _onnx_providers(executable)
    if not providers or "CUDAExecutionProvider" in providers:
        return ""
    return (
        "An NVIDIA GPU was detected, but this environment's onnxruntime is "
        "the CPU-only build, so inference will run on the CPU. To use the "
        "GPU, install onnxruntime-gpu into the WINMOL environment — see "
        "docs/GPU.md.")


def configured_python_executable():
    """The user-provided interpreter from QgsSettings, or None."""
    try:
        from qgis.core import QgsSettings
    except Exception:
        return None
    val = QgsSettings().value(QSETTINGS_PYTHON_KEY, "")
    return str(val).strip() or None


# --- sentinel (install once) -----------------------------------------------

def _file_hash(path) -> str:
    try:
        data = Path(path).read_bytes()
    except Exception:
        data = b""
    return hashlib.sha256(data).hexdigest()[:16]


def _requirements_hash() -> str:
    return _file_hash(plugin_requirements_path())


def _variant_hashes() -> dict:
    """``{'cpu': hash, 'gpu': hash}`` — the requirement files a sentinel
    may legitimately record.

    A GPU environment is installed from plugin-gpu.txt, so its marker
    carries that file's hash; comparing it only against plugin.txt would
    declare every GPU install "incomplete" on the next dialog open and
    offer to reinstall it forever.
    """
    hashes = {"cpu": _requirements_hash()}
    gpu_path = repo_requirements_dir().joinpath("plugin-gpu.txt")
    if gpu_path.exists():
        hashes["gpu"] = _file_hash(gpu_path)
    return hashes


def _marker_path(venv_path) -> str:
    return os.path.join(venv_path, READY_MARKER)


def marker_variant(venv_path):
    """``'cpu'`` / ``'gpu'`` when the sentinel matches a requirements file
    we still ship, else None. Pure file I/O — no interpreter, no
    nvidia-smi — so it stays safe on the GUI thread."""
    try:
        with open(_marker_path(venv_path)) as f:
            stored = json.load(f).get("req_hash")
    except Exception:
        return None
    for variant, digest in _variant_hashes().items():
        if stored == digest:
            return variant
    return None


def marker_matches(venv_path) -> bool:
    """True when the ``.winmol_ready`` sentinel matches the current
    requirements. Pure file I/O — no interpreter is spawned, so this is
    the part of :func:`is_ready` that is safe on a GUI thread."""
    return marker_variant(venv_path) is not None


def is_ready(venv_path, version=None) -> bool:
    """True when the venv exists, runs a supported Python, and was installed
    against the current requirements.

    ``version`` short-circuits the interpreter probe for a caller that has
    just measured it (setup_state.env_info does): re-spawning the venv's
    python to re-learn a version we were handed is pure waste, and it used
    to happen on every Setup-tab repaint.
    """
    py = get_venv_python_path(venv_path)
    if not os.path.exists(py):
        return False
    if version is None:
        version = _python_version(py)
    if not (MIN_PY <= tuple(version)[:2] <= MAX_PY):
        return False   # e.g. a stale 3.9 venv the code can't run
    return marker_matches(venv_path)


def _write_marker(venv_path, gpu=False) -> None:
    req = plugin_requirements_path(gpu=gpu)
    with open(_marker_path(venv_path), "w") as f:
        json.dump({"req_hash": _file_hash(req),
                   "requirements": str(req),
                   "variant": "gpu" if gpu else "cpu"}, f)


def invalidate_marker(venv_path) -> bool:
    """Drop the ``.winmol_ready`` sentinel so :func:`is_ready` reports the
    environment as needing a rebuild.

    Called as the first step of BOTH "Reinstall dependencies" and
    "Delete environment": ``setup_environment`` short-circuits on a valid
    marker, so a repair would otherwise silently do nothing, and a
    half-deleted venv would keep being blessed as ready.

    Returns True when a marker was actually removed.
    """
    try:
        os.remove(_marker_path(venv_path))
        return True
    except OSError:
        return False


# --- disk usage / removal ---------------------------------------------------

def directory_size(path) -> int:
    """Total size in bytes of the files under ``path`` (0 when absent).

    Symlinks are not followed and unreadable entries are skipped: this
    feeds a human-readable "frees N GB" figure, so it must never raise.
    """
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


def path_is_inside(path, root) -> bool:
    """True when ``path`` is ``root`` or a descendant of it.

    The ROOT is realpath'd, the candidate is not. That asymmetry is the
    whole point: a POSIX venv's ``bin/python`` is an ABSOLUTE SYMLINK to
    the base interpreter by construction (see ``create_venv`` — the venv
    is deliberately built without ``--copies``, because symlinks are
    mandatory on macOS), so realpath'ing the leaf resolved WINMOL's own
    managed interpreter to something like
    ``/opt/homebrew/.../bin/python3.11`` and this predicate answered
    False for every macOS and Linux install. Fallout: deletion_plan()
    classified the managed venv as "bring your own" and the Setup tab's
    "Delete environment…" silently deleted nothing, ``EnvInfo.managed``
    was False so half-built venvs were never reported, and the stale
    interpreter setting was not cleared after a deletion.

    Nothing here deletes ``path``; callers use it to ask "is this
    interpreter WINMOL's own?" and then rmtree the ROOT (which removes a
    symlink entry, never its target). So a symlink that merely LIVES
    inside the root is correctly inside, while an interpreter that lives
    elsewhere stays outside and keeps its "forget the setting" treatment.

    The realpath'd candidate is still tried as a fallback, so a path
    that reaches the root through a symlinked *directory* (``/var`` ->
    ``/private/var`` on macOS) or a Windows 8.3 short path is caught
    too — and normcase keeps the Windows case-insensitivity the original
    docstring was written for.
    """
    if not path or not root:
        return False
    b = os.path.normcase(os.path.realpath(root))
    for candidate in (os.path.abspath(path), os.path.realpath(path)):
        a = os.path.normcase(candidate)
        if a == b or a.startswith(b + os.sep):
            return True
    return False


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


def _chmod_retry(func, path):
    """rmtree error handler: clear the read-only bit (Windows venvs ship
    read-only files) and retry once; re-raise if it still fails."""
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
    """Delete WINMOL's managed artifacts. Never raises.

    Returns ``{'planned': [(path, bytes)], 'removed': [path],
    'failed': [(path, message)], 'freed_bytes': int,
    'clear_setting': bool}``. With ``dry_run=True`` the SAME dict shape
    comes back without anything being deleted — the confirmation text and
    its "frees N GB" figure are therefore produced by the exact code path
    that performs the deletion, so the two can never drift.

    ``clear_setting`` is True only when ``configured_exe`` resolves
    INSIDE a tree that is actually being removed; a bring-your-own conda
    interpreter is never touched and never un-configured. The caller (the
    dialog) performs the QgsSettings write — this module stays importable
    without QGIS.
    """
    report = _as_progress(progress)
    venv = venv_location(plugin_dir)
    runtime = os.path.join(managed_root(plugin_dir), "py311")
    models_dir = os.path.join(plugin_dir, MODELS_PATH)
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
        if not _is_managed_path(path, plugin_dir):
            result["failed"].append(
                (path, "refused: outside the managed tree"))
            continue
        if os.path.isdir(path):
            result["planned"].append((path, directory_size(path)))

    model_plan = None
    if remove_models:
        if not _is_managed_path(models_dir, plugin_dir):
            result["failed"].append(
                (models_dir, "refused: outside the managed tree"))
        else:
            model_plan = _plan_models(plugin_dir, models_dir, dry_run=True)
            if model_plan["freed_bytes"] or model_plan["planned"]:
                result["planned"].append(
                    (models_dir, model_plan["freed_bytes"]))

    if dry_run:
        result["freed_bytes"] = sum(size for _p, size in result["planned"])
        return result

    # A half-deleted venv must degrade to "needs rebuild", never stay
    # blessed by is_ready(); do this BEFORE the first rmtree.
    if remove_venv:
        invalidate_marker(venv)
        # The autotune cache describes a batch size measured against THIS
        # environment's onnxruntime; it goes with it. Bytes are negligible,
        # so it is not part of the "frees N GB" accounting.
        try:
            from . import autotune_cache
            if autotune_cache.clear(autotune_cache_location(plugin_dir)):
                report.phase("Removed the batch-size autotune cache.")
        except Exception:                              # never fatal
            pass

    for path, size in list(result["planned"]):
        if path == models_dir:
            continue
        report.phase(f"Removing {path} …")
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


def _plan_models(plugin_dir, models_dir, dry_run, progress=None):
    """Delegate model removal to the registry (it owns the naming rule,
    the ``*.part`` leftovers and the verified-digest memo)."""
    empty = {"planned": [], "removed": [], "failed": [], "freed_bytes": 0}
    try:
        from . import model_registry
        registry = model_registry.load_registry(
            os.path.join(plugin_dir, "config.json"))
    except Exception as exc:
        if not dry_run and progress is not None:
            progress(f"Could not read the model registry: {exc}")
        return empty
    return model_registry.remove_all(registry, models_dir, dry_run=dry_run)


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

def _looks_like_missing_venv_package(exc) -> bool:
    """True for Debian/Ubuntu's "install the python3-venv package" failure.

    Matched on the distro's own wording (it is what `python3 -m venv` prints
    when ensurepip is absent), so the hint is only offered when it is the
    actual cause.
    """
    text = str(exc).lower()
    return ("python3-venv" in text
            or "ensurepip is not available" in text
            or "returned non-zero exit status 1" in text and "venv" in text)


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
        # The python3-venv hint used to live ONLY on the get-pip fallback in
        # ensure_pip(), which a Debian/Ubuntu box never reaches: there the
        # `python3 -m venv` above is what fails, with "You may need to use
        # sudo ... After installing the python3-venv package". Say it here,
        # where it is actually reachable.
        raise RuntimeError(
            f"venv creation failed with {base_python}: {exc}"
            + (" On Debian/Ubuntu install the matching python3-venv package "
               "(e.g. `sudo apt install python3.11-venv`), or point WINMOL at "
               "an existing interpreter with 'Choose interpreter…'."
               if _looks_like_missing_venv_package(exc) else "")) from exc
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


def distribution_installed(python_exe, dist, timeout=60) -> bool:
    """True when ``dist`` is installed in ``python_exe``.

    Uses importlib.metadata rather than importing the package: asking
    whether ``onnxruntime-gpu`` is present by importing ``onnxruntime``
    cannot tell the two distributions apart, which is the entire problem.
    Any failure (no such interpreter, no metadata) answers False.
    """
    try:
        out = subprocess.run(
            [python_exe, "-I", "-c",
             "import importlib.metadata as m, sys;"
             "sys.exit(0 if m.distribution(sys.argv[1]) else 1)", dist],
            capture_output=True, timeout=timeout, env=child_env())
        return out.returncode == 0
    except Exception:
        return False


def uninstall_conflicting_runtime(python_exe, gpu, progress=None) -> bool:
    """Remove the runtime that must not coexist with the one we install.

    onnxruntime and onnxruntime-gpu both provide the ``onnxruntime``
    module (see the header of requirements/gpu.txt). With both installed,
    whichever wrote the files last wins and the other's dangling shared
    libraries produce import errors that read like a broken CUDA install.
    pip will not resolve this for us — the two are unrelated names.

    Returns True when something was actually removed. A failure here is
    logged, not raised: the install that follows is the thing that must
    succeed, and it will fail loudly on its own if this mattered.
    """
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
            progress=progress, label=f"pip uninstall {doomed}", timeout=600)
        return True
    except Exception as exc:
        progress(f"Could not remove {doomed}: {exc}")
        return False


def install_requirements(venv_path, progress=None, gpu=False) -> None:
    install_requirements_into(get_venv_python_path(venv_path),
                              progress=progress, gpu=gpu)


def install_requirements_into(python_exe, progress=None, gpu=False) -> None:
    """pip-install requirements/plugin.txt into an arbitrary interpreter.

    Used both for the managed venv and for an interpreter the user picked in
    the dialog. Streams pip's own output so a multi-minute install visibly
    progresses; ``--no-input`` prevents a hidden prompt that would be
    indistinguishable from a hang, and ``--progress-bar off`` stops the \\r
    bar spam that a QPlainTextEdit cannot render usefully.

    ``gpu=True`` installs requirements/plugin-gpu.txt instead — the same
    environment with onnxruntime-gpu[cuda,cudnn] (about 2.4 GB) in place of
    onnxruntime. Either way the OTHER runtime is uninstalled first; that is
    a correctness requirement, not tidiness.
    """
    progress = _as_progress(progress)
    req = str(plugin_requirements_path(gpu=gpu))
    total = len(_requirement_names(req))
    uninstall_conflicting_runtime(python_exe, gpu, progress=progress)
    progress.phase(
        f"Installing {total} packages from {os.path.basename(req)} into "
        f"{python_exe} — the first run downloads "
        + ("about 2.4 GB of CUDA libraries" if gpu
           else "a few hundred MB")
        + " and can take several minutes …")
    _run_streamed(
        [python_exe, "-u", "-m", "pip", "install", "--upgrade", "--no-input",
         "--progress-bar", "off", "-r", req],
        progress=progress, label=f"pip install -r {os.path.basename(req)}",
        timeout=3600, line_filter=_pip_line_filter(total))
    progress(f"Dependencies installed in {progress.phase_elapsed():.0f}s.")


# --- proof that the runtime we installed is the runtime that runs -----------

#: Runs in the CHILD interpreter — the one that will actually do the
#: inference. Asking the QGIS interpreter (or this one) what providers are
#: available answers about the wrong environment entirely.
#:
#: It reports the installed DISTRIBUTION names as well as the providers,
#: because "onnxruntime-gpu is installed but CUDAExecutionProvider is
#: missing" and "only the CPU package is installed" are different faults
#: with different fixes, and the provider list alone cannot tell them
#: apart. When a model file and a wanted provider are supplied it goes
#: further and builds a real InferenceSession: availability is not
#: usability, and a CUDA build with the wrong cuDNN lists the provider
#: happily and then fails at session creation.
_RUNTIME_PROBE = r"""
import json, sys
out = {"ok": False, "providers": [], "packages": [], "version": None,
       "error": None, "session_providers": None}
try:
    import importlib.metadata as md
    for dist in ("onnxruntime", "onnxruntime-gpu"):
        try:
            md.distribution(dist)
            out["packages"].append(dist)
        except Exception:
            pass
except Exception:
    pass
try:
    import onnxruntime as ort
    out["ok"] = True
    out["version"] = getattr(ort, "__version__", None)
    out["providers"] = list(ort.get_available_providers())
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    print(json.dumps(out))
    sys.exit(0)
model = sys.argv[1] if len(sys.argv) > 1 else ""
want = sys.argv[2] if len(sys.argv) > 2 else ""
if model and want and want in out["providers"]:
    try:
        sess = ort.InferenceSession(
            model, providers=[want, "CPUExecutionProvider"])
        out["session_providers"] = list(sess.get_providers())
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
"""


def _empty_runtime_report(error) -> dict:
    return {"ok": False, "providers": [], "packages": [], "version": None,
            "error": error, "session_providers": None}


def probe_runtime(python_exe, model_path=None, want=None,
                  timeout=300) -> dict:
    """What onnxruntime in ``python_exe`` can actually do.

    ``{'ok', 'providers', 'packages', 'version', 'error',
    'session_providers'}``. ``ok`` is False when onnxruntime could not
    even be imported — which is exactly what a CUDA-13 build on a CUDA-12
    driver does (``ImportError: libcudart.so.13``), so the error string is
    the real diagnosis and must reach the user verbatim.

    Spawns an interpreter: worker threads only. Never raises.
    """
    if not python_exe:
        return _empty_runtime_report("no interpreter configured")
    cmd = [python_exe, "-I", "-c", _RUNTIME_PROBE,
           str(model_path or ""), str(want or "")]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, env=child_env())
    except Exception as exc:
        return _empty_runtime_report(f"probe failed: {exc}")
    for line in reversed((out.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            report = json.loads(line)
        except ValueError:
            continue
        report.setdefault("providers", [])
        report.setdefault("packages", [])
        report.setdefault("session_providers", None)
        return report
    return _empty_runtime_report(
        (out.stderr or out.stdout or "no output").strip()[-400:])


def first_model_path(plugin_dir):
    """Any .onnx already on disk, for the session check — or None.

    A real InferenceSession is the only honest proof that CUDA works, and
    it needs a model. There is no point downloading one for the probe:
    when nothing is on disk yet, the provider list is the best available
    evidence and the report says so.
    """
    models_dir = os.path.join(plugin_dir or _PLUGIN_DIR, MODELS_PATH)
    try:
        names = sorted(os.listdir(models_dir))
    except OSError:
        return None
    for name in names:
        if name.lower().endswith(".onnx"):
            path = os.path.join(models_dir, name)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
    return None


def verify_gpu_runtime(python_exe, plugin_dir=None, progress=None) -> dict:
    """Post-install proof that the GPU runtime is a GPU runtime.

    Returns ``{'ok': bool, 'message': str, 'report': <probe_runtime>}``.
    ``ok`` is True only when CUDAExecutionProvider is available AND — when
    a model was on disk to try it with — an InferenceSession really came
    up on it. Anything else is reported with the interpreter's own error
    text: claiming GPU and delivering CPU is the failure mode this
    function exists to make impossible.
    """
    progress = _as_progress(progress)
    model = first_model_path(plugin_dir)
    progress("Verifying the GPU runtime in the compute environment …")
    report = probe_runtime(python_exe, model_path=model, want=CUDA_PROVIDER)
    ok, message = gpu_verdict(report, checked_with_model=bool(model))
    progress(message)
    return {"ok": ok, "message": message, "report": report}


def gpu_verdict(report, checked_with_model=False):
    """``(ok, message)`` from a :func:`probe_runtime` report. Pure."""
    if not report.get("ok"):
        return False, (
            "The GPU runtime was installed but onnxruntime could not be "
            f"imported: {report.get('error')}. The environment is not "
            "usable — reinstall the dependencies.")
    providers = report.get("providers") or []
    version = report.get("version") or "?"
    if CUDA_PROVIDER not in providers:
        packages = ", ".join(report.get("packages") or []) or "none"
        return False, (
            f"{CUDA_PROVIDER} is NOT available in onnxruntime {version} "
            f"(installed: {packages}; providers: {', '.join(providers)}). "
            "Detection will run on the CPU, several hundred times slower. "
            "This usually means the wheel's CUDA generation does not match "
            "the NVIDIA driver.")
    session = report.get("session_providers")
    if checked_with_model and session is not None:
        if CUDA_PROVIDER not in session:
            return False, (
                f"{CUDA_PROVIDER} is listed by onnxruntime {version} but a "
                f"real session fell back to {', '.join(session)}. "
                "Detection would run on the CPU.")
        return True, (
            f"GPU runtime verified: onnxruntime {version} created a session "
            f"on {CUDA_PROVIDER}.")
    if checked_with_model and report.get("error"):
        return False, (
            f"{CUDA_PROVIDER} is available in onnxruntime {version}, but "
            f"loading a model on it failed: {report.get('error')}")
    return True, (
        f"GPU runtime installed: onnxruntime {version} offers "
        f"{CUDA_PROVIDER}. (No model on disk yet, so no session was "
        "created — the first detection will confirm it.)")


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
            f"downloaded: {names}. Check your internet connection and open "
            "the Setup tab to retry, or Browse to select a local .onnx "
            "model file.")


def setup_environment(venv_path, base_python=None, download=True,
                      plugin_dir=None, progress=None, gpu=False) -> dict:
    """Create the venv and install deps (idempotent via the sentinel).
    Returns {'python': <exe>, 'missing_models': [...], 'gpu': bool}. Raises
    only on a real environment failure (venv/pip/deps); callers convert that
    to a retry.

    ``gpu=True`` builds the CUDA environment (requirements/plugin-gpu.txt,
    about 2.4 GB). It is passed in rather than detected here: the user has
    to be ASKED before that download starts, and the caller is the one that
    can ask. :func:`detect_gpu` is the detection half.

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
        install_requirements(venv_path, progress=report, gpu=gpu)
        _write_marker(venv_path, gpu=gpu)
    else:
        report("WINMOL environment already built; checking models …")
    missing = download_models(plugin_dir, progress=report) if download else []
    report(f"Environment ready in {report.elapsed():.0f}s: "
           f"{get_venv_python_path(venv_path)}")
    if missing:
        report(installed_message(missing))
    return {"python": get_venv_python_path(venv_path),
            "missing_models": missing, "gpu": bool(gpu)}


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
        python = get_venv_python_path(venv_path)
        message = "WINMOL environment ready."
        hint = cuda_capability_hint(python)
        if hint:
            message = f"{message} {hint}"
        result.update(status="ready", python=python, message=message)
        return result

    if not build:
        result.update(
            status="needs_setup",
            message="WINMOL environment not set up yet. Open the plugin and "
                    "use the Setup tab to create it (Python 3.11 + "
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
