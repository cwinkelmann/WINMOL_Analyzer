"""Persist the prediction batch-size autotune result once per device+model.

Why
---
``_autotune_batch_size`` (utils/Prediction.py) times every candidate
micro-batch before the first tile is written. On hardware where a distinct
batch size forces a session/kernel recompile (notably Apple/CoreML), that
sweep is a real, user-visible stall on every single run for a few percent of
steady-state throughput. Cached, the trade flips: the sweep runs ONCE per
(hardware, model, execution provider, tile geometry), the answer is
persisted, and every run after that is free. This module is the cache.

Design constraints
-------------------
* **Stdlib only, no Qt/QGIS** -- it must import inside the TensorFlow-free
  compute venv (``requirements/cpu.txt``) *and* inside QGIS's own Python,
  the same contract as ``installer.py`` / ``model_registry.py``.
* **Never fatal.** A missing, truncated, wrong-version or hostile cache file
  degrades to "no cached entry" and a failed write degrades to a logged
  no-op. A cache is an optimisation; it may never break a run.
* **Atomic writes** (tmp + ``os.replace``) because ``winmol_batch.py
  --jobs`` runs several prediction processes concurrently. Concurrent
  writers are last-writer-wins on a per-file basis; the file is never left
  half-written.

Location
--------
``$WINMOL_AUTOTUNE_CACHE`` when set -- the QGIS plugin sets it to
``<managed_root>/autotune.json`` (see ``plugin_utils.installer
.autotune_cache_location``) so the cache lives with the rest of WINMOL's
managed state. Otherwise a per-user cache dir (``~/Library/Caches/winmol``
on macOS, ``%LOCALAPPDATA%\\winmol`` on Windows, ``$XDG_CACHE_HOME/winmol``
elsewhere). Deleting the file is always safe and simply forces a re-tune.
"""

import hashlib
import json
import os
import platform
import sys
import tempfile
import time

#: Bumped when the stored payload shape changes -- or when the *meaning* of
#: a stored batch size changes; older files are ignored either way. Kept in
#: step with the rr6 lineage's cache format so a pre-existing
#: ``autotune.json`` (same $WINMOL_AUTOTUNE_CACHE location, an rr6-based
#: install) stays valid rather than being silently invalidated by this
#: reimplementation.
SCHEMA_VERSION = 2

#: Overrides the cache location for both the plugin and the batch CLI.
ENV_CACHE_PATH = "WINMOL_AUTOTUNE_CACHE"

#: off | auto | force. Overrides ``Config.prediction_batch_autotune``.
ENV_MODE = "WINMOL_BATCH_AUTOTUNE"

CACHE_FILENAME = "autotune.json"

#: Anything outside this is treated as a corrupt entry.
MAX_SANE_BATCH = 1024

_OFF = ("off", "false", "0", "no", "none", "never", "disabled")
_FORCE = ("force", "true", "1", "yes", "on", "always", "retune")
_AUTO = ("auto", "cached", "once")


# --- mode resolution ---------------------------------------------------

def resolve_mode(config=None):
    """Return ``"off" | "auto" | "force"``.

    ``$WINMOL_BATCH_AUTOTUNE`` wins over ``Config.prediction_batch_autotune``
    so benchmarks and CI can pin the behaviour without editing config. The
    legacy booleans keep their old meaning: ``False`` never tunes, ``True``
    tunes on every run (and refreshes the cache).
    """
    raw = os.environ.get(ENV_MODE)
    if raw is None or not str(raw).strip():
        raw = getattr(config, "prediction_batch_autotune", "auto")
    if raw is True:
        return "force"
    if raw is False or raw is None:
        return "off"
    text = str(raw).strip().lower()
    if text in _OFF:
        return "off"
    if text in _FORCE:
        return "force"
    if text in _AUTO:
        return "auto"
    return "auto"          # unknown value: the safe, self-healing default


# --- location ------------------------------------------------------------

def default_cache_dir():
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Caches",
                            "winmol")
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local")
        return os.path.join(base, "winmol")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, "winmol")


def cache_path():
    """Absolute path of the cache file (it need not exist)."""
    override = os.environ.get(ENV_CACHE_PATH)
    if override and override.strip():
        return os.path.abspath(os.path.expanduser(override.strip()))
    return os.path.join(default_cache_dir(), CACHE_FILENAME)


# --- key -------------------------------------------------------------------

def _model_identity(model):
    """Cheap identity for the model file: name + size + mtime.

    Deliberately NOT a content hash -- the ONNX models are large enough
    that hashing one costs more than the autotune it would save. Size is
    paired with mtime so a re-download or a different model of the same
    length still changes the key; the failure mode of the pair (an
    in-place edit that preserves both) does not occur for downloaded
    artifacts.
    """
    path = getattr(model, "model_path", None) or getattr(model, "path", None)
    if not path:
        return {"name": type(model).__name__}
    try:
        stat = os.stat(path)
        return {"name": os.path.basename(str(path)),
                "size": int(stat.st_size),
                "mtime": int(stat.st_mtime)}
    except OSError:
        return {"name": os.path.basename(str(path))}


def _hardware_identity(hardware):
    ident = {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 0,
    }
    if hardware is not None:
        names = list(getattr(hardware, "gpu_names", None) or [])
        memory = list(getattr(hardware, "gpu_memory_gb", None) or [])
        ident["gpus"] = sorted(str(n) for n in names)
        ident["gpu_memory_gb"] = [round(float(m), 1) for m in memory]
    return ident


def cache_key(model, config, hardware=None):
    """A stable hex digest of everything the optimum batch size depends on."""
    providers = getattr(model, "providers", None)
    payload = {
        "schema": SCHEMA_VERSION,
        "providers": [str(p) for p in (providers or [])],
        "model": _model_identity(model),
        "hardware": _hardware_identity(hardware),
        "tile": [int(getattr(config, "img_width", 0) or 0),
                 int(getattr(config, "img_height", 0) or 0)],
        "channels": int(getattr(config, "n_channels", 0) or 0),
        "max_batch": int(
            getattr(config, "prediction_batch_max_gpu", 0) or 0),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


# --- io ----------------------------------------------------------------

def _read_all(path):
    """The whole cache as a dict of entries; ``{}`` for anything unusable."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if data.get("version") != SCHEMA_VERSION:
        return {}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        return {}
    return entries


def load(key, path=None):
    """The cached batch size for ``key``, or ``None``.

    Returns ``None`` -- never raises -- for a missing file, unparseable
    JSON, a schema-version mismatch, a non-dict payload, or a batch that is
    not a plain positive int within :data:`MAX_SANE_BATCH`.
    """
    entry = _read_all(path or cache_path()).get(key)
    if not isinstance(entry, dict):
        return None
    batch = entry.get("batch")
    if isinstance(batch, bool) or not isinstance(batch, int):
        return None
    if not (1 <= batch <= MAX_SANE_BATCH):
        return None
    return batch


def store(key, batch, meta=None, path=None):
    """Persist ``batch`` for ``key``. Returns True on success, else False.

    A failure to write (read-only home, container without a writable HOME,
    a race with another process) is a no-op, not an error: the next run
    simply re-tunes.
    """
    if isinstance(batch, bool):
        # Checked BEFORE the int() coercion below: int(True) == 1 is a
        # plain int, so this guard would never fire afterwards -- bool is
        # a subclass of int in Python and a stray True/False must still be
        # rejected, not silently stored as batch 1/0.
        return False
    try:
        batch = int(batch)
    except (TypeError, ValueError):
        return False
    if not (1 <= batch <= MAX_SANE_BATCH):
        return False

    path = path or cache_path()
    entries = _read_all(path)
    entry = {"batch": batch, "stored_at": int(time.time())}
    if isinstance(meta, dict):
        entry["meta"] = meta
    entries[key] = entry
    payload = {"version": SCHEMA_VERSION, "entries": entries}

    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".autotune-",
            suffix=".tmp", delete=False)
        try:
            with handle:
                json.dump(payload, handle, sort_keys=True, indent=1)
            os.replace(handle.name, path)     # atomic; concurrent-writer safe
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise
    except OSError:
        return False
    return True


def clear(path=None):
    """Delete the cache file. Returns True if a file was removed."""
    path = path or cache_path()
    try:
        os.unlink(path)
        return True
    except OSError:
        return False
