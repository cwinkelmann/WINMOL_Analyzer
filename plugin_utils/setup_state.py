"""Every decision the Setup tab makes, with no Qt in sight.

The QGIS dialog owns widgets and threads; this module owns the answers.
Which button is enabled, what the environment line says, why Run is
blocked, what a deletion would remove — all of it is a pure function of
plain data here, so the whole decision surface is unit-testable on a
machine with no QGIS (the repo has none in CI).

Rules that keep it that way, and that a reviewer should enforce:

* stdlib + ``plugin_utils.installer`` only. No ``qgis``, no ``PyQt``.
* No ``tr()``. Translatable text lives here as module-level ``TXT_*``
  format templates; the DIALOG is the only place allowed to wrap one in
  ``self.tr(...)`` before ``.format(...)``. Paths, byte counts and file
  names are always named ``{placeholders}`` outside the translatable
  span, never concatenated fragments, so a translator can reorder them.
* Nothing here touches the network, hashes a file, or shells out; the
  probes (``_python_version`` / ``_has_compute_deps`` / ``is_ready``) are
  injectable so tests never spawn a subprocess.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import installer

# --- translatable templates (see the module docstring, rule 2) --------------

TXT_ENV_READY = "Ready — Python {version}"
TXT_ENV_NONE = "Not set up"
TXT_ENV_UNSUPPORTED = "Python {version} — unsupported"
TXT_ENV_DEPS_MISSING = "Dependencies missing"
TXT_ENV_INCOMPLETE = "Incomplete — reinstall dependencies"

TXT_DETAIL_MANAGED = "managed venv"
TXT_DETAIL_BYO = "your own interpreter"
TXT_DETAIL_DEPS_OK = "dependencies present"
TXT_DETAIL_DEPS_BAD = "dependencies missing"
TXT_DETAIL_DEPS_UNKNOWN = "dependencies not checked"
TXT_DETAIL_RUNTIME = "runtime {size}"

TXT_BLOCK_NO_ENV = (
    "No Python environment yet — open the Setup tab and create one.")
TXT_BLOCK_VERSION = (
    "The selected interpreter is Python {version}; WINMOL needs "
    "{want}. Choose another in Setup.")
TXT_BLOCK_DEPS = (
    "The environment is missing onnxruntime/rasterio/geopandas. "
    "Reinstall dependencies in Setup.")
TXT_BLOCK_INCOMPLETE = (
    "The environment is incomplete — reinstall dependencies in Setup.")
TXT_BLOCK_BUSY = "Setup is busy — wait for the current job to finish."

TXT_MODELS_SUMMARY = "Models on disk: {have} of {total} — {size}"

TXT_STATE_MISSING = "not downloaded"
TXT_STATE_UNPINNED = "on disk — no checksum published"
TXT_STATE_PRESENT = "on disk — not yet verified"
TXT_STATE_VERIFIED = "verified"
TXT_STATE_CORRUPT = "CORRUPT — re-download"

#: Model status -> the string shown in the tree's Status column.
STATE_TEXT = {
    "missing": TXT_STATE_MISSING,
    "unpinned": TXT_STATE_UNPINNED,
    "present": TXT_STATE_PRESENT,
    "verified": TXT_STATE_VERIFIED,
    "corrupt": TXT_STATE_CORRUPT,
}

#: States from which a (re-)download is the right action.
DOWNLOADABLE_STATES = ("missing", "corrupt")

# --- data ------------------------------------------------------------------


@dataclass
class EnvInfo:
    """A snapshot of the compute environment, probe results included."""

    exe: Optional[str]
    exists: bool
    version: Optional[Tuple[int, ...]]
    deps_ok: Optional[bool]
    managed: bool
    venv_path: str
    runtime_path: str
    marker_ok: bool


@dataclass
class ModelRow:
    """One registry model, as the Setup tree needs to render it."""

    entry_id: str
    family_id: str
    family_label: str
    label: str
    precision: str
    backend: str
    file: str
    path: str
    size_expected_mb: Optional[float]
    bytes_on_disk: int
    present: bool
    pinned: bool
    verified: bool
    hidden: bool
    recommended: bool
    state: str
    description: str = ""
    f1: Optional[float] = None
    tags: List[str] = field(default_factory=list)


# --- formatting ------------------------------------------------------------

def human_bytes(n) -> str:
    """A short size string: '0 bytes', '63 MB', '1.9 GB'.

    Deliberately NOT translated (rule 3 in the module docstring): it is
    substituted into ``{size}`` placeholders, and a translator reordering
    a number and its unit is a bug, not a feature.
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "0 bytes"
    if n < 0:
        n = 0
    if n < 1024:
        return f"{n} bytes"
    if n < 1024 ** 2:
        return f"{n / 1024:.0f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024 ** 3:.1f} GB"


def version_text(version) -> str:
    """'3.11' / '3.11.15' from a version tuple ('?' when unknown)."""
    if not version:
        return "?"
    return ".".join(str(part) for part in version)


def _supported(version) -> bool:
    if not version:
        return False
    pair = tuple(version[:2])
    return installer.MIN_PY <= pair <= installer.MAX_PY


def _wanted_version_text() -> str:
    low = version_text(installer.MIN_PY)
    high = version_text(installer.MAX_PY)
    return low if low == high else f"{low}-{high}"


# --- environment -----------------------------------------------------------

def env_info(plugin_dir, configured_exe, version_fn=None, deps_fn=None,
             ready_fn=None) -> EnvInfo:
    """Probe the compute environment.

    ``configured_exe`` is what the dialog is currently pointed at (the
    QgsSettings interpreter, or the managed venv's python). The three
    ``*_fn`` seams default to the installer's own probes and exist so
    tests can describe a machine instead of owning one; they are resolved
    at CALL time, so monkeypatching ``installer._python_version`` works.
    """
    version_fn = version_fn or installer._python_version
    deps_fn = deps_fn or installer._has_compute_deps
    ready_fn = ready_fn or installer.is_ready

    venv_path = installer.venv_location(plugin_dir)
    runtime_path = os.path.join(installer.managed_root(plugin_dir), "py311")
    exe = (configured_exe or "").strip() or None
    if exe is None:
        managed_exe = installer.get_venv_python_path(venv_path)
        if os.path.exists(managed_exe):
            exe = managed_exe
    exists = bool(exe) and os.path.exists(exe)
    return EnvInfo(
        exe=exe,
        exists=exists,
        version=tuple(version_fn(exe)) if exists else None,
        deps_ok=bool(deps_fn(exe)) if exists else None,
        managed=bool(exe) and installer.path_is_inside(exe, venv_path),
        venv_path=venv_path,
        runtime_path=runtime_path,
        marker_ok=bool(ready_fn(venv_path)),
    )


def env_state_text(info) -> str:
    """The bold one-liner at the top of Step 1."""
    if not info.exists:
        return TXT_ENV_NONE
    if not _supported(info.version):
        return TXT_ENV_UNSUPPORTED.format(version=version_text(info.version))
    if info.deps_ok is False:
        return TXT_ENV_DEPS_MISSING
    if info.managed and not info.marker_ok:
        return TXT_ENV_INCOMPLETE
    return TXT_ENV_READY.format(version=version_text(info.version))


def env_detail_text(info, usage=None) -> str:
    """The gray second line: what kind of environment, and what it costs.

    ``usage`` is an optional ``{'venv': bytes, 'runtime': bytes}`` map —
    the dialog fills it from ``installer.directory_size``.
    """
    usage = usage or {}
    parts = [TXT_DETAIL_MANAGED if info.managed else TXT_DETAIL_BYO]
    if info.deps_ok is None:
        parts.append(TXT_DETAIL_DEPS_UNKNOWN)
    else:
        parts.append(TXT_DETAIL_DEPS_OK if info.deps_ok
                     else TXT_DETAIL_DEPS_BAD)
    if usage.get("venv"):
        parts.append(human_bytes(usage["venv"]))
    if usage.get("runtime"):
        parts.append(TXT_DETAIL_RUNTIME.format(
            size=human_bytes(usage["runtime"])))
    return " · ".join(parts)


def env_ready(info) -> bool:
    """True when a detection could run right now."""
    if not info.exists or not _supported(info.version):
        return False
    if info.deps_ok is False:
        return False
    if info.managed and not info.marker_ok:
        return False
    return True


def blocking_reason(info, busy_kind=None):
    """Why Run is disabled, or None when it is not.

    One source of truth for three surfaces: the orange banner on the
    detection tab, the disabled Run button's tooltip, and which tab the
    dialog opens on.

    A MISSING MODEL is deliberately never a blocking reason — it is
    recoverable at Run time by the existing download pre-flight, and
    blocking on it would strand a user who never opened the Setup tab.
    """
    if busy_kind in ("env", "download"):
        return TXT_BLOCK_BUSY
    if not info.exists:
        return TXT_BLOCK_NO_ENV
    if not _supported(info.version):
        return TXT_BLOCK_VERSION.format(version=version_text(info.version),
                                        want=_wanted_version_text())
    if info.deps_ok is False:
        return TXT_BLOCK_DEPS
    if info.managed and not info.marker_ok:
        return TXT_BLOCK_INCOMPLETE
    return None


# --- models ----------------------------------------------------------------

def models_summary_text(rows) -> str:
    """'Models on disk: 2 of 22 — 156 MB' over the VISIBLE rows."""
    visible = [row for row in rows if not row.hidden]
    present = [row for row in visible if row.present]
    return TXT_MODELS_SUMMARY.format(
        have=len(present), total=len(visible),
        size=human_bytes(sum(row.bytes_on_disk for row in present)))


def state_text(row) -> str:
    return STATE_TEXT.get(row.state, row.state)


def find_row(rows, entry_id):
    """The row for ``entry_id``, or None."""
    if not entry_id:
        return None
    for row in rows:
        if row.entry_id == entry_id:
            return row
    return None


# --- the interlock ---------------------------------------------------------

def button_states(info, rows, selected_entry_id=None, run_active=False,
                  env_busy=False, dl_busy=False) -> dict:
    """The enable/disable truth table for the Setup tab, keyed by widget
    object name.

    THE single authority: nothing else in the dialog may call
    ``setEnabled`` on a setup button. While anything is running —
    a detection, an environment job or a download — every one of them is
    off, which is what makes "one job at a time" true rather than
    hoped-for.
    """
    busy = bool(run_active or env_busy or dl_busy)
    row = find_row(rows, selected_entry_id)
    downloadable = bool(row is not None
                        and row.state in DOWNLOADABLE_STATES)
    recommended_missing = any(
        r.recommended and r.state in DOWNLOADABLE_STATES for r in rows)
    states = {
        "env_create_button": not env_ready(info),
        "env_choose_button": True,
        "env_repair_button": True,
        "env_delete_button": bool(info.exe) or info.marker_ok,
        "models_refresh_button": True,
        "models_download_button": downloadable,
        "models_download_default_button": recommended_missing,
        "models_verify_button": bool(
            row is not None and row.pinned and row.present),
        "models_delete_button": bool(row is not None and row.present),
        "run_button": env_ready(info),
    }
    if busy:
        return {name: False for name in states}
    return states


# --- deletion --------------------------------------------------------------

def deletion_plan(plugin_dir, configured_exe=None, remove_venv=True,
                  remove_runtime=False, remove_models=False) -> dict:
    """What "Delete environment…" would actually do.

    Returns ``{'kind': 'managed'|'byo'|'none', 'paths': [...],
    'clears_setting': bool}``. ``byo`` means the configured interpreter
    lives outside WINMOL's managed tree: nothing on disk is offered for
    deletion and the only action available is forgetting the setting.
    """
    venv = installer.venv_location(plugin_dir)
    runtime = os.path.join(installer.managed_root(plugin_dir), "py311")
    models = os.path.join(plugin_dir, installer.MODELS_PATH)
    exe = (configured_exe or "").strip() or None

    if exe and not installer.path_is_inside(exe, venv):
        return {"kind": "byo", "paths": [], "clears_setting": False}

    paths = []
    if remove_venv and os.path.isdir(venv):
        paths.append(venv)
    if remove_runtime and os.path.isdir(runtime):
        paths.append(runtime)
    if remove_models and os.path.isdir(models):
        paths.append(models)
    if not paths and not exe:
        return {"kind": "none", "paths": [], "clears_setting": False}
    return {"kind": "managed", "paths": paths,
            "clears_setting": bool(exe and remove_venv
                                   and installer.path_is_inside(exe, venv))}
