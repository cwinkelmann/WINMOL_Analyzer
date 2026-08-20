"""Every decision the Setup tab makes, with no Qt in sight.

The QGIS dialog owns widgets and threads; this module owns the answers.
What the environment line says, why Run is blocked, which button is
enabled, what a deletion would remove — all of it is a pure function of
plain data here, so the whole decision surface is unit-testable on a
machine with no QGIS (the repo has none in CI).

Rules that keep it that way, and that a reviewer should enforce:

* stdlib + ``plugin_utils.installer`` only. No ``qgis``, no ``PyQt``.
* No ``tr()``. Translatable text lives here as module-level ``TXT_*``
  format templates; the DIALOG is the only place allowed to wrap one in
  ``self.tr(...)`` before ``.format(...)``. Paths, byte counts and file
  names are always named ``{placeholders}``, never concatenated
  fragments, so a translator can reorder them.
* Nothing here touches the network or shells out on its own;
  :func:`env_info` delegates to ``installer.resolve_environment``
  (which probes interpreters), so it must only run on a worker thread.
"""

import os
from dataclasses import dataclass
from typing import Optional

from . import installer

# --- translatable templates (see the module docstring, rule 2) --------------

TXT_ENV_READY_MANAGED = "Ready — managed environment ({variant} runtime)"
TXT_ENV_READY_BYO = "Ready — your own interpreter"
TXT_ENV_NONE = "Not set up"
TXT_ENV_ERROR = "Environment problem"

TXT_DETAIL_MANAGED = "managed venv"
TXT_DETAIL_BYO = "your own interpreter"

#: Where the managed environment lives, and the honest caveat about
#: what uninstalling the plugin does NOT do: QGIS's plugin uninstall
#: only deletes the plugin folder, and the managed root is deliberately
#: a sibling of it (installer.managed_root) so a recursive delete
#: cannot trip over venv symlinks. There is no uninstall hook in the
#: QGIS plugin API, so this leftover must be discoverable by hand.
TXT_ENV_LOCATION = (
    "WINMOL's environment lives in {path}{size}. Uninstalling the "
    "plugin does NOT remove it — QGIS only deletes the plugin folder. "
    "Use “Delete environment…” here before uninstalling, or delete "
    "that folder yourself afterwards.")

TXT_BLOCK_NO_ENV = (
    "No Python environment yet — open the Setup tab and create one.")
TXT_BLOCK_BUSY = "Setup is busy — wait for the current job to finish."

TXT_MODELS_SUMMARY = "Models on disk: {have} of {total} — {size}"


# --- data -------------------------------------------------------------------

@dataclass
class EnvInfo:
    """A snapshot of the compute environment, as the Setup tab needs it.

    ``status`` is ``installer.resolve_environment``'s verdict —
    ``byo`` | ``ready`` | ``needs_setup`` | ``error``.
    """

    status: str
    python: Optional[str]
    venv_path: str
    managed: bool
    variant: Optional[str]      # 'cpu' | 'gpu' from the sentinel
    venv_bytes: int
    message: str


# --- formatting -------------------------------------------------------------

def human_bytes(n) -> str:
    """A short size string: '0 bytes', '63 MB', '1.9 GB'. Deliberately
    NOT translated: it fills ``{size}`` placeholders, and a translator
    reordering a number and its unit is a bug, not a feature."""
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


# --- environment ------------------------------------------------------------

def env_info(plugin_dir, configured_exe=None, resolve_fn=None) -> EnvInfo:
    """Probe the compute environment. NEVER call this on the GUI
    thread — ``resolve_environment`` spawns interpreters and is
    measured in seconds. ``resolve_fn`` is injectable so tests can
    describe a machine instead of owning one."""
    resolve_fn = resolve_fn or installer.resolve_environment
    env = resolve_fn(plugin_dir) or {}
    venv_path = env.get("venv_path") or installer.venv_location(plugin_dir)
    exe = env.get("python") or (configured_exe or "").strip() or None
    return EnvInfo(
        status=env.get("status") or "needs_setup",
        python=exe,
        venv_path=venv_path,
        managed=bool(exe) and installer.path_is_inside(exe, venv_path),
        variant=installer.installed_variant(venv_path),
        venv_bytes=installer.directory_size(venv_path),
        message=env.get("message") or "",
    )


def env_ready(info) -> bool:
    """True when a detection could run right now."""
    return info.status in ("byo", "ready")


def env_state_text(info) -> str:
    """The bold one-liner at the top of the Setup tab's Step 1."""
    if info.status == "error":
        return TXT_ENV_ERROR
    if not env_ready(info):
        return TXT_ENV_NONE
    if info.managed:
        return TXT_ENV_READY_MANAGED.format(
            variant=(info.variant or "cpu").upper())
    return TXT_ENV_READY_BYO


def env_detail_text(info) -> str:
    """The gray second line: what kind of environment, where, and what
    it costs on disk."""
    parts = [TXT_DETAIL_MANAGED if info.managed else TXT_DETAIL_BYO]
    if info.python:
        parts.append(info.python)
    if info.venv_bytes:
        parts.append(human_bytes(info.venv_bytes))
    return " · ".join(parts)


def env_location_text(plugin_dir, venv_bytes=None) -> str:
    """The gray line naming the managed root, and saying plainly that
    uninstalling the plugin leaves it behind (QGIS has no uninstall
    hook; see TXT_ENV_LOCATION)."""
    root = installer.managed_root(plugin_dir)
    if not root:
        return ""
    size = f" ({human_bytes(venv_bytes)})" if venv_bytes else ""
    return TXT_ENV_LOCATION.format(path=root, size=size)


def blocking_reason(info, busy=False):
    """Why Run is disabled, or None when it is not. One source of
    truth for the banner, the disabled Run button's tooltip, and which
    tab the dialog opens on. A missing MODEL is deliberately never a
    blocking reason — the Run pre-flight downloads it."""
    if busy:
        return TXT_BLOCK_BUSY
    if info.status == "error":
        return info.message or TXT_ENV_ERROR
    if not env_ready(info):
        return TXT_BLOCK_NO_ENV
    return None


# --- the proactive pre-run offer (lite) -------------------------------------
#
# rr6 decided this from a probed AcceleratorStatus (EnvProbeWorker — a
# documented cut here). The lite decision uses what the dialog already
# holds: the sentinel's installed variant and the cached nvidia-smi
# probe. Pure functions of plain data, so the whole decision surface is
# testable off QGIS; nothing here ever measures anything.

#: The short idle-GPU line the DETECTION tab carries, so the offer is
#: not hidden on a tab the user never opens.
TXT_NUDGE_GPU_IDLE = (
    "{gpu} is sitting idle — the installed runtime is CPU-only, so "
    "detection takes roughly 5 s per image tile instead of 12 ms.")

#: The pre-run interruption. This is the one that matters: a user who
#: never opens the Setup tab otherwise learns about the idle GPU only
#: after waiting out a run that took hours instead of minutes.
TXT_PRERUN_TITLE = "This run will use the CPU, not your GPU"
TXT_PRERUN_GPU_IDLE = (
    "{gpu} is in this machine, but WINMOL's environment has the CPU-only "
    "inference runtime, so this detection will run on the CPU.\n\n"
    "Measured on this hardware: about 12 ms per image tile on the GPU "
    "against roughly 5 s on the CPU. That is the difference between a run "
    "of minutes and a run of hours.\n\n"
    "Installing the GPU runtime downloads about 2.4 GB once. Nothing else "
    "about the environment changes, and this detection is not started "
    "until the install finishes — press Run again afterwards.\n\n"
    "Running on the CPU is a fine answer, and it is remembered: this "
    "question is not asked again. The Setup tab keeps an “Install GPU "
    "runtime” button for whenever you change your mind.")
TXT_PRERUN_INSTALL = "Install the GPU runtime (2.4 GB)"
TXT_PRERUN_RUN_ANYWAY = "Run on the CPU anyway"

#: pre_run_decision results.
PRERUN_RUN_CPU = "run_cpu"
PRERUN_OFFER = "offer"

#: Substrings marking a run failure as a GPU/accelerator DEVICE failure -- the
#: model executed on the GPU but the GPU stack (driver / cuDNN / cuBLAS) could
#: not run it (issue #24: "CUDNN_BACKEND_API_FAILED" on an older card). NOT
#: out-of-memory (a capacity problem the prediction path already absorbs by
#: shrinking the micro-batch) -- a "this GPU cannot run the model" problem whose
#: remedy is to fall back to the CPU.
_GPU_FAILURE_MARKERS = (
    "cudnn", "cublas", "cufft", "curand", "cusparse",
    "cuda error", "cudaerror", "cuda_error",
)


def looks_like_gpu_failure(text) -> bool:
    """True if a failed run's output points at a GPU/accelerator device
    failure for which retrying on the CPU is the remedy (issue #24).
    Out-of-memory is excluded on purpose: the prediction path already handles
    it by halving the batch, so a smaller batch -- not the CPU -- is the fix."""
    low = str(text).lower()
    if "out of memory" in low or "failed to allocate memory" in low:
        return False
    return any(marker in low for marker in _GPU_FAILURE_MARKERS)


def accelerator_token(gpu_label) -> str:
    """The value persisted when the user chooses "run on the CPU
    anyway". State plus GPU name rather than a bare "yes, dismissed":
    a dismissal is an answer about THIS machine in THIS configuration —
    drop a different card in and the question is worth asking once
    more."""
    return f"gpu_idle|{gpu_label or 'An NVIDIA GPU'}"


def should_nudge(installed_variant, gpu_present) -> bool:
    """True when a surface outside the Setup tab should say something:
    the sentinel records a CPU-only install AND an NVIDIA GPU answered
    the probe. Anything else — no GPU, the GPU runtime already
    installed, no managed sentinel at all (variant None) — stays
    silent: there is either nothing to offer or nothing to install
    into."""
    return bool(gpu_present) and installed_variant == "cpu"


def accel_nudge_text(gpu_label) -> str:
    """The Detection tab's one-line idle-GPU warning."""
    return TXT_NUDGE_GPU_IDLE.format(gpu=gpu_label or "An NVIDIA GPU")


def pre_run_decision(installed_variant, gpu_present, dismissed_token,
                     machine_token) -> str:
    """What pressing Run should do about an idle GPU.

    Returns :data:`PRERUN_OFFER` (put the install-vs-CPU question on
    screen) or :data:`PRERUN_RUN_CPU` (just run). Offers only when the
    installed runtime is CPU-only, a GPU is present, and
    ``machine_token`` (:func:`accelerator_token` for this machine) has
    not already been dismissed — an answer is an answer, once per
    machine/configuration; a DIFFERENT stored token means the hardware
    changed and the question is worth asking once more."""
    if not should_nudge(installed_variant, gpu_present):
        return PRERUN_RUN_CPU
    if dismissed_token and str(dismissed_token) == str(machine_token):
        return PRERUN_RUN_CPU
    return PRERUN_OFFER


# --- models -----------------------------------------------------------------

def models_summary_text(rows) -> str:
    """'Models on disk: 1 of 2 — 17 bytes' over ``model_status`` rows."""
    present = [row for row in rows if row.installed]
    return TXT_MODELS_SUMMARY.format(
        have=len(present), total=len(rows),
        size=human_bytes(sum(row.bytes_on_disk for row in present)))


def find_row(rows, entry_id):
    """The row for ``entry_id``, or None."""
    if not entry_id:
        return None
    for row in rows:
        if row.entry_id == entry_id:
            return row
    return None


# --- the interlock ----------------------------------------------------------

def button_states(info, rows, selected_entry_id=None, busy=False) -> dict:
    """The enable/disable truth table for the Setup tab, keyed by
    widget object name. THE single authority: nothing else in the
    dialog may call ``setEnabled`` on a setup button. While any job is
    running everything is off, which is what makes "one job at a time"
    true rather than hoped-for."""
    row = find_row(rows, selected_entry_id)
    states = {
        "env_create_button": not env_ready(info),
        "env_choose_button": True,
        # A configured bring-your-own interpreter keeps this button
        # live even with no managed venv on disk: it is the only way
        # to reach the "Forget this interpreter" offer.
        "env_delete_button": bool(
            info.managed or info.venv_bytes
            or (info.python and not info.managed)),
        "models_refresh_button": True,
        "models_download_button": bool(
            row is not None and not row.installed),
        "models_download_default_button": any(
            r.is_default and not r.installed for r in rows),
        "models_verify_button": bool(
            row is not None and row.installed and row.pinned),
        "models_delete_button": bool(row is not None and row.installed),
        "run_button": env_ready(info),
    }
    if busy:
        return {name: False for name in states}
    return states


# --- deletion ---------------------------------------------------------------

def deletion_plan(plugin_dir, configured_exe=None, remove_venv=True,
                  remove_runtime=False, remove_models=False) -> dict:
    """What "Delete environment…" would actually do.

    Returns ``{'kind': 'managed'|'byo'|'none', 'paths': [...],
    'clears_setting': bool}``. ``byo`` means the configured interpreter
    lives outside WINMOL's managed tree: nothing on disk is offered
    for deletion and the only action available is forgetting the
    setting."""
    venv = installer.venv_location(plugin_dir)
    runtime = os.path.join(installer.managed_root(plugin_dir), "py311")
    models = installer.models_location(plugin_dir)
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
            "clears_setting": bool(exe and remove_venv)}
