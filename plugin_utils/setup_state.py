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
TXT_ENV_CHECKING = "Checking the environment…"
TXT_ENV_UNSUPPORTED = "Python {version} — unsupported"
TXT_ENV_DEPS_MISSING = "Dependencies missing"
TXT_ENV_INCOMPLETE = "Incomplete — reinstall dependencies"

TXT_DETAIL_MANAGED = "managed venv"
TXT_DETAIL_BYO = "your own interpreter"
TXT_DETAIL_DEPS_OK = "dependencies present"
TXT_DETAIL_DEPS_BAD = "dependencies missing"
TXT_DETAIL_DEPS_UNKNOWN = "dependencies not checked"
TXT_DETAIL_RUNTIME = "runtime {size}"

#: Where the managed environment lives, and the honest caveat about what
#: uninstalling the plugin does NOT do. QGIS's plugin uninstall is
#: unloadPlugin() + removeDir(<profile>/python/plugins/WINMOL_Analyzer)
#: and nothing else (pyplugin_installer/installer.py::uninstallPlugin),
#: and the managed root is deliberately a SIBLING of the plugin folder
#: (installer.managed_root) so that recursive delete cannot trip over
#: venv symlinks. There is no uninstall hook in the QGIS plugin API, so
#: this leftover has to be discoverable and removable by hand.
TXT_ENV_LOCATION = (
    "WINMOL's environment lives in {path}{size}. Uninstalling the plugin "
    "does NOT remove it — QGIS only deletes the plugin folder. Use "
    "“Delete environment…” here before uninstalling, or "
    "delete that folder yourself afterwards.")

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

# --- accelerator ------------------------------------------------------------
#
# The numbers below are measured, not estimated: an RTX 4080 SUPER running
# the 512x512 Spruce_Deadwood fp16 model took 12.4 ms per tile on
# CUDAExecutionProvider and 5265.7 ms on CPUExecutionProvider — a factor of
# 423. They appear in user-facing text because "GPU recommended" is advice
# nobody acts on and "5 s a tile instead of 12 ms" is not.

TXT_ACCEL_UNKNOWN = "Checking which processor will run the detection…"
TXT_ACCEL_CPU_ONLY = (
    "CPU only — no usable NVIDIA GPU on this machine. Detection will be "
    "slow: roughly 5 s per image tile, against 12 ms on a GPU.")
TXT_ACCEL_COREML = (
    "{gpu} — detection runs on the Apple GPU (CoreML). No extra download "
    "is needed.")
TXT_ACCEL_GPU_IDLE = (
    "{gpu} found, but the installed runtime is CPU-only, so the GPU is "
    "sitting idle. Detection takes roughly 5 s per image tile instead of "
    "12 ms. Install the GPU runtime to use it.")
TXT_ACCEL_GPU_NO_ENV = (
    "{gpu} found. Create the environment with the GPU runtime and "
    "detection runs at about 12 ms per image tile instead of 5 s.")
TXT_ACCEL_GPU_ACTIVE = (
    "{gpu} — the CUDA runtime is installed and active. Detection runs on "
    "the GPU (about 12 ms per image tile).")
TXT_ACCEL_GPU_UNVERIFIED = (
    "{gpu} found and the CUDA runtime is installed, but nothing has run on "
    "it yet — there is no model on disk to try it with. The first detection "
    "will say whether the GPU is really being used.")
TXT_ACCEL_GPU_BROKEN = (
    "{gpu} found and the GPU runtime is installed, but CUDA is not usable, "
    "so detection still runs on the CPU. {reason}")
TXT_ACCEL_GPU_UNUSABLE = (
    "{gpu} found, but WINMOL cannot use it here: {reason} Detection runs "
    "on the CPU, roughly 5 s per image tile.")

#: The short line the DETECTION tab carries, so the offer is not hidden
#: on a tab the user never opens. Deliberately one sentence: it sits above
#: the input fields, next to a button that opens Setup.
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

TXT_GPU_OFFER = (
    "{gpu} detected.\n\nInstall the GPU runtime (about 2.4 GB to "
    "download)?\n\nDetection is several hundred times faster on it — "
    "measured at 12 ms per image tile against 5.3 s on the CPU. Nothing "
    "else about the environment changes, and you can switch back later "
    "with 'Reinstall dependencies'.")
TXT_GPU_OFFER_TITLE = "Install the GPU runtime?"

#: The three states the Setup tab must be able to say out loud, plus the
#: honest edge cases. ``can_install`` is the only one that offers the
#: one-click action.
ACCEL_UNKNOWN = "unknown"
ACCEL_CPU_ONLY = "cpu_only"
ACCEL_COREML = "coreml"
ACCEL_GPU_IDLE = "gpu_idle"
ACCEL_GPU_ACTIVE = "gpu_active"
ACCEL_GPU_BROKEN = "gpu_broken"
ACCEL_GPU_UNUSABLE = "gpu_unusable"

# --- data ------------------------------------------------------------------


@dataclass
class EnvInfo:
    """A snapshot of the compute environment, probe results included.

    ``probed`` distinguishes a measured snapshot (:func:`env_info`, which
    spawns interpreters and therefore only ever runs on a worker thread)
    from the cheap :func:`env_seed` the dialog paints with while that
    worker is still running. An unprobed snapshot is read OPTIMISTICALLY:
    an interpreter that exists is assumed usable until the probe says
    otherwise, because the alternative — claiming "unsupported Python"
    for a second on every open — is a lie the user would act on.
    """

    exe: Optional[str]
    exists: bool
    version: Optional[Tuple[int, ...]]
    deps_ok: Optional[bool]
    managed: bool
    venv_path: str
    runtime_path: str
    marker_ok: bool
    probed: bool = True


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

def _env_paths(plugin_dir, configured_exe):
    """(venv_path, runtime_path, exe, exists) — stat() only."""
    venv_path = installer.venv_location(plugin_dir)
    runtime_path = os.path.join(installer.managed_root(plugin_dir), "py311")
    exe = (configured_exe or "").strip() or None
    if exe is None:
        managed_exe = installer.get_venv_python_path(venv_path)
        if os.path.exists(managed_exe):
            exe = managed_exe
    return venv_path, runtime_path, exe, bool(exe) and os.path.exists(exe)


def env_seed(plugin_dir, configured_exe, env=None) -> EnvInfo:
    """A probe-free EnvInfo for the FIRST paint.

    :func:`env_info` spawns up to three interpreters and is measured in
    seconds; running it while the dialog is being constructed is what put
    a multi-second freeze on every open. This builds the same record out
    of stat() calls and one small JSON read, marks it ``probed=False``,
    and leaves the dialog to refine it from a worker thread.

    ``env`` is the dict ``installer.resolve_environment`` already produced
    at plugin load: when it describes the same interpreter and reports it
    usable, its dependency verdict is carried over so a configured user's
    detail line does not flicker through "dependencies not checked".
    """
    venv_path, runtime_path, exe, exists = _env_paths(plugin_dir,
                                                      configured_exe)
    env = env or {}
    deps_ok = None
    resolved = (env.get("python") or "").strip()
    if exists and resolved and env.get("status") in (
            "byo", "ready", "installed"):
        if os.path.normcase(resolved) == os.path.normcase(exe):
            deps_ok = True
    return EnvInfo(
        exe=exe,
        exists=exists,
        version=None,
        deps_ok=deps_ok,
        managed=bool(exe) and installer.path_is_inside(exe, venv_path),
        venv_path=venv_path,
        runtime_path=runtime_path,
        # The marker is a JSON file plus a hash of requirements.txt; no
        # interpreter is spawned for it (installer.is_ready would).
        marker_ok=installer.marker_matches(venv_path),
        probed=False,
    )


def env_info(plugin_dir, configured_exe, version_fn=None, deps_fn=None,
             ready_fn=None) -> EnvInfo:
    """Probe the compute environment. NEVER call this on the GUI thread.

    ``configured_exe`` is what the dialog is currently pointed at (the
    QgsSettings interpreter, or the managed venv's python). The three
    ``*_fn`` seams default to the installer's own probes and exist so
    tests can describe a machine instead of owning one; they are resolved
    at CALL time, so monkeypatching ``installer._python_version`` works.

    The measured version is handed to ``ready_fn`` for the managed venv,
    whose interpreter is the very one just probed — ``is_ready`` used to
    spawn it a second time to re-learn the same number.
    """
    version_fn = version_fn or installer._python_version
    deps_fn = deps_fn or installer._has_compute_deps
    ready_fn = ready_fn or installer.is_ready

    venv_path, runtime_path, exe, exists = _env_paths(plugin_dir,
                                                      configured_exe)
    version = tuple(version_fn(exe)) if exists else None
    managed = bool(exe) and installer.path_is_inside(exe, venv_path)
    return EnvInfo(
        exe=exe,
        exists=exists,
        version=version,
        deps_ok=bool(deps_fn(exe)) if exists else None,
        managed=managed,
        venv_path=venv_path,
        runtime_path=runtime_path,
        marker_ok=bool(ready_fn(venv_path,
                                version=version if managed else None)),
        probed=True,
    )


def env_state_text(info) -> str:
    """The bold one-liner at the top of Step 1."""
    if not info.exists:
        return TXT_ENV_NONE
    if not info.probed:
        return TXT_ENV_CHECKING
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


def env_location_text(plugin_dir, usage=None) -> str:
    """The gray line naming the managed root, and saying plainly that
    uninstalling the plugin leaves it behind.

    The user's report: "'<profile>/winmol' stays untouched after
    deinstalling". It does, by design and unavoidably — QGIS offers no
    uninstall hook at all (a plugin's ``unload()`` is called identically
    on disable, reload and shutdown, so deleting from there would wipe a
    multi-GB venv every time QGIS closes). The only honest answer is to
    say where it is and how to remove it.

    ``usage`` is the same optional ``{'venv': bytes, 'runtime': bytes}``
    map :func:`env_detail_text` takes; the total is appended when known.
    """
    root = installer.managed_root(plugin_dir)
    if not root:
        return ""
    total = sum(int(usage.get(key) or 0)
                for key in ("venv", "runtime")) if usage else 0
    size = f" ({human_bytes(total)})" if total else ""
    return TXT_ENV_LOCATION.format(path=root, size=size)


def env_ready(info) -> bool:
    """True when a detection could run right now."""
    if not info.exists:
        return False
    if not info.probed:
        # Optimistic by design (see EnvInfo): an interpreter that is on
        # disk is assumed usable until the worker's probe lands. This is
        # exactly what the pre-Setup-tab dialog did, and it is why the
        # first paint costs nothing.
        return True
    if not _supported(info.version):
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
    if not info.probed:
        return None
    if not _supported(info.version):
        return TXT_BLOCK_VERSION.format(version=version_text(info.version),
                                        want=_wanted_version_text())
    if info.deps_ok is False:
        return TXT_BLOCK_DEPS
    if info.managed and not info.marker_ok:
        return TXT_BLOCK_INCOMPLETE
    return None


# --- accelerator ------------------------------------------------------------


@dataclass
class AcceleratorStatus:
    """What will actually run the inference, and what to do about it.

    Built from two measurements taken on a worker thread — an nvidia-smi
    probe (:func:`installer.detect_gpu`) and the CHILD interpreter's
    provider list (:func:`installer.probe_runtime`) — and from nothing
    else. In particular it never asks the QGIS interpreter what it can
    do: the QGIS interpreter is not the one that runs the model.
    """

    state: str
    text: str
    can_install: bool = False
    gpu_label: str = ""


def requirements_choice(probe) -> str:
    """Which requirements file this machine should be built from.

    The whole install-time decision in one pure function, so every case
    that matters — GPU present, no GPU, nvidia-smi missing, nvidia-smi
    wedged, macOS, ARM — is a unit test rather than a machine somebody
    has to own.
    """
    return (installer.GPU_REQUIREMENTS if (probe is not None
                                           and probe.present)
            else installer.CPU_REQUIREMENTS)


def gpu_offer_text(probe) -> str:
    """The confirmation the user sees before 2.4 GB starts downloading.

    Nothing installs the GPU runtime without this: an unannounced
    multi-gigabyte download on a metered link is not a favour.
    """
    return TXT_GPU_OFFER.format(
        gpu=probe.label if probe is not None else "An NVIDIA GPU")


def accelerator_status(probe=None, report=None) -> AcceleratorStatus:
    """The Setup tab's accelerator line, from probe + provider report.

    ``report`` is a :func:`installer.probe_runtime` dict, or None while
    the worker is still measuring — in which case nothing is claimed.
    """
    if report is None:
        return AcceleratorStatus(ACCEL_UNKNOWN, TXT_ACCEL_UNKNOWN)

    providers = report.get("providers") or []
    packages = report.get("packages") or []
    label = probe.label if probe is not None else "NVIDIA GPU"

    if probe is None or not probe.has_hardware:
        # No NVIDIA hardware. macOS gets CoreML from the stock wheel and
        # is already as fast as it is going to get; saying "CPU only"
        # there would be a lie that sends the user shopping for a GPU.
        # A session that was tried and did NOT bind CoreML overrides that:
        # unlike CUDA, CoreML needs no separate libraries, so there is
        # nothing to offer — but there is also nothing to boast about.
        if installer.COREML_PROVIDER in providers and not (
                installer.session_attempted(report)
                and not installer.provider_confirmed(
                    report, installer.COREML_PROVIDER)):
            return AcceleratorStatus(
                ACCEL_COREML,
                TXT_ACCEL_COREML.format(gpu="Apple GPU"),
                gpu_label="Apple GPU")
        return AcceleratorStatus(ACCEL_CPU_ONLY, TXT_ACCEL_CPU_ONLY)

    if installer.CUDA_PROVIDER in providers:
        # THE provider list is not evidence. onnxruntime-gpu lists
        # CUDAExecutionProvider whenever the wheel was built with it; on a
        # machine whose CUDA/cuDNN libraries are not on the loader path it
        # then builds a CPU session and prints a warning nobody sees. This
        # branch used to report "installed and active" for exactly that
        # machine, at 10311 ms per tile. Only a session that came up on
        # CUDA is allowed to say so — and the sentence explaining the
        # failure is installer.gpu_verdict's, so the Setup tab and the
        # post-install verification cannot tell two different stories.
        ok, reason = installer.gpu_verdict(report, checked_with_model=True)
        if ok and installer.provider_confirmed(
                report, installer.CUDA_PROVIDER):
            return AcceleratorStatus(
                ACCEL_GPU_ACTIVE, TXT_ACCEL_GPU_ACTIVE.format(gpu=label),
                gpu_label=label)
        if installer.session_attempted(report):
            return AcceleratorStatus(
                ACCEL_GPU_BROKEN,
                TXT_ACCEL_GPU_BROKEN.format(gpu=label, reason=reason),
                gpu_label=label)
        # Nothing was tried, so nothing is known. Not "active" (the lie
        # this fix removes) and not "broken" either — which would nag the
        # user into re-downloading 2.4 GB that is very likely fine.
        return AcceleratorStatus(
            ACCEL_UNKNOWN, TXT_ACCEL_GPU_UNVERIFIED.format(gpu=label),
            gpu_label=label)

    if not probe.present:
        # A GPU we can see but must not install for: no wheels for this
        # platform, or a driver too old for the CUDA the wheels carry.
        return AcceleratorStatus(
            ACCEL_GPU_UNUSABLE,
            TXT_ACCEL_GPU_UNUSABLE.format(
                gpu=label, reason=probe.detail or "unsupported setup."),
            gpu_label=label)

    if installer.GPU_RUNTIME_DIST in packages:
        # The expensive download already happened and CUDA still is not
        # there. Offering the same 2.4 GB again would be cruel; name the
        # real reason instead.
        reason = report.get("error") or (
            f"{installer.CUDA_PROVIDER} is missing from the installed "
            "build — its CUDA generation probably does not match the "
            "NVIDIA driver.")
        return AcceleratorStatus(
            ACCEL_GPU_BROKEN,
            TXT_ACCEL_GPU_BROKEN.format(gpu=label, reason=reason),
            gpu_label=label)

    # No onnxruntime at all means there is no environment yet, which is a
    # different sentence from "the runtime you installed is the wrong one"
    # even though the remedy — install the GPU one — is the same.
    template = (TXT_ACCEL_GPU_IDLE if packages
                else TXT_ACCEL_GPU_NO_ENV)
    return AcceleratorStatus(
        ACCEL_GPU_IDLE, template.format(gpu=label),
        can_install=True, gpu_label=label)


def accelerator_from_machine(python_exe, plugin_dir=None,
                             detect=None, probe_fn=None) -> AcceleratorStatus:
    """:func:`accelerator_status` for a real machine. WORKER THREAD ONLY.

    Runs nvidia-smi and spawns the child interpreter, so it must never be
    reached from a repaint. The two seams keep the tests machine-free.

    The child is asked to build a REAL session when a model is on disk —
    :func:`accelerator_status` refuses to claim an accelerator on anything
    less, and without a model path the probe would never try, so the answer
    could only ever be "unverified".
    """
    detect = detect or installer.detect_gpu
    probe_fn = probe_fn or installer.probe_runtime
    probe = detect()
    # With no interpreter there is nothing to ask about providers, but the
    # GPU question is still answerable and still worth answering: it is what
    # lets "Create environment for me" offer the CUDA build on the FIRST
    # install instead of a CPU one the user has to redo.
    if python_exe:
        want = (installer.CUDA_PROVIDER
                if probe is not None and probe.has_hardware
                else installer.COREML_PROVIDER)
        report = probe_fn(python_exe,
                          model_path=installer.first_model_path(plugin_dir),
                          want=want)
    else:
        report = {
            "ok": False, "providers": [], "packages": [],
            "version": None, "error": "no compute environment yet",
            "session_providers": None}
    return accelerator_status(probe, report)


# --- the proactive offer ----------------------------------------------------
#
# Everything above answers "what will run this?". This section answers the
# two questions that make the answer REACHABLE, and both are pure functions
# of an AcceleratorStatus that some worker thread already measured:
#
#   * accel_nudge_text  — what the Detection tab says, so the offer is not
#     buried on a tab the user has no reason to open;
#   * pre_run_decision  — whether pressing Run should stop and ask first.
#
# Neither ever measures anything. They take the CACHED status the dialog is
# already holding, which is what keeps them off the list of calls banned
# from the GUI thread (tests/test_dialog_lint.py::BANNED_CALLS).

#: pre_run_decision kinds.
PRERUN_NONE = "none"
PRERUN_GPU_IDLE = "gpu_idle"

#: The states that get a proactive nudge at all. Only one qualifies, and
#: the reason the others do not is worth writing down:
#:
#: * ``cpu_only``   — no GPU. There is nothing the user can do, so saying
#:                    it twice is nagging, not information.
#: * ``coreml``     — already on the Apple GPU; no download would help.
#: * ``gpu_active`` — already using the GPU. Nothing to offer.
#: * ``gpu_broken`` / ``gpu_unusable`` — a GPU we must NOT sell a 2.4 GB
#:                    download for; ``can_install`` is False for both and
#:                    the Setup tab already names the real reason.
#: * ``unknown``    — the probe has not landed. Claiming anything here
#:                    would be a guess, and a guess must never stop a run.
NUDGE_STATES = (ACCEL_GPU_IDLE,)


@dataclass
class PreRunDecision:
    """Whether Run should interrupt, and with what.

    ``kind`` is :data:`PRERUN_NONE` for every case in which the run must
    simply proceed — which includes the case that matters most for
    responsiveness: the accelerator probe has not answered yet. A user
    who pressed Run is not made to wait on a subprocess that measures
    something they might not even need to hear.
    """

    kind: str = PRERUN_NONE
    title: str = ""
    text: str = ""
    install_label: str = ""
    run_label: str = ""
    token: str = ""
    gpu_label: str = ""

    @property
    def asks(self) -> bool:
        """True when the dialog must put a question on screen."""
        return self.kind != PRERUN_NONE


def accelerator_token(accel) -> str:
    """The value persisted when the user chooses "run on the CPU anyway".

    State plus GPU name rather than a bare "yes, dismissed": a dismissal
    is an answer about THIS machine in THIS configuration. Drop a
    different card in, and the question is worth asking once more.
    """
    if accel is None:
        return ""
    return f"{accel.state}|{accel.gpu_label}"


def should_nudge(accel) -> bool:
    """True when a surface outside the Setup tab should say something.

    ``can_install`` is required as well as the state, so the nudge and the
    Setup tab's button can never disagree about whether there is an action
    behind the sentence.
    """
    return bool(accel is not None
                and accel.state in NUDGE_STATES
                and accel.can_install)


def accel_nudge_text(accel) -> str:
    """The Detection tab's one-line warning, or '' when there is none."""
    if not should_nudge(accel):
        return ""
    return TXT_NUDGE_GPU_IDLE.format(gpu=accel.gpu_label or "An NVIDIA GPU")


def pre_run_decision(accel, dismissed=None) -> PreRunDecision:
    """What pressing Run should do about an idle GPU.

    ``accel`` is the dialog's CACHED :class:`AcceleratorStatus` (None
    while the probe is still running); ``dismissed`` is whatever was
    persisted under ``installer.QSETTINGS_GPU_PROMPT_KEY`` last time the
    user said "run on the CPU anyway".

    Three rules, in order:

    1. No measurement, no question. An unfinished probe means the run
       starts — never a stall on the GUI thread waiting for nvidia-smi.
    2. Only ``gpu_idle`` asks; see :data:`NUDGE_STATES` for why each of
       the other states stays silent. A machine with no GPU is never
       nagged about one.
    3. An answer is an answer. Once the token for this machine has been
       stored, the question is not asked again — the Setup tab's button
       is the way back.
    """
    if not should_nudge(accel):
        return PreRunDecision()
    token = accelerator_token(accel)
    if dismissed and str(dismissed) == token:
        return PreRunDecision()
    label = accel.gpu_label or "An NVIDIA GPU"
    return PreRunDecision(
        kind=PRERUN_GPU_IDLE,
        title=TXT_PRERUN_TITLE,
        text=TXT_PRERUN_GPU_IDLE.format(gpu=label),
        install_label=TXT_PRERUN_INSTALL,
        run_label=TXT_PRERUN_RUN_ANYWAY,
        token=token,
        gpu_label=label)


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
                  env_busy=False, dl_busy=False, accel=None) -> dict:
    """The enable/disable truth table for the Setup tab, keyed by widget
    object name.

    THE single authority: nothing else in the dialog may call
    ``setEnabled`` on a setup button. While anything is running —
    a detection, an environment job or a download — every one of them is
    off, which is what makes "one job at a time" true rather than
    hoped-for.

    ``models_treeWidget`` is in here for the same reason: it carries an
    ``itemDoubleClicked -> download`` connection, so leaving it live while
    the buttons are dead is a signal path straight around the interlock.

    ``accel`` is the :class:`AcceleratorStatus`; ``env_gpu_button`` is
    live only in the one state where installing the GPU runtime is the
    right answer. Default None (not yet probed) keeps it dead, so the
    button never offers a 2.4 GB download on a guess.
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
        "env_gpu_button": bool(accel is not None and accel.can_install
                               and env_ready(info)),
        "models_refresh_button": True,
        "models_download_button": downloadable,
        "models_download_default_button": recommended_missing,
        "models_verify_button": bool(
            row is not None and row.pinned and row.present),
        "models_delete_button": bool(row is not None and row.present),
        "models_treeWidget": True,
        "models_variants_checkBox": True,
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
