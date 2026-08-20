"""Is there an NVIDIA GPU worth installing a CUDA runtime for — and,
once installed, did it actually get one?

:func:`probe` answers the INSTALLER's PRE-install question, asked
before any onnxruntime exists to ask instead — so it imports nothing
but the stdlib (no onnxruntime, no Qt, no ``classes.*``). ``nvidia-smi``
always runs with a timeout: a wedged driver would otherwise hang for as
long as the kernel takes.

:func:`verify_gpu_providers` answers the POST-install question, by
running the CHILD venv's own ``python -c "import onnxruntime"`` — the
only onnxruntime this module ever touches is that subprocess's, so the
"no onnxruntime" import rule above still holds for this process. It
imports :mod:`plugin_utils.childenv`, itself pure stdlib, to give that
child the same sanitized environment the real run uses.
"""

import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

from .childenv import run_isolated

#: Seconds before a wedged ``nvidia-smi`` is given up on. A healthy
#: driver answers in ~50 ms; a broken one costs a pause, not a hang.
NVIDIA_SMI_TIMEOUT = 8.0

#: Minimum NVIDIA driver for the CUDA 12.x runtime the wheels carry.
#: Below this the wheels load but every CUDA call fails.
MIN_DRIVER = {"Linux": (525, 60), "Windows": (527, 41)}

#: Platforms NVIDIA publishes onnxruntime-gpu wheels for.
SUPPORTED_SYSTEMS = ("Linux", "Windows")
SUPPORTED_MACHINES = ("x86_64", "amd64", "x64")

# Probe outcomes. Only ``ok`` means "install the GPU runtime".
STATUS_OK = "ok"                       #: a driver and at least one GPU
STATUS_NONE = "none"                   #: nvidia-smi ran, listed no GPU
STATUS_NO_DRIVER = "no-driver"         #: nvidia-smi absent or erroring
STATUS_TIMEOUT = "timeout"             #: nvidia-smi wedged
STATUS_UNSUPPORTED = "unsupported"     #: macOS / ARM — no wheels exist
STATUS_OLD_DRIVER = "old-driver"       #: GPU found, driver too old


@dataclass
class GpuProbe:
    """What ``nvidia-smi`` said, plus the verdict drawn from it."""

    status: str
    names: List[str] = field(default_factory=list)
    driver_version: Optional[str] = None
    detail: str = ""

    @property
    def present(self) -> bool:
        """True when a GPU exists AND the GPU runtime can serve it."""
        return self.status == STATUS_OK

    @property
    def label(self) -> str:
        """'NVIDIA GeForce RTX 4080 SUPER', or 'NVIDIA GPU' fallback."""
        if not self.names:
            return "NVIDIA GPU"
        if len(self.names) == 1:
            return self.names[0]
        return f"{self.names[0]} (+{len(self.names) - 1} more)"


def platform_supported(system=None, machine=None) -> bool:
    """True on Linux/Windows x86_64 — the only targets with wheels."""
    system = system or platform.system()
    machine = (machine or platform.machine() or "").lower()
    return system in SUPPORTED_SYSTEMS and machine in SUPPORTED_MACHINES


def parse_driver_version(text):
    """(major, minor) from '580.159.03', or None when unparseable."""
    match = re.match(r"\s*(\d+)\.(\d+)", str(text or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def driver_new_enough(driver_version, system=None) -> bool:
    """True when the driver can run the CUDA 12 wheels. An unparseable
    version counts as new enough: the post-install probe reports real
    failures better than refusing over an unreadable string would."""
    parsed = parse_driver_version(driver_version)
    if parsed is None:
        return True
    minimum = MIN_DRIVER.get(system or platform.system())
    if minimum is None:
        return True
    return parsed >= minimum


def _run_nvidia_smi(timeout, fields="name,driver_version", nounits=False):
    """``(None, stdout)`` on success, ``(status, stdout)`` on failure.
    The one place that builds and runs an ``nvidia-smi --query-gpu``
    argv; :func:`run_nvidia_smi_query` is the front door for callers
    that do not need the timeout/no-driver distinction."""
    fmt = "csv,noheader" + (",nounits" if nounits else "")
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=" + fields,
             "--format=" + fmt],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return STATUS_TIMEOUT, ""
    except (OSError, ValueError):
        # FileNotFoundError: no NVIDIA driver installed. The common
        # case, and not an error — most machines have no NVIDIA GPU.
        return STATUS_NO_DRIVER, ""
    if result.returncode != 0:
        return STATUS_NO_DRIVER, result.stdout or ""
    return None, result.stdout or ""


def run_nvidia_smi_query(fields, timeout=NVIDIA_SMI_TIMEOUT, nounits=False):
    """Stripped, non-empty output lines of ``nvidia-smi
    --query-gpu=<fields> --format=csv,noheader[,nounits]``, or ``None``
    on ANY failure (absent, erroring, or wedged driver). Never raises —
    the shared query for every caller that only needs the lines
    (model_registry's device probe, winmol_batch's GPU count,
    Prediction's free-VRAM read)."""
    try:
        failure, stdout = _run_nvidia_smi(timeout, fields=fields,
                                          nounits=nounits)
    except Exception:
        return None
    if failure is not None:
        return None
    return [ln.strip() for ln in stdout.splitlines() if ln.strip()]


def probe(system=None, machine=None, timeout=NVIDIA_SMI_TIMEOUT,
          runner=None) -> GpuProbe:
    """Look for an NVIDIA GPU the GPU runtime could actually use.
    ``runner`` is the test seam (same contract as _run_nvidia_smi).
    Never raises."""
    system = system or platform.system()
    if not platform_supported(system, machine):
        return GpuProbe(
            status=STATUS_UNSUPPORTED,
            detail=(f"{system}/{machine or platform.machine()} has no "
                    "onnxruntime-gpu wheels."))

    runner = runner or _run_nvidia_smi
    failure, stdout = runner(timeout)
    if failure == STATUS_TIMEOUT:
        return GpuProbe(
            status=STATUS_TIMEOUT,
            detail=(f"nvidia-smi did not answer within {timeout:.0f}s; "
                    "treating this machine as CPU-only."))
    if failure is not None:
        return GpuProbe(status=STATUS_NO_DRIVER,
                        detail="nvidia-smi is not available.")

    names, driver = [], None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if parts[0]:
            names.append(parts[0])
        if driver is None and len(parts) > 1 and parts[1]:
            driver = parts[1]

    if not names:
        return GpuProbe(status=STATUS_NONE, driver_version=driver,
                        detail="nvidia-smi reported no GPUs.")
    if not driver_new_enough(driver, system):
        low = MIN_DRIVER.get(system)
        return GpuProbe(
            status=STATUS_OLD_DRIVER, names=names, driver_version=driver,
            detail=(f"driver {driver} is older than the "
                    f"{low[0]}.{low[1]} the CUDA 12 wheels need."))
    return GpuProbe(status=STATUS_OK, names=names, driver_version=driver,
                    detail=f"driver {driver}" if driver else "")


def wants_gpu_runtime(probe_result=None) -> bool:
    """True when this machine should get onnxruntime-gpu. Never
    raises."""
    try:
        result = probe_result if probe_result is not None else probe()
        return result.present
    except Exception:
        return False


# --- post-install provider verdict ------------------------------------------
#: NOT the definitive proof — that is the provider-demotion warning
#: OnnxSegmenter._verify_providers prints on the first real detection run
#: (utils/onnx_runtime.py). This is the visible, install-time signal a
#: gpu-variant install got what it asked for, restored from rc11.

CUDA_PROVIDER = "CUDAExecutionProvider"
GPU_RUNTIME_DIST = "onnxruntime-gpu"

#: Seconds before a wedged/downloading child interpreter is given up on.
PROVIDER_PROBE_TIMEOUT = 60.0

#: stdlib-only: printed lines are parsed back out, never eval'd. The
#: WINMOL_PROBE: sentinel anchors parsing — nvidia wheels can emit
#: registration diagnostics on import, which would otherwise shift the
#: positional lines and silently corrupt the verdict.
_PROBE_SENTINEL = "WINMOL_PROBE:"
_PROVIDER_PROBE_CODE = (
    "import onnxruntime as ort\n"
    "print('WINMOL_PROBE:' + ort.__version__)\n"
    "print('WINMOL_PROBE:' + ','.join(ort.get_available_providers()))\n"
)

_DEFINITIVE_CHECK = (
    "The definitive check happens at the first detection run (a "
    "provider-demotion warning there means CPU after all).")


def _probe_providers(venv_python, timeout=PROVIDER_PROBE_TIMEOUT) -> dict:
    """``{'ok', 'version', 'providers', 'error'}`` from the CHILD venv's
    own onnxruntime. ``ok`` is False on any failure — bad exit code,
    unparseable output, a crash, or a timeout — with ``error`` set to a
    short diagnosis. Never raises."""
    empty = {"ok": False, "version": None, "providers": [], "error": None}
    try:
        out = run_isolated(venv_python, _PROVIDER_PROBE_CODE, timeout)
    except subprocess.TimeoutExpired:
        return {**empty, "error": f"timed out after {timeout:.0f}s"}
    except Exception as exc:
        return {**empty, "error": f"{type(exc).__name__}: {exc}"}
    lines = [ln[len(_PROBE_SENTINEL):].strip()
             for ln in (out.stdout or "").splitlines()
             if ln.startswith(_PROBE_SENTINEL)]
    if out.returncode != 0 or len(lines) < 2:
        detail = (out.stderr or out.stdout or "no output").strip()
        return {**empty, "error": detail[-400:] or "no output"}
    providers = [p for p in lines[1].split(",") if p]
    return {
        "ok": True, "version": lines[0],
        "providers": providers, "error": None,
    }


def _provider_verdict(report) -> str:
    """The verdict TEXT for a :func:`_probe_providers` report. Pure."""
    if not report.get("ok"):
        detail = (report.get("error") or "").strip() or "no diagnosis"
        return (
            f"Could not verify the GPU runtime in the compute "
            f"environment: {detail}. {_DEFINITIVE_CHECK}")

    version = report.get("version") or "?"
    providers = report.get("providers") or []
    if CUDA_PROVIDER in providers:
        return f"GPU runtime ready: onnxruntime {version} offers CUDA."
    return (
        f"{GPU_RUNTIME_DIST} is installed, but onnxruntime {version} does "
        f"not offer CUDA (providers: {', '.join(providers) or 'none'}). "
        f"This run will fall back to the CPU runtime. {_DEFINITIVE_CHECK}")


def verify_gpu_providers(venv_python, timeout=PROVIDER_PROBE_TIMEOUT) -> str:
    """Install-time verdict on a just-installed gpu-variant venv: does
    the CHILD interpreter's onnxruntime actually offer CUDA?

    Spawns ``venv_python`` — never the QGIS process, which has no
    onnxruntime of its own — through the same :func:`child_env` the real
    run uses, so a loader-path problem shows up here instead of at the
    first detection. Never raises; a probe crash/timeout is reported as
    an honest "could not verify", not an exception."""
    return _provider_verdict(_probe_providers(venv_python, timeout=timeout))
