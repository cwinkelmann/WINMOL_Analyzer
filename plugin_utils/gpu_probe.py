"""Is there an NVIDIA GPU worth installing a CUDA runtime for?

Answers the INSTALLER's question — asked before any onnxruntime exists
to ask instead — so it imports nothing but the stdlib (no onnxruntime,
no Qt, no ``classes.*``). ``nvidia-smi`` always runs with a timeout: a
wedged driver would otherwise hang for as long as the kernel takes.
"""

import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

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


def _run_nvidia_smi(timeout):
    """``(None, stdout)`` on success, ``(status, stdout)`` on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
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
