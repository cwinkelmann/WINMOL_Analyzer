from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

# Share of unified memory reported as the "GPU" budget on Apple Silicon. The
# CPU, the OS and the raster reader all live in the same pool, so handing the
# planner the full figure over-sizes the prediction batch (ExecutionPlan tiers
# on >= 20 GB / >= 12 GB of dedicated VRAM).
UNIFIED_MEMORY_GPU_SHARE = 0.5


@dataclass
class HardwareInfo:
    cpu_count: int
    total_ram_gb: float
    gpu_count: int
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)
    # What actually runs the inference: 'cuda' | 'coreml' | 'cpu', plus the
    # name to show the user. Defaulted so existing constructors keep working.
    accelerator: str = "cpu"
    accelerator_label: str = "CPU"

    @classmethod
    def detect(cls) -> "HardwareInfo":
        cpu_count = os.cpu_count() or 1
        total_ram_gb = cls._detect_total_ram_gb()
        gpu_names = cls._detect_gpu_names_nvidia_smi()
        gpu_memory_gb = cls._detect_gpu_memory_gb_nvidia_smi()

        gpu_names, gpu_memory_gb = cls._apply_cuda_visible_devices(
            gpu_names, gpu_memory_gb
        )

        # Ask the runtime that actually performs inference (onnxruntime) which
        # device it will use, rather than inferring one from installed
        # packages. Returns None when onnxruntime cannot be imported at all.
        # CUDA_VISIBLE_DEVICES="" / "-1" is a whole-process CPU pin, so it
        # short-circuits the probe; _apply_cuda_visible_devices above has
        # already emptied the NVIDIA list for it.
        kind = (None if cls._cuda_hidden_via_env()
                else cls._detect_accelerator_kind())

        if gpu_names:
            # nvidia-smi stays the authority for NVIDIA count and per-GPU
            # memory; onnxruntime only confirms it can use them. A CPU-only
            # onnxruntime build (or a forced CPU run) means those GPUs are
            # unreachable, so do not let the planner size a GPU run for them.
            # kind is None only when onnxruntime could not be asked at all --
            # trust nvidia-smi then rather than downgrading a working box.
            # WINMOL_DISABLE_METAL is deliberately NOT consulted here: it is a
            # Metal-only escape hatch and must never discard NVIDIA GPUs.
            if kind in (None, "cuda"):
                accelerator = "cuda"
            else:
                gpu_names, gpu_memory_gb = [], []
                accelerator = "cpu"
        elif kind == "coreml" and not cls._metal_disabled_via_env():
            # Apple Silicon: no nvidia-smi, but onnxruntime runs the model on
            # the integrated GPU/ANE through CoreML.
            gpu_names, gpu_memory_gb = cls._metal_gpu(total_ram_gb)
            accelerator = "coreml" if gpu_names else "cpu"
        else:
            accelerator = "cpu"

        gpu_count = len(gpu_names)

        if gpu_memory_gb and gpu_count != len(gpu_memory_gb):
            if len(gpu_memory_gb) > gpu_count:
                gpu_memory_gb = gpu_memory_gb[:gpu_count]
            else:
                gpu_memory_gb.extend([0.0] * (gpu_count - len(gpu_memory_gb)))

        return cls(
            cpu_count=cpu_count,
            total_ram_gb=total_ram_gb,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_memory_gb=gpu_memory_gb,
            accelerator=accelerator,
            accelerator_label=cls._label(accelerator),
        )

    @staticmethod
    def _label(accelerator: str) -> str:
        try:
            from utils.onnx_runtime import ACCELERATOR_LABELS
            return ACCELERATOR_LABELS.get(accelerator, "CPU")
        except Exception:
            return {
                "cuda": "NVIDIA GPU (CUDA)",
                "coreml": "Apple Silicon GPU (Metal/CoreML)",
            }.get(accelerator, "CPU")

    @staticmethod
    def _detect_accelerator_kind() -> Optional[str]:
        """'cuda' | 'coreml' | 'cpu' per onnxruntime, or None if unavailable.

        None means "could not ask" — callers must then fall back to the
        nvidia-smi verdict rather than downgrading a working CUDA box. The
        import is guarded because HardwareInfo is imported at module scope by
        winmol_run, and a missing onnxruntime must surface at model load with
        a clear message, not as an import-time crash.
        """
        try:
            from utils import onnx_runtime
            kind, _ = onnx_runtime.accelerator()
            return kind
        except Exception:
            return None

    @staticmethod
    def _detect_total_ram_gb() -> float:
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            return 0.0

    @staticmethod
    def _detect_gpu_names_nvidia_smi() -> List[str]:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return []
            return [line.strip()
                    for line in result.stdout.splitlines() if line.strip()]
        except Exception:
            return []

    @classmethod
    def _detect_gpu_memory_gb_nvidia_smi(cls) -> List[float]:
        return cls._query_gpu_memory_gb('memory.total')

    @staticmethod
    def _query_gpu_memory_gb(field: str) -> List[float]:
        """``nvidia-smi --query-gpu=<field>`` in GB; ``[]`` on any failure."""
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu={field}',
                 '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return []
            values = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(round(float(line) / 1024.0, 2))
                except Exception:
                    continue
            return values
        except Exception:
            return []

    @classmethod
    def free_gpu_memory_gb(cls) -> List[float]:
        """Per-visible-GPU FREE memory, in GB. Empty list when unknown.

        ``gpu_memory_gb`` reports memory.TOTAL, which is the wrong number for
        a safety cap: a desktop GPU is also driving the display and may
        already be hosting another process. This asks for memory.free and
        applies the same CUDA_VISIBLE_DEVICES filtering, so callers see only
        the devices this process may use.

        Treat the answer as an upper bound, never as permission -- another
        process can allocate between the probe and the run, and WDDM reports
        a shared pool. The OOM fallback in Prediction stays the second line
        of defence.
        """
        values = cls._query_gpu_memory_gb('memory.free')
        if not values:
            return []
        _, filtered = cls._apply_cuda_visible_devices(
            [''] * len(values), values)
        return filtered

    @staticmethod
    def _cuda_hidden_via_env() -> bool:
        """True when CUDA_VISIBLE_DEVICES hides every device ("" or "-1").

        The device *filtering* lives in _apply_cuda_visible_devices; this only
        answers whether the user asked for a CPU-only process.
        """
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        return raw is not None and raw.strip() in ("", "-1")

    @staticmethod
    def _metal_disabled_via_env() -> bool:
        """WINMOL_DISABLE_METAL — the Apple-Silicon-only escape hatch.

        Scoped to the CoreML branch on purpose. It once shared a helper with
        the CUDA_VISIBLE_DEVICES check, which made setting it on an NVIDIA box
        silently discard every GPU and drop the run to cpu_stream.
        """
        disable = str(os.environ.get("WINMOL_DISABLE_METAL", "")).strip()
        return disable.lower() in ("1", "true", "yes")

    @classmethod
    def _metal_gpu(
        cls,
        total_ram_gb: float,
    ) -> tuple[List[str], List[float]]:
        """The Apple Silicon entry, once onnxruntime has confirmed CoreML."""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return [], []
        # Unified memory: report a share of system RAM, not all of it (see
        # UNIFIED_MEMORY_GPU_SHARE).
        mem = (round(total_ram_gb * UNIFIED_MEMORY_GPU_SHARE, 2)
               if total_ram_gb else 0.0)
        return [cls._label("coreml")], [mem]

    @staticmethod
    def _apply_cuda_visible_devices(
        gpu_names: List[str],
        gpu_memory_gb: List[float],
    ) -> tuple[List[str], List[float]]:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        # Not set -> keep all GPUs visible
        if raw is None:
            return gpu_names, gpu_memory_gb

        raw = raw.strip()

        # Empty string or "-1" -> CPU-only
        if raw == "" or raw == "-1":
            return [], []

        indices = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                indices.append(int(token))
            except ValueError:
                # ignore non-integer tokens
                continue

        if not indices:
            return [], []

        filtered_names = []
        filtered_mem = []

        for idx in indices:
            if 0 <= idx < len(gpu_names):
                filtered_names.append(gpu_names[idx])
                if 0 <= idx < len(gpu_memory_gb):
                    filtered_mem.append(gpu_memory_gb[idx])

        return filtered_names, filtered_mem
