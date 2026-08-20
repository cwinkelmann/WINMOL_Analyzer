from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import List

from plugin_utils import container


@dataclass
class HardwareInfo:
    cpu_count: int
    total_ram_gb: float
    gpu_count: int
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)

    @classmethod
    def detect(cls) -> "HardwareInfo":
        # Container-aware: inside a cgroup these must report what THIS
        # process may use, not what the host has. Sizing the vector pool
        # and prediction batch from host RAM inside `--memory=8g` is an
        # OOM kill (see plugin_utils/container).
        cpu_count = container.cpu_count()
        total_ram_gb = cls._detect_total_ram_gb()
        gpu_names = cls._detect_gpu_names_nvidia_smi()
        gpu_memory_gb = cls._detect_gpu_memory_gb_nvidia_smi()

        gpu_names, gpu_memory_gb = cls._apply_cuda_visible_devices(
            gpu_names, gpu_memory_gb
        )

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
        )

    @staticmethod
    def _detect_total_ram_gb() -> float:
        try:
            return round(
                container.total_memory_bytes() / (1024 ** 3), 2)
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

    @staticmethod
    def _detect_gpu_memory_gb_nvidia_smi() -> List[float]:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total',
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
