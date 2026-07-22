"""The plugin must say when a present NVIDIA GPU is unusable by its env.

requirements/plugin.txt -> base.txt installs `onnxruntime`, the CPU-only wheel.
`onnxruntime-gpu` is a separate package, so a plugin-installed NVIDIA box can
never reach CUDA no matter what driver is present. The install is deliberately
NOT changed (that would drag a large CUDA dependency into every environment) —
instead the user is told, with the remedy.

All stubbed: no subprocesses, no GPU.
"""
import pytest

from plugin_utils import installer

CPU_ONLY = ["CPUExecutionProvider"]
WITH_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def box(monkeypatch):
    """Fake (nvidia-smi present?, providers offered) for the hint."""
    def configure(gpu_present, providers):
        monkeypatch.setattr(
            installer, "_nvidia_gpu_present", lambda: gpu_present)
        monkeypatch.setattr(
            installer, "_onnx_providers", lambda exe: list(providers))
    return configure


def test_hint_when_nvidia_gpu_present_but_runtime_is_cpu_only(box):
    box(True, CPU_ONLY)
    hint = installer.cuda_capability_hint("/venv/bin/python")
    assert "onnxruntime-gpu" in hint
    assert "docs/GPU.md" in hint


def test_no_hint_when_cuda_is_available(box):
    box(True, WITH_CUDA)
    assert installer.cuda_capability_hint("/venv/bin/python") == ""


def test_no_hint_without_an_nvidia_gpu(box):
    """A Mac must never be told to install onnxruntime-gpu."""
    box(False, CPU_ONLY)
    assert installer.cuda_capability_hint("/venv/bin/python") == ""


def test_no_hint_when_providers_cannot_be_probed(box):
    """Probe failure is not evidence of anything — stay quiet."""
    box(True, [])
    assert installer.cuda_capability_hint("/venv/bin/python") == ""


def test_no_hint_without_an_interpreter(box):
    box(True, CPU_ONLY)
    assert installer.cuda_capability_hint(None) == ""
