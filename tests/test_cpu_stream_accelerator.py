"""cpu_stream must not disable CoreML on Apple Silicon.

The planner picks cpu_stream whenever there is no CUDA GPU. On a Mac
that is for lack of *CUDA* — CoreML is still available and ~18x faster
than the CPU provider — so cpu_stream must only pin the ONNX CPU
provider when the user explicitly chose the cpu backend, or the machine
offers no accelerator at all.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from winmol_run import _cpu_stream_forces_onnx_cpu  # noqa: E402

_CPU = ["CPUExecutionProvider"]
_COREML = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_apple_silicon_cpu_stream_keeps_coreml():
    # the reported bug: fp32 on an M2, cpu_stream, was pinned to CPU.
    assert _cpu_stream_forces_onnx_cpu("auto", _COREML) is False


def test_cuda_box_in_cpu_stream_keeps_cuda():
    assert _cpu_stream_forces_onnx_cpu("auto", _CUDA) is False


def test_real_cpu_only_box_forces_cpu():
    assert _cpu_stream_forces_onnx_cpu("auto", _CPU) is True


def test_explicit_cpu_backend_forces_cpu_even_with_coreml():
    # the user asked for CPU — respect it, even on Apple Silicon.
    assert _cpu_stream_forces_onnx_cpu("cpu", _COREML) is True
    assert _cpu_stream_forces_onnx_cpu("CPU", _CUDA) is True
