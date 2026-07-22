"""The session must tell the truth about the device it actually runs on.

onnxruntime does NOT fail when a requested execution provider is unavailable:
``InferenceSession(..., providers=["CUDAExecutionProvider", ...])`` builds a
working CPU session and emits only a Python ``UserWarning``. Measured on this
tree (macOS/arm64, onnxruntime 1.27.0): requesting CUDA yields a session whose
``get_providers()`` is ``['CPUExecutionProvider']`` — while every banner in the
codebase derived the device from the *requested* list and printed
"NVIDIA GPU (CUDA)".

These tests pin the fix: compare requested against ``session.get_providers()``
and report what the session actually bound. All stubbed — no GPU, no model, no
network.
"""
import platform

import pytest

from classes.HardwareInfo import HardwareInfo
from utils import onnx_runtime

CUDA_REQUEST = ["CUDAExecutionProvider", "CPUExecutionProvider"]
COREML_REQUEST = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
CPU_ONLY = ["CPUExecutionProvider"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("WINMOL_ONNX_FORCE_CPU", "WINMOL_ONNX_PROVIDERS",
                "WINMOL_ONNX_PROFILE", "WINMOL_DISABLE_METAL",
                "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)


class FakeSession:
    """Stand-in for ort.InferenceSession honouring only some providers."""

    def __init__(self, active):
        self._active = list(active)

    def get_providers(self):
        return list(self._active)

    def get_inputs(self):
        return [_IO("input", [None, 512, 512, 3])]

    def get_outputs(self):
        return [_IO("output", [None, 512, 512, 1])]


class _IO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


def make_segmenter(monkeypatch, requested, active, available=None):
    """Build an OnnxSegmenter whose session binds exactly `active`."""
    if available is None:
        available = list(active)
    monkeypatch.setattr(
        onnx_runtime, "_available_providers", lambda: list(available))
    monkeypatch.setattr(
        onnx_runtime.ort, "InferenceSession",
        lambda path, sess_options=None, providers=None: FakeSession(active))
    return onnx_runtime.OnnxSegmenter("/models/fake.onnx",
                                      providers=list(requested))


# --- verify_session_providers: the two reason branches -------------------

def test_verify_reports_no_demotion_when_request_is_honoured(monkeypatch):
    monkeypatch.setattr(
        onnx_runtime, "_available_providers", lambda: list(CUDA_REQUEST))
    active, demoted, reason = onnx_runtime.verify_session_providers(
        CUDA_REQUEST, CUDA_REQUEST)
    assert active == CUDA_REQUEST
    assert demoted == []
    assert reason is None


def test_verify_reason_when_provider_not_in_this_build(monkeypatch):
    """CPU-only 'onnxruntime' wheel: CUDA is not even offered."""
    monkeypatch.setattr(
        onnx_runtime, "_available_providers", lambda: list(CPU_ONLY))
    _, demoted, reason = onnx_runtime.verify_session_providers(
        CUDA_REQUEST, CPU_ONLY)
    assert demoted == ["CUDAExecutionProvider"]
    # Must name the actual remedy: onnxruntime-gpu is a DIFFERENT package.
    assert "onnxruntime-gpu" in reason


def test_verify_reason_when_provider_available_but_unbound(monkeypatch):
    """onnxruntime-gpu installed but CUDA/cuDNN/driver mismatched."""
    monkeypatch.setattr(
        onnx_runtime, "_available_providers", lambda: list(CUDA_REQUEST))
    _, demoted, reason = onnx_runtime.verify_session_providers(
        CUDA_REQUEST, CPU_ONLY)
    assert demoted == ["CUDAExecutionProvider"]
    assert "did not initialise" in reason or "not in the active" in reason
    # Stay factual: we cannot diagnose the driver from here.
    assert "onnxruntime-gpu" not in reason


def test_verify_ignores_cpu_fallback_as_a_demotion(monkeypatch):
    """CPU is always appended as a fallback; dropping it is not a demotion."""
    monkeypatch.setattr(
        onnx_runtime, "_available_providers", lambda: list(COREML_REQUEST))
    _, demoted, _ = onnx_runtime.verify_session_providers(
        COREML_REQUEST, ["CoreMLExecutionProvider"])
    assert demoted == []


# --- active_accelerator: driven by what the session BOUND ----------------

def test_active_accelerator_is_cpu_when_only_cpu_is_bound():
    assert onnx_runtime.active_accelerator(CPU_ONLY)[0] == "cpu"


def test_active_accelerator_reports_cuda_when_cuda_is_bound():
    kind, label = onnx_runtime.active_accelerator(CUDA_REQUEST)
    assert kind == "cuda"
    assert label == onnx_runtime.ACCELERATOR_LABELS["cuda"]


def test_active_accelerator_coreml_only_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert onnx_runtime.active_accelerator(COREML_REQUEST)[0] == "coreml"
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert onnx_runtime.active_accelerator(COREML_REQUEST)[0] == "cpu"


# --- OnnxSegmenter: the honesty contract ---------------------------------

def test_segmenter_warns_loudly_when_cuda_silently_falls_back(
        monkeypatch, capsys):
    seg = make_segmenter(
        monkeypatch, CUDA_REQUEST, CPU_ONLY, available=CPU_ONLY)

    assert seg.active_providers == CPU_ONLY
    assert seg.demoted == ["CUDAExecutionProvider"]
    assert seg.accelerator == "cpu"
    assert seg.accelerator_label == onnx_runtime.ACCELERATOR_LABELS["cpu"]

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "CUDAExecutionProvider" in out
    assert "onnxruntime-gpu" in out
    # The whole point: never claim the device the session is not using.
    assert onnx_runtime.ACCELERATOR_LABELS["cuda"] not in out


def test_segmenter_is_silent_when_cuda_is_honoured(monkeypatch, capsys):
    seg = make_segmenter(monkeypatch, CUDA_REQUEST, CUDA_REQUEST)
    assert seg.demoted == []
    assert seg.accelerator == "cuda"
    assert "WARNING" not in capsys.readouterr().out


def test_segmenter_warns_when_coreml_falls_back(monkeypatch, capsys):
    seg = make_segmenter(
        monkeypatch, COREML_REQUEST, CPU_ONLY, available=CPU_ONLY)
    assert seg.demoted == ["CoreMLExecutionProvider"]
    assert seg.accelerator == "cpu"
    out = capsys.readouterr().out
    assert "CoreMLExecutionProvider" in out
    assert onnx_runtime.ACCELERATOR_LABELS["coreml"] not in out


def test_segmenter_publishes_verified_result_for_the_banner(monkeypatch):
    """winmol_run corrects the already-printed 'Hardware detected:' line."""
    make_segmenter(monkeypatch, CUDA_REQUEST, CPU_ONLY, available=CPU_ONLY)
    last = onnx_runtime.last_active_report()
    assert last is not None
    assert last["accelerator"] == "cpu"
    assert last["active_providers"] == CPU_ONLY
    assert last["demoted"] == ["CUDAExecutionProvider"]


def test_segmenter_coreml_honoured_on_apple_silicon(monkeypatch, capsys):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    seg = make_segmenter(monkeypatch, COREML_REQUEST, COREML_REQUEST)
    assert seg.accelerator == "coreml"
    assert seg.demoted == []
    assert "WARNING" not in capsys.readouterr().out


# --- HardwareInfo: name the unusable NVIDIA GPUs -------------------------

def test_hardware_records_nvidia_gpus_this_runtime_cannot_use(monkeypatch):
    """The RTX 4080 report: nvidia-smi sees it, onnxruntime cannot use it."""
    monkeypatch.setattr(
        HardwareInfo, "_detect_gpu_names_nvidia_smi",
        staticmethod(lambda: ["NVIDIA GeForce RTX 4080"]))
    monkeypatch.setattr(
        HardwareInfo, "_detect_gpu_memory_gb_nvidia_smi",
        staticmethod(lambda: [16.0]))
    monkeypatch.setattr(
        onnx_runtime, "_available_providers", lambda: list(CPU_ONLY))

    hw = HardwareInfo.detect()

    assert hw.accelerator == "cpu"
    assert hw.gpu_names == []          # planner must not size a GPU run
    assert hw.unusable_gpu_names == ["NVIDIA GeForce RTX 4080"]


def test_hardware_leaves_unusable_empty_when_cuda_works(monkeypatch):
    monkeypatch.setattr(
        HardwareInfo, "_detect_gpu_names_nvidia_smi",
        staticmethod(lambda: ["NVIDIA GeForce RTX 4080"]))
    monkeypatch.setattr(
        HardwareInfo, "_detect_gpu_memory_gb_nvidia_smi",
        staticmethod(lambda: [16.0]))
    monkeypatch.setattr(
        onnx_runtime, "_available_providers", lambda: list(CUDA_REQUEST))

    hw = HardwareInfo.detect()

    assert hw.accelerator == "cuda"
    assert hw.gpu_names == ["NVIDIA GeForce RTX 4080"]
    assert hw.unusable_gpu_names == []
