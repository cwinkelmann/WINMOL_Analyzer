"""The startup banner must never name a device the session is not using.

Covers the reporting side of the provider-verification fix:

  * "Hardware detected:" names NVIDIA GPUs this runtime cannot use, with the
    remedy (the user's "RTX 4080, GPU usage zero, way too slow" report).
  * "Environment:" prints Active providers and marks the device (verified)
    once a session exists, instead of echoing the requested list.
  * A session that bound something other than what was expected produces a
    Correction line AND downgrades the hardware, so the planner cannot go on
    sizing a GPU run for a CPU session.
"""
import pytest

import winmol_run
from classes.Config import Config
from classes.HardwareInfo import HardwareInfo
from utils import onnx_runtime


@pytest.fixture(autouse=True)
def clean_last_active(monkeypatch):
    monkeypatch.setattr(onnx_runtime, "_LAST_ACTIVE", None)


def make_processing():
    proc = winmol_run.ImageProcessing.__new__(winmol_run.ImageProcessing)
    proc.config = Config()
    proc.model_path = "/models/fake.onnx"
    return proc


class FakeModel:
    def __init__(self, kind, label):
        self.accelerator = kind
        self.accelerator_label = label


# --- Hardware detected: name the unusable GPUs ---------------------------

def test_hardware_line_names_gpus_this_runtime_cannot_use(
        monkeypatch, capsys):
    hw = HardwareInfo(
        cpu_count=8, total_ram_gb=32.0, gpu_count=0,
        gpu_names=[], gpu_memory_gb=[],
        accelerator="cpu", accelerator_label="CPU",
        unusable_gpu_names=["NVIDIA GeForce RTX 4080"])
    monkeypatch.setattr(HardwareInfo, "detect", staticmethod(lambda: hw))

    make_processing().detect_hardware()

    out = capsys.readouterr().out
    assert "RTX 4080" in out
    assert "onnxruntime-gpu" in out
    assert "docs/GPU.md" in out


def test_hardware_line_stays_quiet_when_the_gpu_works(monkeypatch, capsys):
    hw = HardwareInfo(
        cpu_count=8, total_ram_gb=32.0, gpu_count=1,
        gpu_names=["NVIDIA GeForce RTX 4080"], gpu_memory_gb=[16.0],
        accelerator="cuda", accelerator_label="NVIDIA GPU (CUDA)")
    monkeypatch.setattr(HardwareInfo, "detect", staticmethod(lambda: hw))

    make_processing().detect_hardware()

    out = capsys.readouterr().out
    assert "onnxruntime-gpu" not in out
    assert "Visible GPUs" in out


# --- Environment: prefer the observed session ----------------------------

def test_environment_reports_active_providers_once_verified(
        monkeypatch, capsys):
    monkeypatch.setattr(onnx_runtime, "runtime_report", lambda: {
        "onnxruntime_version": "1.27.0",
        "available_providers": ["CUDAExecutionProvider",
                                "CPUExecutionProvider"],
        "selected_providers": ["CUDAExecutionProvider",
                               "CPUExecutionProvider"],
        "accelerator": "cuda",
        "accelerator_label": "NVIDIA GPU (CUDA)",
        "override": None,
    })
    monkeypatch.setattr(onnx_runtime, "_LAST_ACTIVE", {
        "active_providers": ["CPUExecutionProvider"],
        "requested_providers": ["CUDAExecutionProvider",
                                "CPUExecutionProvider"],
        "demoted": ["CUDAExecutionProvider"],
        "reason": "did not initialise",
        "accelerator": "cpu",
        "accelerator_label": "CPU",
    })

    make_processing().report_runtime_env()

    out = capsys.readouterr().out
    assert "Active providers: CPUExecutionProvider" in out
    assert "Device: CPU (verified)" in out
    # The requested list may still be shown as "Selected", but the DEVICE must
    # never claim the GPU the session did not bind.
    assert "Device: NVIDIA GPU (CUDA)" not in out


def test_environment_ignores_a_report_from_a_different_request(
        monkeypatch, capsys):
    """A session built under another provider request proves nothing here.

    Batch runs load a model per image, and the verified result is cached
    module-level; without this guard a stale CPU observation would overwrite
    the report for an unrelated, correctly-configured run.
    """
    monkeypatch.setattr(onnx_runtime, "runtime_report", lambda: {
        "onnxruntime_version": "1.27.0",
        "available_providers": ["CoreMLExecutionProvider",
                                "CPUExecutionProvider"],
        "selected_providers": ["CoreMLExecutionProvider",
                               "CPUExecutionProvider"],
        "accelerator": "coreml",
        "accelerator_label": "Apple Silicon GPU (Metal/CoreML)",
        "override": None,
    })
    # Left behind by an earlier, CPU-forced session — a different request.
    monkeypatch.setattr(onnx_runtime, "_LAST_ACTIVE", {
        "active_providers": ["CPUExecutionProvider"],
        "requested_providers": ["CPUExecutionProvider"],
        "demoted": [],
        "reason": None,
        "accelerator": "cpu",
        "accelerator_label": "CPU",
    })

    make_processing().report_runtime_env()

    out = capsys.readouterr().out
    assert "Device: CPU (verified)" not in out
    assert "not yet verified" in out


def test_environment_marks_device_unverified_before_any_session(
        monkeypatch, capsys):
    monkeypatch.setattr(onnx_runtime, "runtime_report", lambda: {
        "onnxruntime_version": "1.27.0",
        "available_providers": ["CPUExecutionProvider"],
        "selected_providers": ["CPUExecutionProvider"],
        "accelerator": "cpu",
        "accelerator_label": "CPU",
        "override": None,
    })

    make_processing().report_runtime_env()

    out = capsys.readouterr().out
    assert "not yet verified" in out


# --- Correction after the model is loaded --------------------------------

def test_correction_downgrades_hardware_when_session_is_cpu(capsys):
    proc = make_processing()
    proc.config.hardware = HardwareInfo(
        cpu_count=8, total_ram_gb=32.0, gpu_count=1,
        gpu_names=["NVIDIA GeForce RTX 4080"], gpu_memory_gb=[16.0],
        accelerator="cuda", accelerator_label="NVIDIA GPU (CUDA)")

    proc._correct_accelerator_after_load(FakeModel("cpu", "CPU"))

    hw = proc.config.hardware
    assert hw.accelerator == "cpu"
    assert hw.gpu_names == []
    assert hw.gpu_count == 0
    assert hw.unusable_gpu_names == ["NVIDIA GeForce RTX 4080"]
    out = capsys.readouterr().out
    assert "Correction" in out
    assert "expected NVIDIA GPU (CUDA)" in out


def test_no_correction_when_the_session_matches(capsys):
    proc = make_processing()
    proc.config.hardware = HardwareInfo(
        cpu_count=8, total_ram_gb=32.0, gpu_count=1,
        gpu_names=["NVIDIA GeForce RTX 4080"], gpu_memory_gb=[16.0],
        accelerator="cuda", accelerator_label="NVIDIA GPU (CUDA)")

    proc._correct_accelerator_after_load(
        FakeModel("cuda", "NVIDIA GPU (CUDA)"))

    assert proc.config.hardware.gpu_names == ["NVIDIA GeForce RTX 4080"]
    assert "Correction" not in capsys.readouterr().out


def test_correction_tolerates_a_model_without_provider_info(capsys):
    """Legacy Keras models have no .accelerator — must not crash."""
    proc = make_processing()
    proc.config.hardware = HardwareInfo(
        cpu_count=8, total_ram_gb=32.0, gpu_count=0,
        gpu_names=[], gpu_memory_gb=[],
        accelerator="cpu", accelerator_label="CPU")

    proc._correct_accelerator_after_load(object())

    assert "Correction" not in capsys.readouterr().out
