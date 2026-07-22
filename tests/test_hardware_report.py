"""Accelerator detection and the runtime-environment banner.

Regression cover for two user-visible lies on a Mac:

  * ``Hardware detected: CPUs=8, RAM=24.0 GB, GPUs=0`` — the Metal probe was
    gated on the ``tensorflow-metal`` package, which can never be installed in
    the TF-free plugin venv (requirements/plugin.txt), so the accelerator that
    actually runs inference (onnxruntime + CoreML) was reported as absent.
  * ``Tensorflow error: ...`` / ``Check CUDA environment`` — a TensorFlow/CUDA
    banner printed by a runtime that uses neither.

All tests are pure unit tests: no models, no network, no GPU required. The
accelerator is faked by patching the provider list onnxruntime reports.
"""
import platform

import pytest

from classes.Config import Config
from classes.ExecutionPlan import build_execution_plan
from classes.HardwareInfo import HardwareInfo
from utils import onnx_runtime

CPU_ONLY_PROVIDERS = ["CPUExecutionProvider"]
COREML_PROVIDERS = [
    "CoreMLExecutionProvider", "AzureExecutionProvider",
    "CPUExecutionProvider",
]
CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture(autouse=True)
def clean_accelerator_env(monkeypatch):
    """The suite itself runs with WINMOL_ONNX_FORCE_CPU=1; start from neutral.

    Every test opts in to the env it wants, so none of them silently inherit
    the harness' CPU pin.
    """
    for var in ("WINMOL_ONNX_FORCE_CPU", "WINMOL_ONNX_PROVIDERS",
                "WINMOL_DISABLE_METAL", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)


def fake_providers(monkeypatch, providers):
    monkeypatch.setattr(
        onnx_runtime, "_available_providers", lambda: list(providers))


def fake_platform(monkeypatch, system, machine="x86_64"):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)


def fake_nvidia_smi(monkeypatch, names, memory_gb):
    monkeypatch.setattr(
        HardwareInfo, "_detect_gpu_names_nvidia_smi",
        staticmethod(lambda: list(names)))
    monkeypatch.setattr(
        HardwareInfo, "_detect_gpu_memory_gb_nvidia_smi",
        staticmethod(lambda: list(memory_gb)))


def no_nvidia_smi(monkeypatch):
    fake_nvidia_smi(monkeypatch, [], [])


# --------------------------------------------------------------------------
# Apple Silicon
# --------------------------------------------------------------------------

def test_apple_silicon_reports_the_metal_accelerator(monkeypatch):
    """The bug the user reported: GPUs=0 on a machine with a working GPU."""
    fake_platform(monkeypatch, "Darwin", "arm64")
    no_nvidia_smi(monkeypatch)
    fake_providers(monkeypatch, COREML_PROVIDERS)

    hw = HardwareInfo.detect()

    assert hw.gpu_count == 1
    assert hw.accelerator == "coreml"
    assert "Metal/CoreML" in hw.accelerator_label
    assert "Metal/CoreML" in hw.gpu_names[0]


def test_metal_detection_does_not_need_tensorflow(monkeypatch):
    """It must fire in the TF-free plugin venv, where the old gate could not.

    Simulated by making the tensorflow-metal metadata lookup fail, which is
    exactly the state of requirements/plugin.txt.
    """
    import importlib.metadata as md
    fake_platform(monkeypatch, "Darwin", "arm64")
    no_nvidia_smi(monkeypatch)
    fake_providers(monkeypatch, COREML_PROVIDERS)

    def boom(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", boom)

    assert HardwareInfo.detect().gpu_count == 1


def test_apple_silicon_gpu_memory_is_a_unified_memory_budget(monkeypatch):
    """Reporting all of unified RAM as GPU memory over-sizes the batch."""
    fake_platform(monkeypatch, "Darwin", "arm64")
    no_nvidia_smi(monkeypatch)
    fake_providers(monkeypatch, COREML_PROVIDERS)
    monkeypatch.setattr(
        HardwareInfo, "_detect_total_ram_gb", staticmethod(lambda: 24.0))

    hw = HardwareInfo.detect()

    assert hw.gpu_memory_gb == [12.0]


def test_intel_mac_is_not_reported_as_apple_silicon(monkeypatch):
    fake_platform(monkeypatch, "Darwin", "x86_64")
    no_nvidia_smi(monkeypatch)
    fake_providers(monkeypatch, COREML_PROVIDERS)

    hw = HardwareInfo.detect()

    assert hw.gpu_count == 0
    assert hw.accelerator == "cpu"


# --------------------------------------------------------------------------
# NVIDIA — the path that must keep working
# --------------------------------------------------------------------------

def test_nvidia_names_and_memory_come_from_nvidia_smi(monkeypatch):
    fake_platform(monkeypatch, "Linux")
    fake_nvidia_smi(
        monkeypatch, ["NVIDIA A100", "NVIDIA A100"], [40.0, 40.0])
    fake_providers(monkeypatch, CUDA_PROVIDERS)

    hw = HardwareInfo.detect()

    assert hw.gpu_count == 2
    assert hw.gpu_names == ["NVIDIA A100", "NVIDIA A100"]
    assert hw.gpu_memory_gb == [40.0, 40.0]
    assert hw.accelerator == "cuda"
    assert hw.accelerator_label == "NVIDIA GPU (CUDA)"


def test_cpu_only_onnxruntime_build_does_not_claim_cuda(monkeypatch):
    """nvidia-smi sees GPUs but onnxruntime cannot use them -> CPU."""
    fake_platform(monkeypatch, "Linux")
    fake_nvidia_smi(monkeypatch, ["NVIDIA A100"], [40.0])
    fake_providers(monkeypatch, CPU_ONLY_PROVIDERS)

    hw = HardwareInfo.detect()

    assert hw.accelerator == "cpu"
    assert hw.accelerator_label == "CPU"
    assert hw.gpu_count == 0


def test_missing_onnxruntime_leaves_the_nvidia_path_intact(monkeypatch):
    """HardwareInfo must not become CPU-only just because ort is unimportable.

    Losing onnxruntime is a hard error at model load; it must not silently
    downgrade a working CUDA box to cpu_stream first.
    """
    fake_platform(monkeypatch, "Linux")
    fake_nvidia_smi(monkeypatch, ["NVIDIA A100"], [40.0])
    monkeypatch.setattr(
        HardwareInfo, "_detect_accelerator_kind", staticmethod(lambda: None))

    hw = HardwareInfo.detect()

    assert hw.gpu_count == 1
    assert hw.accelerator == "cuda"


@pytest.mark.parametrize("value", ["", "-1"])
def test_cuda_visible_devices_forces_cpu(monkeypatch, value):
    fake_platform(monkeypatch, "Darwin", "arm64")
    fake_nvidia_smi(monkeypatch, ["NVIDIA A100"], [40.0])
    fake_providers(monkeypatch, COREML_PROVIDERS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)

    hw = HardwareInfo.detect()

    assert hw.gpu_count == 0
    assert hw.accelerator == "cpu"


def test_cuda_visible_devices_selects_a_subset(monkeypatch):
    fake_platform(monkeypatch, "Linux")
    fake_nvidia_smi(
        monkeypatch, ["NVIDIA A100", "NVIDIA H100"], [40.0, 80.0])
    fake_providers(monkeypatch, CUDA_PROVIDERS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    hw = HardwareInfo.detect()

    assert hw.gpu_names == ["NVIDIA H100"]
    assert hw.gpu_memory_gb == [80.0]
    assert hw.accelerator == "cuda"


# --------------------------------------------------------------------------
# Forcing CPU
# --------------------------------------------------------------------------

def test_force_cpu_env_is_honoured_on_apple_silicon(monkeypatch):
    fake_platform(monkeypatch, "Darwin", "arm64")
    no_nvidia_smi(monkeypatch)
    fake_providers(monkeypatch, COREML_PROVIDERS)
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")

    hw = HardwareInfo.detect()

    assert hw.gpu_count == 0
    assert hw.accelerator == "cpu"
    assert hw.accelerator_label == "CPU"


def test_force_cpu_env_is_honoured_on_an_nvidia_box(monkeypatch):
    """The deliberate CPU pin must still win over a visible CUDA GPU."""
    fake_platform(monkeypatch, "Linux")
    fake_nvidia_smi(monkeypatch, ["NVIDIA A100"], [40.0])
    fake_providers(monkeypatch, CUDA_PROVIDERS)
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")

    hw = HardwareInfo.detect()

    assert hw.gpu_count == 0
    assert hw.gpu_names == []
    assert hw.accelerator == "cpu"


def test_explicit_cpu_provider_list_is_honoured(monkeypatch):
    fake_platform(monkeypatch, "Darwin", "arm64")
    no_nvidia_smi(monkeypatch)
    fake_providers(monkeypatch, COREML_PROVIDERS)
    monkeypatch.setenv("WINMOL_ONNX_PROVIDERS", "CPUExecutionProvider")

    hw = HardwareInfo.detect()

    assert hw.gpu_count == 0
    assert hw.accelerator == "cpu"


def test_disable_metal_escape_hatch_still_works(monkeypatch):
    fake_platform(monkeypatch, "Darwin", "arm64")
    no_nvidia_smi(monkeypatch)
    fake_providers(monkeypatch, COREML_PROVIDERS)
    monkeypatch.setenv("WINMOL_DISABLE_METAL", "1")

    assert HardwareInfo.detect().gpu_count == 0


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_disable_metal_does_not_discard_nvidia_gpus(monkeypatch, value):
    """WINMOL_DISABLE_METAL is a Metal-only escape hatch.

    It once shared a helper with the CUDA_VISIBLE_DEVICES check, so exporting
    it on a CUDA box reported GPUs=0 -> cpu_stream -> WINMOL_ONNX_FORCE_CPU=1:
    a silent, total loss of GPU acceleration.
    """
    fake_platform(monkeypatch, "Linux")
    fake_nvidia_smi(monkeypatch, ["NVIDIA A100"], [40.0])
    fake_providers(monkeypatch, CUDA_PROVIDERS)
    monkeypatch.setenv("WINMOL_DISABLE_METAL", value)

    hw = HardwareInfo.detect()

    assert hw.accelerator == "cuda"
    assert hw.gpu_count == 1
    assert hw.gpu_names == ["NVIDIA A100"]
    assert hw.gpu_memory_gb == [40.0]

    plan = build_execution_plan(Config(), hw, RASTER, "Trees")
    assert plan.prediction_mode == "stream"


# --------------------------------------------------------------------------
# onnx_runtime helpers
# --------------------------------------------------------------------------

def test_selected_providers_prefers_cuda_over_coreml(monkeypatch):
    fake_providers(
        monkeypatch, ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                      "CPUExecutionProvider"])
    assert onnx_runtime.selected_providers()[0] == "CUDAExecutionProvider"


def test_selected_providers_always_falls_back_to_cpu(monkeypatch):
    fake_providers(monkeypatch, COREML_PROVIDERS)
    fake_platform(monkeypatch, "Darwin", "arm64")
    assert onnx_runtime.selected_providers()[-1] == "CPUExecutionProvider"


def test_runtime_report_names_the_override(monkeypatch):
    fake_providers(monkeypatch, COREML_PROVIDERS)
    fake_platform(monkeypatch, "Darwin", "arm64")
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")

    report = onnx_runtime.runtime_report()

    assert report["override"] == "WINMOL_ONNX_FORCE_CPU"
    assert report["selected_providers"] == CPU_ONLY_PROVIDERS
    assert report["accelerator_label"] == "CPU"
    assert report["onnxruntime_version"]


# --------------------------------------------------------------------------
# ExecutionPlan must not route a Metal accelerator into a CUDA-only path
# --------------------------------------------------------------------------

RASTER = {
    "width": 8000, "height": 8000, "bands": 3, "dtype": "uint8",
    "pixel_size_x": 0.02, "pixel_size_y": 0.02, "estimated_input_gb": 0.6,
}


def coreml_hardware():
    return HardwareInfo(
        cpu_count=8,
        total_ram_gb=24.0,
        gpu_count=1,
        gpu_names=["Apple Silicon GPU (Metal/CoreML)"],
        gpu_memory_gb=[12.0],
        accelerator="coreml",
        accelerator_label="Apple Silicon GPU (Metal/CoreML)",
    )


@pytest.mark.parametrize("backend", ["auto", "single_gpu", "multi_gpu"])
def test_metal_never_selects_the_cuda_only_multi_gpu_path(backend):
    config = Config()
    config.prediction_backend = backend

    plan = build_execution_plan(config, coreml_hardware(), RASTER, "Trees")

    assert plan.prediction_mode == "stream"
    assert plan.gpu_workers == 1


def test_hardware_info_constructs_without_the_new_fields():
    """Existing positional construction must keep working."""
    hw = HardwareInfo(cpu_count=4, total_ram_gb=8.0, gpu_count=0)
    assert hw.accelerator == "cpu"
    assert hw.accelerator_label == "CPU"


# --------------------------------------------------------------------------
# The banner
# --------------------------------------------------------------------------

def _image_processor(model_path="model.onnx"):
    import winmol_run
    return winmol_run.ImageProcessing(
        model_path, "in.tif", "stem.tif", "out", "Stems")


def test_banner_has_no_tensorflow_or_cuda_noise(monkeypatch, capsys):
    """The TF-free plugin venv is the normal case, not an error."""
    import winmol_run

    fake_platform(monkeypatch, "Darwin", "arm64")
    fake_providers(monkeypatch, COREML_PROVIDERS)
    monkeypatch.setattr(winmol_run, "_import_tensorflow", lambda: None)
    monkeypatch.setattr(winmol_run, "_nvidia_driver_version", lambda: None)

    _image_processor().report_runtime_env()
    out = capsys.readouterr().out

    assert "Tensorflow error" not in out
    assert "Check CUDA environment" not in out
    assert "cuDNN" not in out
    assert "No NVIDIA GPU available" not in out
    assert "onnxruntime" in out
    assert "Apple Silicon GPU (Metal/CoreML)" in out


def test_banner_mentions_tensorflow_only_for_a_legacy_keras_model(
        monkeypatch, capsys):
    import winmol_run

    class FakeTF:
        __version__ = "2.16.2"

    fake_platform(monkeypatch, "Linux")
    fake_providers(monkeypatch, CPU_ONLY_PROVIDERS)
    monkeypatch.setattr(winmol_run, "_import_tensorflow", lambda: FakeTF())
    monkeypatch.setattr(winmol_run, "_nvidia_driver_version", lambda: None)

    _image_processor("model.onnx").report_runtime_env()
    assert "TensorFlow" not in capsys.readouterr().out

    _image_processor("legacy.hdf5").report_runtime_env()
    out = capsys.readouterr().out
    assert "TensorFlow 2.16.2" in out


def test_banner_reports_the_nvidia_driver_only_when_present(
        monkeypatch, capsys):
    import winmol_run

    fake_platform(monkeypatch, "Linux")
    fake_providers(monkeypatch, CUDA_PROVIDERS)
    monkeypatch.setattr(winmol_run, "_import_tensorflow", lambda: None)
    monkeypatch.setattr(winmol_run, "_nvidia_driver_version", lambda: "550.54")

    _image_processor().report_runtime_env()
    out = capsys.readouterr().out

    assert "550.54" in out
    assert "NVIDIA GPU (CUDA)" in out


def test_hardware_line_speaks_of_cpu_cores_and_an_accelerator(
        monkeypatch, capsys):
    fake_platform(monkeypatch, "Darwin", "arm64")
    no_nvidia_smi(monkeypatch)
    fake_providers(monkeypatch, COREML_PROVIDERS)

    _image_processor().detect_hardware()
    out = capsys.readouterr().out

    assert "CPU cores=" in out
    assert "GPUs=0" not in out
    assert "accelerator=Apple Silicon GPU (Metal/CoreML)" in out


def test_hardware_line_says_cpu_when_there_is_no_accelerator(
        monkeypatch, capsys):
    fake_platform(monkeypatch, "Linux")
    no_nvidia_smi(monkeypatch)
    fake_providers(monkeypatch, CPU_ONLY_PROVIDERS)

    _image_processor().detect_hardware()
    out = capsys.readouterr().out

    assert "GPUs=0" not in out
    assert "accelerator=CPU (no GPU acceleration available)" in out
