"""A present GPU is not a usable GPU.

``detect_device`` picks the model PRECISION (``_DEVICE_PRECISION``:
fp16 for gpu, int8 for cpu). The managed venv is built from
requirements/cpu.txt unless the user opts into the GPU runtime, so on any
NVIDIA box a default install has a card and a CPU-only onnxruntime.
Judging by the card alone handed that install the fp16 variant it cannot
accelerate -- it runs (ORT converts the fp16 weights to fp32 once at
session load), just as the wrong, slower variant.
"""

import sys
import types

import pytest

from plugin_utils import model_registry as mr


@pytest.fixture(autouse=True)
def _no_forced_device(monkeypatch):
    monkeypatch.delenv("WINMOL_DEVICE", raising=False)
    # Keep every test off real hardware and off Apple-Silicon short-circuit.
    monkeypatch.setattr(mr.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mr, "_probe_nvidia", lambda: "gpu")


def _marker(monkeypatch, variant):
    """Fake installer.installed_variant for the lazily-imported call."""
    fake = types.ModuleType("plugin_utils.installer")
    fake.installed_variant = lambda venv: variant
    monkeypatch.setitem(sys.modules, "plugin_utils.installer", fake)


def test_cpu_only_venv_on_an_nvidia_box_selects_cpu(monkeypatch):
    """The reported case: card present, CPU-only runtime installed."""
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    _marker(monkeypatch, "cpu")
    assert mr.detect_device("/some/venv") == "cpu"
    assert mr._DEVICE_PRECISION["cpu"] == "int8"


def test_gpu_venv_on_an_nvidia_box_still_selects_gpu(monkeypatch):
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    _marker(monkeypatch, "gpu")
    assert mr.detect_device("/some/venv") == "gpu"


def test_no_sentinel_falls_back_to_the_probe(monkeypatch):
    """Env not built yet: trust the card so a first run can still offer
    the GPU model it is about to be able to use."""
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    _marker(monkeypatch, None)
    assert mr.detect_device("/some/venv") == "gpu"
    assert mr.detect_device() == "gpu"      # no venv path at all


def test_imported_onnxruntime_is_the_truth_and_beats_the_sentinel(
    monkeypatch,
):
    """In the compute child the runtime itself is authoritative -- a
    stale sentinel must not override what the session can actually do."""
    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    _marker(monkeypatch, "gpu")             # sentinel disagrees
    assert mr.detect_device("/some/venv") == "cpu"

    ort.get_available_providers = lambda: [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert mr.detect_device("/some/venv") == "gpu"


def test_forced_device_still_wins(monkeypatch):
    monkeypatch.setenv("WINMOL_DEVICE", "gpu")
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    _marker(monkeypatch, "cpu")
    assert mr.detect_device("/some/venv") == "gpu"


def test_cpu_verdict_skips_the_nvidia_probe_entirely(monkeypatch):
    """_probe_nvidia carries a 20 s timeout; a CPU-only env must not pay
    it just to be told about a card it cannot use."""
    called = []
    monkeypatch.setattr(
        mr, "_probe_nvidia", lambda: (called.append(1), "gpu")[1])
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    _marker(monkeypatch, "cpu")

    assert mr.detect_device("/some/venv") == "cpu"
    assert called == []


def test_broken_installer_import_degrades_to_the_probe(monkeypatch):
    """Never let a sentinel read failure block model selection."""
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    fake = types.ModuleType("plugin_utils.installer")

    def _boom(venv):
        raise OSError("unreadable")

    fake.installed_variant = _boom
    monkeypatch.setitem(sys.modules, "plugin_utils.installer", fake)
    assert mr.detect_device("/some/venv") == "gpu"
