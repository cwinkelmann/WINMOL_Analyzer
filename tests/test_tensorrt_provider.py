"""TensorRT is opt-in, engine-cached, and never selects itself.

It is the one optimisation in this lineage that is NOT bit-identical to
what it replaces: measured on 8 real R13 tiles, 3 px of 2,097,152 flipped
across the 0.5 threshold (IoU 0.999542) but max|diff| reached 0.398. So
the wiring has to guarantee it stays off unless an operator asks, and
that when it IS on it caches its engines -- a cold build measured ~14 s
per input shape, and edge tiles have shapes of their own.

These run without a GPU: they exercise the selection and option-building
logic, not onnxruntime.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("WINMOL_ONNX_TENSORRT", "WINMOL_TRT_CACHE",
                "WINMOL_ONNX_PROVIDERS", "WINMOL_ONNX_FORCE_CPU"):
        monkeypatch.delenv(var, raising=False)


def test_tensorrt_is_off_by_default(monkeypatch):
    from utils import onnx_runtime as rt
    monkeypatch.setattr(rt, "_available_providers", lambda: [
        "TensorrtExecutionProvider", "CUDAExecutionProvider",
        "CPUExecutionProvider"])
    assert rt.tensorrt_enabled() is False
    names = rt.provider_names(rt.selected_providers())
    assert "TensorrtExecutionProvider" not in names, (
        "TensorRT must never select itself -- it is not bit-identical")
    assert names == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_opting_in_puts_tensorrt_in_front_of_cuda(monkeypatch, tmp_path):
    from utils import onnx_runtime as rt
    monkeypatch.setattr(rt, "_available_providers", lambda: [
        "TensorrtExecutionProvider", "CUDAExecutionProvider",
        "CPUExecutionProvider"])
    monkeypatch.setenv("WINMOL_ONNX_TENSORRT", "1")
    monkeypatch.setenv("WINMOL_TRT_CACHE", str(tmp_path / "engines"))
    names = rt.provider_names(rt.selected_providers())
    # In FRONT of CUDA, not instead of it: ops TensorRT will not take
    # (the graph's cubic Resize is the likely one) must fall through to
    # CUDA rather than to the CPU.
    assert names == ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                     "CPUExecutionProvider"]


def test_opting_in_without_the_runtime_present_changes_nothing(monkeypatch):
    """The shim is listed even when TensorRT cannot load; if the EP is not
    actually available we must not pretend."""
    from utils import onnx_runtime as rt
    monkeypatch.setattr(rt, "_available_providers", lambda: [
        "CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setenv("WINMOL_ONNX_TENSORRT", "1")
    names = rt.provider_names(rt.selected_providers())
    assert names == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_engine_cache_options_are_attached(monkeypatch, tmp_path):
    from utils import onnx_runtime as rt
    cache = tmp_path / "engines"
    monkeypatch.setenv("WINMOL_TRT_CACHE", str(cache))
    entries = rt._with_trt_options(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider"])
    trt = entries[0]
    assert isinstance(trt, tuple), "TensorRT needs the (name, options) form"
    name, opts = trt
    assert name == "TensorrtExecutionProvider"
    assert opts["trt_engine_cache_enable"] is True
    assert opts["trt_engine_cache_path"] == str(cache)
    assert cache.is_dir(), "the cache directory must exist before the session"
    # everything else stays a bare name
    assert entries[1] == "CUDAExecutionProvider"


def test_provider_names_normalises_both_forms():
    from utils.onnx_runtime import provider_names
    assert provider_names([
        ("TensorrtExecutionProvider", {"trt_engine_cache_enable": True}),
        "CUDAExecutionProvider",
    ]) == ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    assert provider_names([]) == []
    assert provider_names(None) == []


def test_explicit_provider_override_still_wins(monkeypatch):
    from utils import onnx_runtime as rt
    monkeypatch.setenv("WINMOL_ONNX_PROVIDERS", "CPUExecutionProvider")
    assert rt.provider_names(rt.selected_providers()) == [
        "CPUExecutionProvider"]
