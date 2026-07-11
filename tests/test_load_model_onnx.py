"""Unit tests for the ONNX branch of ``IO.load_model_from_path``.

These verify the *dispatch* logic only: a ``.onnx`` path must be loaded via
``winmol_unet.runtime.OnnxSegmenter`` (the runtime adapter that duck-types the
Keras model), and a missing ``winmol_unet`` install must raise a clear,
actionable error. A stub ``winmol_unet`` module is injected into ``sys.modules``
so the tests run without onnxruntime or the real package present.

The Keras (``.hdf5``) branch is exercised by the end-to-end pipeline test and is
deliberately not imported here (it would pull in TensorFlow).
"""

import sys
import types

import pytest

from utils import IO


@pytest.fixture
def stub_onnx_segmenter(monkeypatch):
    """Inject a fake ``winmol_unet.runtime.OnnxSegmenter`` and record calls."""
    calls = []

    class FakeOnnxSegmenter:
        def __init__(self, model_path, providers=None):
            self.model_path = model_path
            self.providers = providers
            calls.append(model_path)

        def predict_on_batch(self, x):  # pragma: no cover - interface marker
            return x

    pkg = types.ModuleType("winmol_unet")
    runtime = types.ModuleType("winmol_unet.runtime")
    runtime.OnnxSegmenter = FakeOnnxSegmenter
    pkg.runtime = runtime
    monkeypatch.setitem(sys.modules, "winmol_unet", pkg)
    monkeypatch.setitem(sys.modules, "winmol_unet.runtime", runtime)
    return calls, FakeOnnxSegmenter


def test_onnx_path_dispatches_to_onnx_segmenter(stub_onnx_segmenter):
    calls, FakeOnnxSegmenter = stub_onnx_segmenter
    model = IO.load_model_from_path("/models/deeplabv3plus.onnx")
    assert isinstance(model, FakeOnnxSegmenter)
    assert calls == ["/models/deeplabv3plus.onnx"]
    # It must expose the interface the analyzer calls on every model object.
    assert hasattr(model, "predict_on_batch")


def test_onnx_extension_is_case_insensitive(stub_onnx_segmenter):
    calls, FakeOnnxSegmenter = stub_onnx_segmenter
    model = IO.load_model_from_path("/models/MODEL.ONNX")
    assert isinstance(model, FakeOnnxSegmenter)
    assert calls == ["/models/MODEL.ONNX"]


def test_onnx_without_winmol_unet_raises_helpful_error(monkeypatch):
    # Ensure winmol_unet cannot be imported.
    monkeypatch.setitem(sys.modules, "winmol_unet", None)
    monkeypatch.setitem(sys.modules, "winmol_unet.runtime", None)
    with pytest.raises(RuntimeError) as exc:
        IO.load_model_from_path("/models/deeplabv3plus.onnx")
    msg = str(exc.value).lower()
    assert "winmol_unet" in msg
    assert ".onnx" in msg or "onnx" in msg
