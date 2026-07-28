"""Unit tests for the ONNX branch of ``IO.load_model_from_path`` and the
vendored ``utils.onnx_runtime.OnnxSegmenter``.

Dispatch tests inject a fake segmenter so they run without onnxruntime. A
separate real-inference test builds a tiny NHWC and NCHW ONNX model on the fly
(onnxruntime required) to prove the segmenter is layout-aware and returns NHWC.

The runtime is TensorFlow-free: a non-``.onnx`` model path (a legacy Keras
``.hdf5``/``.h5``/``.keras``) is rejected with a RuntimeError that points at
the HDF5->ONNX converter, without importing TensorFlow at all.
"""

import sys
import types

import numpy as np
import pytest

from utils import IO


@pytest.fixture
def stub_onnx_segmenter(monkeypatch):
    """Replace the vendored OnnxSegmenter with a recorder."""
    calls = []

    class FakeOnnxSegmenter:
        def __init__(self, model_path, providers=None):
            self.model_path = model_path
            self.providers = providers
            calls.append(model_path)

        def predict_on_batch(self, x):  # pragma: no cover - interface marker
            return x

    mod = types.ModuleType("utils.onnx_runtime")
    mod.OnnxSegmenter = FakeOnnxSegmenter
    monkeypatch.setitem(sys.modules, "utils.onnx_runtime", mod)
    return calls, FakeOnnxSegmenter


def test_onnx_path_dispatches_to_onnx_segmenter(stub_onnx_segmenter):
    calls, FakeOnnxSegmenter = stub_onnx_segmenter
    model = IO.load_model_from_path("/models/deeplabv3plus.onnx")
    assert isinstance(model, FakeOnnxSegmenter)
    assert calls == ["/models/deeplabv3plus.onnx"]
    assert hasattr(model, "predict_on_batch")


def test_onnx_extension_is_case_insensitive(stub_onnx_segmenter):
    calls, FakeOnnxSegmenter = stub_onnx_segmenter
    model = IO.load_model_from_path("/models/MODEL.ONNX")
    assert isinstance(model, FakeOnnxSegmenter)
    assert calls == ["/models/MODEL.ONNX"]


def test_onnx_without_onnxruntime_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "utils.onnx_runtime", None)
    with pytest.raises(RuntimeError) as exc:
        IO.load_model_from_path("/models/deeplabv3plus.onnx")
    assert "onnxruntime" in str(exc.value).lower()


@pytest.mark.parametrize(
    "model_path",
    ["/models/legacy.hdf5", "/models/legacy.h5", "/models/legacy.keras",
     "/models/LEGACY.HDF5"])
def test_non_onnx_model_raises_converter_naming_error(model_path):
    """The runtime is TensorFlow-free: a legacy Keras model must fail fast
    with a clear message pointing at the HDF5->ONNX converter, and it must
    NOT import TensorFlow to do so (no TF needed to run this test)."""
    tf_loaded_before = "tensorflow" in sys.modules
    with pytest.raises(RuntimeError) as exc:
        IO.load_model_from_path(model_path)
    msg = str(exc.value)
    assert "scripts/convert_models_to_onnx.py" in msg
    assert ".onnx" in msg
    # The rejection is a pure string check -- the call imports no TensorFlow.
    assert ("tensorflow" in sys.modules) == tf_loaded_before


# --- vendored segmenter: real inference, layout-aware ----------------------

def _tiny_onnx(path, layout):
    """Build a 1-conv sigmoid ONNX model with NHWC or NCHW I/O (16x16 for
    speed) so we can exercise the segmenter without a large fixture model."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    s = 16
    if layout == "NCHW":
        in_shape, out_shape = ["b", 3, s, s], ["b", 1, s, s]
        w = helper.make_tensor("w", TensorProto.FLOAT, [1, 3, 1, 1],
                               np.zeros(3, dtype=np.float32))
        conv = helper.make_node("Conv", ["input", "w"], ["c"])
        act = helper.make_node("Sigmoid", ["c"], ["output"])
    else:  # NHWC: transpose in -> conv (NCHW) -> transpose out
        in_shape, out_shape = ["b", s, s, 3], ["b", s, s, 1]
        w = helper.make_tensor("w", TensorProto.FLOAT, [1, 3, 1, 1],
                               np.zeros(3, dtype=np.float32))
        t1 = helper.make_node("Transpose", ["input"], ["nchw"],
                              perm=[0, 3, 1, 2])
        conv = helper.make_node("Conv", ["nchw", "w"], ["c"])
        act = helper.make_node("Sigmoid", ["c"], ["nchw_out"])
        t2 = helper.make_node("Transpose", ["nchw_out"], ["output"],
                              perm=[0, 2, 3, 1])
    nodes = [conv, act] if layout == "NCHW" else [t1, conv, act, t2]
    graph = helper.make_graph(
        nodes, "m",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        [w])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return str(path)


@pytest.mark.parametrize("layout", ["NHWC", "NCHW"])
def test_vendored_segmenter_is_layout_aware(tmp_path, layout):
    pytest.importorskip("onnxruntime")
    from utils.onnx_runtime import OnnxSegmenter
    path = _tiny_onnx(tmp_path / f"{layout}.onnx", layout)
    seg = OnnxSegmenter(path)
    assert seg.input_layout == layout
    x = np.random.rand(2, 16, 16, 3).astype("float32")   # always NHWC in
    y = seg.predict_on_batch(x)
    assert y.shape == (2, 16, 16, 1)                      # always NHWC out
    assert y.dtype == np.float32
    assert (y >= 0).all() and (y <= 1).all()             # sigmoid
