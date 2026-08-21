"""The in-graph cubic resize must survive a DOWNSCALE on the CPU.

Regression for the crash that took out the CPU container on any
orthomosaic finer than the model's target GSD (Bremerhagen, 1.23 cm/px):

    FAIL : Non-zero status code returned while running Resize node.
    Name:'winmol_pre_resize' upsamplebase.h:579 ScalesValidation
    'Cubic' mode only supports: ... other scales >= 1 without antialias

`graph` emits the Resize in NCHW, which every provider accepts. With a
QUANTIZED model attached, onnxruntime's NhwcTransformer rewrites the
subgraph to NHWC to reach the CPU int8 kernels, and cubic Resize refuses
NHWC downsampling without `antialias`. It runs only at ORT_ENABLE_ALL --
the default -- so nothing below that level reproduces it, and CUDA never
does. See utils.onnx_runtime._session_options.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def _qdq_model_with_ingraph_resize(target=8):
    """The smallest graph that reproduces it: NCHW cubic Resize feeding a
    QDQ (quantized) subgraph, which is what pulls NhwcTransformer in."""
    from onnx import TensorProto as T
    from onnx import helper

    nodes = [
        helper.make_node("Transpose", ["X"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node("Cast", ["nchw"], ["f32"], to=T.FLOAT),
        helper.make_node("Div", ["f32", "c255"], ["norm"]),
        helper.make_node("Shape", ["nchw"], ["shp"]),
        helper.make_node("Slice", ["shp", "c0", "c2", "cax"], ["nc"]),
        helper.make_node("Concat", ["nc", "chw"], ["sizes"], axis=0),
        helper.make_node(
            "Resize", ["norm", "", "", "sizes"], ["resized"],
            mode="cubic", cubic_coeff_a=-0.5, exclude_outside=1,
            coordinate_transformation_mode="half_pixel",
            nearest_mode="floor", name="winmol_pre_resize"),
        # A full QDQ Conv -- quantized ACTIVATION *and* quantized WEIGHTS,
        # requantized output. Anything less (a plain Conv on a dequantized
        # input) does not fuse to QLinearConv, NhwcTransformer never fires,
        # and the bug does not reproduce.
        helper.make_node("QuantizeLinear", ["resized", "sc", "zp"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "sc", "zp"], ["xdq"]),
        helper.make_node("DequantizeLinear", ["wq", "wsc", "wzp"], ["wdq"]),
        helper.make_node("Conv", ["xdq", "wdq"], ["conv"], kernel_shape=[1, 1]),
        helper.make_node("QuantizeLinear", ["conv", "sc", "zp"], ["yq"]),
        helper.make_node("DequantizeLinear", ["yq", "sc", "zp"], ["Y"]),
    ]
    inits = [
        helper.make_tensor("c255", T.FLOAT, [], [255.0]),
        helper.make_tensor("c0", T.INT64, [1], [0]),
        helper.make_tensor("c2", T.INT64, [1], [2]),
        helper.make_tensor("cax", T.INT64, [1], [0]),
        helper.make_tensor("chw", T.INT64, [2], [target, target]),
        helper.make_tensor("sc", T.FLOAT, [], [1.0 / 255]),
        helper.make_tensor("zp", T.UINT8, [], [0]),
        helper.make_tensor("wq", T.UINT8, [1, 3, 1, 1], [1, 1, 1]),
        helper.make_tensor("wsc", T.FLOAT, [], [1.0 / 127]),
        helper.make_tensor("wzp", T.UINT8, [], [0]),
    ]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("X", T.UINT8, ["N", "H", "W", 3])],
        [helper.make_tensor_value_info("Y", T.FLOAT, ["N", 1, target, target])],
        initializer=inits)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)]).SerializeToString()


def _run(model, opts, size):
    sess = ort.InferenceSession(
        model, opts, providers=["CPUExecutionProvider"])
    tile = np.zeros((1, size, size, 3), dtype=np.uint8)
    return sess.run(None, {"X": tile})[0]


def test_default_options_reproduce_the_downscale_crash():
    """Guards the guard: if onnxruntime ever stops rejecting this, the
    fix below is dead weight and should be revisited rather than kept
    for a reason that no longer holds."""
    model = _qdq_model_with_ingraph_resize()
    with pytest.raises(Exception, match="(?i)resize|cubic|scale"):
        _run(model, ort.SessionOptions(), size=64)      # 64 -> 8, scale 0.125


def test_project_options_allow_the_downscale():
    from utils.onnx_runtime import _session_options
    model = _qdq_model_with_ingraph_resize()
    out = _run(model, _session_options(), size=64)
    assert out.shape == (1, 1, 8, 8)


def test_upscale_was_never_affected():
    """The bug needed a downscale; upsampling always worked, so this
    pins that the fix did not change the working direction."""
    from utils.onnx_runtime import _session_options
    model = _qdq_model_with_ingraph_resize()
    for opts in (ort.SessionOptions(), _session_options()):
        assert _run(model, opts, size=4).shape == (1, 1, 8, 8)  # 4 -> 8
