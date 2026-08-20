"""Move tile normalization and resampling into the ONNX graph.

Background: v0.5.0 resampled on the GPU with
``tf.image.resize(method='bicubic', antialias=False)``. The
reimplementation moved that to the CPU -- first skimage on the consumer,
then GDAL ``out_shape`` inside the read. Both changed the numbers, and
the GDAL route pushed 4x the bytes through GDAL's GLOBAL block cache
(default 5% of RAM), which is what makes throughput decay and then
collapse on a large ortho.

Prepending the work to the model graph removes the whole class of
problem: GDAL goes back to plain native-resolution reads, and the resize
runs on whatever device the session is bound to.

Numerically this is NOT an approximation of v0.5.0 -- it is the same
operator. Verified against ``tf.image.resize`` bicubic on real Tegel
tiles:

    cubic_coeff_a=-0.5, half_pixel, exclude_outside=1
        max|d| 4.2e-07   mean|d| 3.4e-08   pixels >1/255: 0.0000%

which is float32 rounding. The three settings all matter: PyTorch's
bicubic hardcodes a=-0.75 and cannot reproduce this at all (max|d| 0.04,
20-30% of pixels), and leaving exclude_outside at its 0 default gives
max|d| 0.011.

Feeding uint8 also cuts PCIe traffic 4x versus sending float32.
"""
import hashlib
import os

import numpy as np
import onnx
from onnx import TensorProto, helper

#: Keys cubic coefficient TensorFlow uses. Do not "modernise" to -0.75:
#: that is the OpenCV/PIL/PyTorch convention and yields different pixels.
TF_CUBIC_COEFF_A = -0.5


def _model_input(model):
    initialisers = {i.name for i in model.graph.initializer}
    for vi in model.graph.input:
        if vi.name not in initialisers:
            return vi
    return model.graph.input[0]


def build_preprocessed_model(model_path, target_hw, out_path=None,
                             antialias=False):
    """Return a path to `model_path` with uint8 -> normalize -> resize in front.

    The wrapped model takes NHWC uint8 tiles at ANY spatial size (dims stay
    dynamic) and resizes them to ``target_hw`` inside the graph.

    antialias=True widens the kernel with the downsampling factor (ONNX
    Resize `antialias`, opset >= 18) -- GDAL-like AA semantics, but
    deterministic and portable. False is the v0.5.0 behavior.
    """
    th, tw = int(target_hw[0]), int(target_hw[1])
    model = onnx.load(str(model_path))
    inp = _model_input(model)
    dims = [d.dim_value if d.HasField('dim_value') else None
            for d in inp.type.tensor_type.shape.dim]
    nchw = len(dims) == 4 and dims[1] == 3

    if out_path is None:
        key = hashlib.sha256(
            f"{os.path.realpath(model_path)}|{os.path.getmtime(model_path)}"
            f"|{th}x{tw}|{nchw}|{TF_CUBIC_COEFF_A}|aa{int(bool(antialias))}"
            .encode()
        ).hexdigest()[:16]
        out_path = os.path.join(
            os.path.dirname(os.path.realpath(model_path)),
            f".winmol_pre_{key}.onnx")
    if os.path.exists(out_path):
        return out_path

    src = "winmol_pre_input"
    nodes = [
        # NHWC uint8 -> NCHW uint8. Resize needs NCHW regardless of what
        # the wrapped model wants; we transpose back below if it is NHWC.
        helper.make_node("Transpose", [src], ["pre_nchw"],
                         perm=[0, 3, 1, 2], name="winmol_pre_transpose"),
        helper.make_node("Cast", ["pre_nchw"], ["pre_f32"],
                         to=TensorProto.FLOAT, name="winmol_pre_cast"),
        helper.make_node("Div", ["pre_f32", "winmol_pre_255"], ["pre_norm"],
                         name="winmol_pre_div"),
        # Target size is built from the RUNTIME batch/channel dims so the
        # graph stays valid for any batch size.
        helper.make_node("Shape", ["pre_nchw"], ["pre_shape"],
                         name="winmol_pre_shape"),
        helper.make_node("Slice", ["pre_shape", "winmol_pre_0",
                                   "winmol_pre_2", "winmol_pre_ax0"],
                         ["pre_nc"], name="winmol_pre_slice"),
        helper.make_node("Concat", ["pre_nc", "winmol_pre_hw"],
                         ["pre_sizes"], axis=0, name="winmol_pre_concat"),
        helper.make_node(
            "Resize", ["pre_norm", "", "", "pre_sizes"], ["pre_resized"],
            mode="cubic",
            cubic_coeff_a=TF_CUBIC_COEFF_A,
            coordinate_transformation_mode="half_pixel",
            exclude_outside=1,
            nearest_mode="floor",
            name="winmol_pre_resize",
            **({"antialias": 1} if antialias else {})),
    ]
    if nchw:
        nodes[-1].output[0] = inp.name
    else:
        nodes.append(helper.make_node(
            "Transpose", ["pre_resized"], [inp.name], perm=[0, 2, 3, 1],
            name="winmol_pre_untranspose"))

    inits = [
        helper.make_tensor("winmol_pre_255", TensorProto.FLOAT, [],
                           [255.0]),
        helper.make_tensor("winmol_pre_0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("winmol_pre_2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("winmol_pre_ax0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("winmol_pre_hw", TensorProto.INT64, [2],
                           [th, tw]),
    ]

    new_input = helper.make_tensor_value_info(
        src, TensorProto.UINT8, ["N", "H", "W", 3])
    model.graph.input.remove(inp)
    model.graph.input.insert(0, new_input)
    model.graph.initializer.extend(inits)
    model.graph.node.insert(0, nodes[0])
    for i, n in enumerate(nodes[1:], start=1):
        model.graph.node.insert(i, n)

    # Resize with `sizes` and exclude_outside needs opset >= 11; cubic_coeff_a
    # semantics settled by 13; `antialias` exists from 18. Raise only if the
    # model is older than what the built node needs.
    min_opset = 18 if antialias else 13
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx") and op.version < min_opset:
            op.version = min_opset
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return out_path


def as_uint8_nhwc(tiles):
    """Stack raw tiles into the NHWC uint8 batch the wrapped graph wants."""
    batch = np.stack([np.ascontiguousarray(t) for t in tiles], axis=0)
    if batch.ndim == 3:
        batch = batch[..., None]
    if batch.shape[-1] > 3:
        batch = batch[..., :3]
    return np.ascontiguousarray(batch, dtype=np.uint8)
