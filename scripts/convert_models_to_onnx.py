#!/usr/bin/env python3
"""Convert the Zenodo Keras (HDF5) U-Net models to ONNX and verify parity.

Offline, dev-only tool (needs TensorFlow + tf2onnx from
requirements/convert.txt). The analyzer itself runs the resulting .onnx via the
vendored OnnxSegmenter and needs no TensorFlow.

The models are Keras 2 HDF5, NHWC, input (512,512,3) -> sigmoid output
(512,512,1). We emit ONNX keeping the native NHWC layout (opset 17, I/O named
"input"/"output"); the vendored OnnxSegmenter is layout-aware and feeds NHWC
directly (it also handles NCHW models such as the PyTorch exports). We then
assert the ONNX output matches the Keras output within a tight tolerance on
real orthomosaic tiles.

Usage:
    PYTHONHASHSEED=0 python scripts/convert_models_to_onnx.py \
        [--only General] [--out-dir standalone/model_onnx]
"""

import argparse
import os
import sys

# Keras-2 HDF5 needs the legacy shim under TF>=2.16; set before importing TF.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np              # noqa: E402

MODEL_DIR = os.path.join(REPO, "standalone", "model")
# config.json key -> local Keras file (the four shipped Zenodo models)
MODELS = {
    "General": "model_UNet_GenDS_512_2023-02-27_211141.hdf5",
    "Beech": "model_UNet_SpecDS_Beech_512_2023-02-28_042751.hdf5",
    "Spruce": "model_UNet_SpecDS_Spruce_512_2023-02-27_061925.hdf5",
    "Spruce_Deadwood":
        "model_UNet_SpecDS_Spruce_Deadwood_512_2024-12-19_194758.hdf5",
}
FIXTURE_CROP = os.path.join(REPO, "tests", "fixtures", "crop_input.tif")


def _real_tiles(n=4, size=512):
    """A batch of NHWC [N,size,size,3] float32 [0,1] tiles from the fixture."""
    import rasterio   # only needed for the parity check
    with rasterio.open(FIXTURE_CROP) as s:
        img = (s.read([1, 2, 3]).transpose(1, 2, 0) / 255.0).astype("float32")
    h, w, _ = img.shape
    tiles = []
    offs = [(0, 0), (h - size, 0), (0, w - size), (h - size, w - size)]
    for r, c in offs[:n]:
        r, c = max(0, r), max(0, c)
        tiles.append(img[r:r + size, c:c + size, :])
    return np.ascontiguousarray(np.stack(tiles))


def convert_one(name, out_dir, model_dir=MODEL_DIR):
    import tensorflow as tf   # noqa: F401 (env parity)
    import tf2onnx
    from tensorflow import keras

    hdf5 = os.path.join(model_dir, MODELS[name])
    if not os.path.exists(hdf5):
        print(f"[skip] {name}: {hdf5} not found")
        return None
    onnx_path = os.path.join(out_dir, f"{name}.onnx")

    print(f"[load] {name} <- {os.path.basename(hdf5)}")
    model = keras.models.load_model(hdf5, compile=False)
    in_shape = model.input_shape       # (None,512,512,3)
    assert tuple(in_shape[1:]) == (512, 512, 3), in_shape

    in_name = model.inputs[0].name.split(":")[0]
    out_name = model.outputs[0].name.split(":")[0]
    print(f"[convert] keras I/O names: in={in_name} out={out_name} "
          f"-> NHWC input/output, opset 17")

    spec = (tf.TensorSpec((None, 512, 512, 3), tf.float32, name=in_name),)
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=17,
        output_path=onnx_path,
    )
    _normalize_io_names(onnx_path)
    print(f"[ok] wrote {onnx_path}")
    return onnx_path, model


def _normalize_io_names(onnx_path):
    """Rename graph I/O to the stable contract names 'input'/'output'."""
    import onnx
    m = onnx.load(onnx_path)
    g = m.graph
    old_in = g.input[0].name
    old_out = g.output[0].name
    ren = {old_in: "input", old_out: "output"}
    for node in g.node:
        node.input[:] = [ren.get(x, x) for x in node.input]
        node.output[:] = [ren.get(x, x) for x in node.output]
    g.input[0].name = "input"
    g.output[0].name = "output"
    onnx.save(m, onnx_path)


def parity(name, onnx_path, keras_model):
    import onnx
    import onnxruntime as ort

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    din = [d.dim_param or d.dim_value
           for d in m.graph.input[0].type.tensor_type.shape.dim]
    dout = [d.dim_param or d.dim_value
            for d in m.graph.output[0].type.tensor_type.shape.dim]
    print(f"[graph] input {m.graph.input[0].name} {din} | "
          f"output {m.graph.output[0].name} {dout}")

    x_nhwc = _real_tiles()
    y_keras = np.asarray(keras_model.predict(x_nhwc, verbose=0))  # NHWC

    sess = ort.InferenceSession(onnx_path,
                                providers=["CPUExecutionProvider"])
    # layout-aware feed/read (matches the vendored OnnxSegmenter design)
    in_shape = sess.get_inputs()[0].shape        # symbolic batch + dims
    x = x_nhwc if in_shape[-1] == 3 else np.ascontiguousarray(
        x_nhwc.transpose(0, 3, 1, 2))
    y = sess.run(["output"], {"input": np.ascontiguousarray(x)})[0]
    out_shape = sess.get_outputs()[0].shape
    y_onnx = y if out_shape[-1] == 1 else y.transpose(0, 2, 3, 1)

    d = np.abs(y_keras - y_onnx)
    print(f"[parity] {name}: shape keras {y_keras.shape} onnx {y_onnx.shape} | "
          f"max {d.max():.3e} mean {d.mean():.3e} "
          f"| keras range [{y_keras.min():.3f},{y_keras.max():.3f}]")
    # The pipeline consumes the BINARIZED mask (>0.5); this is the decisive
    # parity metric. Raw-probability max diff is a loose sanity bound only
    # (TF vs onnxruntime fp accumulation in a deep conv net).
    disagree = float(np.mean((y_keras > 0.5) != (y_onnx > 0.5)))
    print(f"[parity] {name}: 0.5-threshold pixel disagreement "
          f"{disagree * 100:.4f}%")
    return float(d.max()), disagree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="convert a single key")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="dir holding the source .hdf5 files")
    ap.add_argument("--out-dir",
                    default=os.path.join(REPO, "standalone", "model_onnx"))
    ap.add_argument("--raw-tol", type=float, default=1e-3,
                    help="sanity bound on raw-probability max diff")
    ap.add_argument("--no-parity", action="store_true",
                    help="convert only, skip the fixture-based parity check "
                         "(used in the CI image build, which has no fixture)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    keys = [args.only] if args.only else list(MODELS)
    worst_raw, worst_disagree = 0.0, 0.0
    for name in keys:
        res = convert_one(name, args.out_dir, model_dir=args.model_dir)
        if res is None:
            continue
        onnx_path, model = res
        if args.no_parity:
            print(f"[convert-only] {name} -> {onnx_path}\n")
            continue
        raw, disagree = parity(name, onnx_path, model)
        worst_raw = max(worst_raw, raw)
        worst_disagree = max(worst_disagree, disagree)
        print()
    if args.no_parity:
        return
    # Gate: the binarized stem mask must be identical (0 disagreement);
    # raw probability diff is only sanity-bounded.
    ok = worst_disagree == 0.0 and worst_raw <= args.raw_tol
    print(f"[gate] worst 0.5-threshold disagreement: {worst_disagree*100:.4f}% "
          f"| worst raw max-diff: {worst_raw:.3e} (<= {args.raw_tol:.1e}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
