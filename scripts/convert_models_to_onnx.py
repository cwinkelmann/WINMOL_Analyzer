#!/usr/bin/env python3
"""Convert a Keras (HDF5) U-Net segmentation model to ONNX.

Offline, dev-only tool -- needs TensorFlow + tf2onnx, which are NOT part
of the runtime requirements. The analyzer itself runs the resulting
``.onnx`` via the vendored ``utils.onnx_runtime.OnnxSegmenter`` and needs
no TensorFlow; ``utils/IO.py`` rejects legacy ``.hdf5``/``.h5`` models
and points here.

The WINMOL U-Net models are Keras 2 HDF5, NHWC, input (512,512,3) ->
sigmoid output (512,512,1). Conversion keeps the native NHWC layout;
``OnnxSegmenter`` reads the I/O layout straight from the ONNX graph, so
the exported model works without further changes. The graph's I/O
tensors are renamed to the stable names "input"/"output" for a
predictable contract.

Usage:
    python scripts/convert_models_to_onnx.py MODEL.hdf5
    python scripts/convert_models_to_onnx.py MODEL.hdf5 -o out/model.onnx
    python scripts/convert_models_to_onnx.py a.hdf5 b.h5 --out-dir onnx/
"""
import argparse
import os
import sys

# Keras-2 HDF5 needs the legacy shim under TF>=2.16; harmless without TF,
# and must be set before TensorFlow is ever imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _output_path(input_path, out_dir, output):
    if output:
        return output
    base = os.path.splitext(os.path.basename(input_path))[0]
    directory = out_dir or os.path.dirname(input_path) or "."
    return os.path.join(directory, base + ".onnx")


def _normalize_io_names(onnx_path):
    """Rename the graph's I/O tensors to the stable names 'input'/'output'."""
    import onnx

    m = onnx.load(onnx_path)
    g = m.graph
    old_in, old_out = g.input[0].name, g.output[0].name
    renamed = {old_in: "input", old_out: "output"}
    for node in g.node:
        node.input[:] = [renamed.get(x, x) for x in node.input]
        node.output[:] = [renamed.get(x, x) for x in node.output]
    g.input[0].name = "input"
    g.output[0].name = "output"
    onnx.save(m, onnx_path)


def convert(input_path, output_path, opset=17):
    """Load a Keras HDF5 model and export it to ONNX at output_path.

    TensorFlow/tf2onnx are imported here rather than at module scope, so
    ``--help`` and argument parsing work without TensorFlow installed.
    """
    import tensorflow as tf
    import tf2onnx
    from tensorflow import keras

    print(f"[load] {input_path}")
    model = keras.models.load_model(input_path, compile=False)
    in_shape = tuple(model.input_shape[1:])
    if in_shape != (512, 512, 3):
        print(f"[warn] unexpected input shape {in_shape}, expected "
              "(512, 512, 3)")

    in_name = model.inputs[0].name.split(":")[0]
    spec = (tf.TensorSpec((None,) + model.input_shape[1:], tf.float32,
                          name=in_name),)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"[convert] opset {opset} -> {output_path}")
    tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=opset, output_path=output_path)
    _normalize_io_names(output_path)
    print(f"[ok] wrote {output_path}")
    return output_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert Keras HDF5 U-Net models to ONNX (dev-only; "
                    "needs TensorFlow + tf2onnx).")
    ap.add_argument("inputs", nargs="+", metavar="MODEL.hdf5",
                    help="one or more Keras .hdf5/.h5 model files")
    ap.add_argument("-o", "--output", default=None,
                    help="output .onnx path (only valid for a single "
                         "input model)")
    ap.add_argument("--out-dir", default=None,
                    help="directory for the converted .onnx files "
                         "(default: alongside each input model)")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version (default: 17)")
    args = ap.parse_args(argv)

    if args.output and len(args.inputs) > 1:
        ap.error("--output can only be used with a single input model")

    written = []
    for input_path in args.inputs:
        if not os.path.exists(input_path):
            ap.error(f"input model not found: {input_path}")
        out = _output_path(input_path, args.out_dir, args.output)
        written.append(convert(input_path, out, opset=args.opset))

    print(f"[done] converted {len(written)} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
