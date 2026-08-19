#!/usr/bin/env python
"""Verify that a resize implementation is v0.5-equivalent, on real data.

v0.5.0 resized every tile with `tf.image.resize(..., method='bicubic',
antialias=False)`: a fixed 4-tap Catmull-Rom kernel (a=-0.5), half-pixel
centers, no anti-aliasing, on float32 after /255. Anything that claims
v0.5 parity must reproduce that kernel. This script checks two candidates
against the verified TF-equivalent reference (an ONNX Resize node with
cubic_coeff_a=-0.5, half_pixel, exclude_outside=1 -- matches tf bicubic
to 4.2e-07, see utils/onnx_preprocess.py):

  rc12   the CUDA RawKernel from the 0.6.1-rc12 handoff, replicated
         bit-for-bit in NumPy (clamped indices, raw-tap weights). Verdict
         from 2026-08-10 on Barnekow: interior <= 0.014 DN, border band
         (2 px, clamp-vs-renormalize) <= 4.6 DN, model-level IoU >= 0.9999,
         total foreground within 0.004%.

  gdal   the in-read `Resampling.cubic` this repo uses on the fast path.
         NOT expected to pass: GDAL widens the kernel with the decimation
         factor (it anti-aliases) -- measured +20% stems / +26% volume vs
         v0.5 semantics at R13's 2.29x; see docs/resize-mechanics.md.

Phases:
  kernel   input-level max/mean |diff|, interior vs 2px border band
  model    both resizes through the same ONNX U-Net, binarized-core IoU
           and foreground counts (needs --model)

    python benchmark/bench_resize_parity.py --ortho <o.tif> --probe rc12
    python benchmark/bench_resize_parity.py --ortho <o.tif> --probe gdal \
        --model standalone/model_onnx/Spruce.onnx --phase model
"""
import argparse
import functools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.windows import Window  # noqa: E402

IMG, THR, CROP = 512, 0.5, 4


@functools.lru_cache(maxsize=8)
def catmull_rom_matrix(n_in, n_out):
    """The rc12 CUDA kernel as a 1-D matrix: Catmull-Rom a=-0.5, half-pixel
    centers, weights from the unclamped tap position, indices edge-clamped.
    Cached: every window in a run shares the same shapes."""
    a, scale = -0.5, n_in / n_out
    M = np.zeros((n_out, n_in), dtype=np.float64)
    for o in range(n_out):
        s = (o + 0.5) * scale - 0.5
        i0 = math.floor(s)
        for k in range(-1, 3):
            raw = i0 + k
            x = abs(s - raw)
            if x <= 1.0:
                w = (a + 2) * x**3 - (a + 3) * x**2 + 1
            elif x < 2.0:
                w = a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a
            else:
                w = 0.0
            M[o, min(max(raw, 0), n_in - 1)] += w
    return M


def rc12_resize(img_f32, out=IMG):
    h, w, _ = img_f32.shape
    t = np.tensordot(catmull_rom_matrix(h, out), img_f32, (1, 0))
    return np.tensordot(t, catmull_rom_matrix(w, out),
                        (1, 1)).transpose(0, 2, 1).astype(np.float32)


_REF_SESSIONS = {}


def v05_resize(img_f32, out=IMG):
    """Reference: the same Resize node utils/onnx_preprocess.py prepends."""
    import onnxruntime as ort
    from onnx import TensorProto, helper
    h, w, _ = img_f32.shape
    key = (h, w, out)
    if key not in _REF_SESSIONS:
        # Same attribute set as utils.onnx_preprocess.build_preprocessed_model
        # prepends in production -- import the coefficient so they cannot drift.
        from utils.onnx_preprocess import TF_CUBIC_COEFF_A
        node = helper.make_node(
            "Resize", ["x", "roi", "scales", "sizes"], ["y"], mode="cubic",
            cubic_coeff_a=TF_CUBIC_COEFF_A,
            coordinate_transformation_mode="half_pixel",
            exclude_outside=1, nearest_mode="floor")
        tvi = helper.make_tensor_value_info
        graph = helper.make_graph(
            [node], "resize",
            [tvi("x", TensorProto.FLOAT, [1, 3, h, w])],
            [tvi("y", TensorProto.FLOAT, [1, 3, out, out])],
            initializer=[
                helper.make_tensor("roi", TensorProto.FLOAT, [0], []),
                helper.make_tensor("scales", TensorProto.FLOAT, [0], []),
                helper.make_tensor("sizes", TensorProto.INT64, [4],
                                   [1, 3, out, out]),
            ])
        m = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 18)])
        m.ir_version = 8
        _REF_SESSIONS[key] = ort.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"])
    y = _REF_SESSIONS[key].run(
        None, {"x": img_f32.transpose(2, 0, 1)[None]})[0]
    return y[0].transpose(1, 2, 0)


def candidate(src, name, win):
    nat = src.read([1, 2, 3], window=win).transpose(1, 2, 0)
    f = np.divide(nat, np.float32(255.0), dtype=np.float32)
    if name == "rc12":
        return rc12_resize(f), f
    g = src.read([1, 2, 3], window=win, out_shape=(3, IMG, IMG),
                 resampling=Resampling.cubic).transpose(1, 2, 0)
    return np.divide(g, np.float32(255.0), dtype=np.float32), f


def content_windows(src, wpx, n):
    """Interior windows ranked by variance, spread across the raster."""
    dec = src.read(2, out_shape=(src.height // 16, src.width // 16))
    cands = []
    for r in range(wpx, src.height - 2 * wpx, wpx):
        for c in range(wpx, src.width - 2 * wpx, wpx):
            blk = dec[r // 16:(r + wpx) // 16, c // 16:(c + wpx) // 16]
            if blk.size and (blk > 0).mean() > 0.98:
                cands.append((float(blk.std()), c, r))
    cands.sort(reverse=True)
    step = max(1, len(cands) // (n * 3))
    return [(c, r) for _, c, r in cands[::step][:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", required=True)
    ap.add_argument("--probe", choices=["rc12", "gdal"], default="rc12")
    ap.add_argument("--phase", choices=["kernel", "model"], default="kernel")
    ap.add_argument("--model", help="ONNX U-Net (required for --phase model)")
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--factor", type=float, default=None,
                    help="force a decimation factor (window = 512*factor); "
                    "default derives the window from tile_size=15m")
    args = ap.parse_args()

    src = rasterio.open(args.ortho)
    if args.factor:
        wpx = int(IMG * args.factor)
    else:
        wpx = math.ceil(15.0 / src.res[0]) - 1
    print(f"{os.path.basename(args.ortho)}  GSD {src.res[0]*100:.3f} cm/px  "
          f"window {wpx}px -> {IMG}  factor {wpx/IMG:.2f}x  probe={args.probe}")
    wins = content_windows(src, wpx, args.windows)

    if args.phase == "kernel":
        stats = []
        for c, r in wins:
            cand, f = candidate(src, args.probe, Window(c, r, wpx, wpx))
            d = np.abs(cand - v05_resize(f)) * 255.0
            stats.append((d.max(), d[2:-2, 2:-2].max(), d.mean()))
        s = np.array(stats)
        print(f"vs v0.5 reference over {len(wins)} windows:")
        print(f"  whole    max|d| {s[:, 0].max():8.4f} DN   "
              f"mean {s[:, 2].mean():.6f} DN")
        print(f"  interior max|d| {s[:, 1].max():8.4f} DN   "
              "(border band excluded)")
        ok = s[:, 1].max() < 0.1
        verdict = ("v0.5-EQUIVALENT (interior < 0.1 DN)" if ok
                   else "NOT equivalent")
        print(f"  -> {verdict}")
        return 0 if ok else 1

    if not args.model:
        sys.exit("--phase model needs --model")
    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    nchw = sess.get_inputs()[0].shape[1] == 3

    def infer(t):
        x = t[None].transpose(0, 3, 1, 2) if nchw else t[None]
        y = np.squeeze(sess.run(None, {iname: x.astype(np.float32)})[0])
        if y.ndim == 3:
            y = y[0] if y.shape[0] == 1 else y[..., 0]
        return y[CROP:-CROP, CROP:-CROP]

    tot_a = tot_b = 0
    worst = 1.0
    for c, r in wins:
        cand, f = candidate(src, args.probe, Window(c, r, wpx, wpx))
        ba = infer(cand) >= THR
        bb = infer(v05_resize(f)) >= THR
        u = (ba | bb).sum()
        iou = float((ba & bb).sum() / u) if u else 1.0
        worst = min(worst, iou)
        tot_a += int(ba.sum())
        tot_b += int(bb.sum())
        print(f"  win({c:5d},{r:5d})  fg {ba.sum():6d} vs "
              f"{bb.sum():6d}  IoU {iou:.5f}")
    pct = 100 * (tot_a - tot_b) / max(tot_b, 1)
    print(f"total fg {tot_a} vs {tot_b} ({pct:+.3f}%)  "
          f"worst IoU {worst:.5f}")
    return 0 if worst > 0.999 else 1


if __name__ == "__main__":
    sys.exit(main())
