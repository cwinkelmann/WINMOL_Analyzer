"""Before/after evidence for the nodata-aware edge fill on barnekow.

Drives the REAL orchestrator (winmol_run.py, process_type=Stems) twice on a
crop at barnekow's top diagonal boundary -- fill OFF then ON, toggled via
WINMOL_CONFIG_OVERRIDES_JSON -- and reports:
  * spurious foreground in a near-boundary band (should collapse with fill on),
  * interior foreground (should stay at parity).
Saves a side-by-side PNG. Skips if the model or ortho is missing.

winmol_run.py is the path the QGIS plugin and batch CLI use: it applies the
env override to the Config that flows into predict_stream_to_raster (the wired
streaming predictor). The standalone run_pipeline uses a DIFFERENT legacy
per-tile predictor (predict_with_resampling_per_tile) that this feature does
not touch, so it must not be used for this validation.

Usage:
  python benchmark/validate_edge_fill_barnekow.py \
      --ortho /path/20220212_Barnekow_4.tiff \
      --model standalone/model_onnx/General.onnx \
      --out   /tmp/barnekow_edge_fill.png
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.windows import Window


def _predict(crop_path, model_path, fill_on, workdir, root):
    """Run winmol_run.py (Stems) with the fill toggled; return the stem map."""
    stem = os.path.join(workdir, "stem_fill%d.tif" % int(fill_on))
    prefix = os.path.join(workdir, "out_fill%d" % int(fill_on)) + os.sep
    os.makedirs(workdir, exist_ok=True)
    env = dict(os.environ)
    env["WINMOL_CONFIG_OVERRIDES_JSON"] = json.dumps(
        {"fill_invalid_before_prediction": bool(fill_on)})
    env["WINMOL_ONNX_FORCE_CPU"] = "1"
    env["WINMOL_BATCH_AUTOTUNE"] = "off"
    env["PYTHONHASHSEED"] = "0"
    subprocess.run(
        [sys.executable, "-u", os.path.join(root, "winmol_run.py"),
         model_path, crop_path, stem, prefix, "Stems"],
        cwd=root, env=env, check=True)
    with rasterio.open(stem) as ds:
        return ds.read(1) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ortho", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/barnekow_edge_fill.png")
    ap.add_argument("--win", type=int, default=3000,
                    help="crop size at the top boundary")
    args = ap.parse_args()

    if not (os.path.exists(args.ortho) and os.path.exists(args.model)):
        print("SKIP: ortho or model missing")
        return 0

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Crop a window spanning the top diagonal boundary.
    crop = args.ortho + ".topcrop.tif"
    with rasterio.open(args.ortho) as s:
        w = min(args.win, s.width)
        h = min(args.win, s.height)
        window = Window((s.width - w) // 2, 0, w, h)
        data = s.read(window=window)
        prof = s.profile.copy()
        prof.update(width=w, height=h,
                    transform=s.window_transform(window))
    with rasterio.open(crop, "w", **prof) as d:
        d.write(data)
    alpha = data[3] if data.shape[0] >= 4 else np.full((h, w), 255, np.uint8)

    workdir = crop + ".work"
    off = _predict(crop, args.model, False, workdir, root)
    on = _predict(crop, args.model, True, workdir, root)

    # The stem map is written at the (coarser) model grid, not the crop's
    # native resolution, so resample the alpha-derived validity mask to the
    # stem map's shape before comparing (off and on share that shape).
    from skimage.transform import resize
    valid = resize((alpha > 0).astype(np.float32), off.shape,
                   order=0, preserve_range=True) > 0.5
    # Near-boundary band: valid pixels within 60 px of an invalid pixel.
    from scipy.ndimage import binary_dilation
    band = binary_dilation(~valid, iterations=60) & valid
    interior = valid & ~band

    def frac(mask_pred, region):
        n = int(region.sum())
        return 0.0 if n == 0 else float((mask_pred & region).sum()) / n

    print("near-boundary fg frac  off=%.5f on=%.5f" % (
        frac(off, band), frac(on, band)))
    print("interior      fg frac  off=%.5f on=%.5f" % (
        frac(off, interior), frac(on, interior)))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        ax[0].imshow(off, cmap="gray")
        ax[0].set_title("fill OFF (artifacts at boundary)")
        ax[1].imshow(on, cmap="gray")
        ax[1].set_title("fill ON")
        for a in ax:
            a.axis("off")
        fig.savefig(args.out, dpi=120, bbox_inches="tight")
        print("wrote", args.out)
    except Exception as exc:
        print("figure skipped:", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
