#!/usr/bin/env python3
"""Subprocess driver for model conformance: prove ONE model runs the pipeline
the same way as the reference.

Invoked by tests/test_model_conformance.py in a FRESH process per model
(mixed Keras-2/Keras-3/ONNX formats cannot share a process: the
TF_USE_LEGACY_KERAS routing is process-wide). The parent sets the right env;
this driver loads the model, probes the I/O contract, predicts the fixture
crop, runs the full vector chain, exports a GeoPackage, and writes a JSON
verdict with structural checks + metrics.

Usage: python tests/conformance_driver.py <model_path> <out_dir>
"""

import json
import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO_ROOT, TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")


def build_config():
    from classes.Config import Config
    with open(os.path.join(FIXTURES_DIR, "config_snapshot.json")) as f:
        snapshot = json.load(f)
    config = Config()
    for key, value in snapshot.items():
        setattr(config, key, value)
    return config


def run(model_path, out_dir):
    import numpy as np
    import rasterio

    from utils import IO
    from utils import Prediction as Pred
    from utils import Quantification as Quant
    from utils import Skeletonization as Skel
    from utils import Vectorization as Vec

    result = {"model": model_path, "ok": False, "checks": {}, "metrics": {}}
    checks, metrics = result["checks"], result["metrics"]
    config = build_config()

    # 1. load ------------------------------------------------------------
    t0 = time.time()
    model = IO.load_model_from_path(model_path)
    metrics["load_s"] = round(time.time() - t0, 2)
    metrics["loader"] = type(model).__name__
    checks["loads"] = True

    # 2. I/O contract probe ----------------------------------------------
    probe = np.random.RandomState(0).rand(1, 512, 512, 3).astype("float32")
    out = np.asarray(model.predict_on_batch(probe))
    checks["contract_shape"] = out.shape == (1, 512, 512, 1)
    checks["contract_sigmoid_range"] = bool(
        float(out.min()) >= 0.0 and float(out.max()) <= 1.0)

    # 3. predict the fixture crop ------------------------------------------
    crop_path = os.path.join(FIXTURES_DIR, "crop_input.tif")
    img, profile = IO.load_orthomosaic(crop_path, config)
    t0 = time.time()
    pred, out_profile = Pred.predict_with_resampling_per_tile(
        img, dict(profile), model, config)
    metrics["predict_s"] = round(time.time() - t0, 2)
    pred = pred.astype(np.uint8)

    # The prediction GRID must be model-independent: same shape and
    # georeferencing as the reference stem map, binary values only.
    with rasterio.open(os.path.join(FIXTURES_DIR, "stem_map.tif")) as ref:
        checks["grid_shape_matches_reference"] = \
            pred.shape == (ref.height, ref.width)
        checks["grid_transform_matches_reference"] = \
            out_profile["transform"] == ref.transform
    checks["prediction_is_binary"] = bool(np.isin(pred, (0, 1)).all())
    metrics["stem_px_pct"] = round(100.0 * float((pred > 0).mean()), 3)

    # persist the stem map so the parent can compute cross-model agreement
    pred_out = {
        "driver": "GTiff", "height": pred.shape[0], "width": pred.shape[1],
        "count": 1, "dtype": "uint8", "crs": out_profile.get("crs"),
        "transform": out_profile["transform"], "compress": "deflate",
    }
    with rasterio.open(os.path.join(out_dir, "pred.tif"), "w",
                       **pred_out) as dst:
        dst.write(pred, 1)

    # 4. full vector chain --------------------------------------------------
    t0 = time.time()
    segments = Skel.find_segments(pred, config, out_profile)
    metrics["n_parts"] = len(segments)
    segments = Vec.restore_geoinformation(segments, config, out_profile)
    stems = Vec.build_stem_parts(segments)
    stems = Vec.connect_stems(stems, config)
    Vec.rebuild_endnodes_from_stems(stems)
    stems = Quant.quantify_stems(stems, pred, out_profile, config)
    metrics["vector_s"] = round(time.time() - t0, 2)
    metrics["n_stems"] = len(stems)
    checks["vector_chain_completes"] = True

    lengths = [float(s.length) for s in stems]
    volumes = [float(s.volume) for s in stems]
    metrics["total_length_m"] = round(sum(lengths), 1)
    metrics["total_volume_m3"] = round(sum(volumes), 3)
    checks["measures_finite"] = bool(
        np.isfinite(lengths).all() and np.isfinite(volumes).all())
    checks["measures_nonnegative"] = bool(
        all(v >= 0 for v in lengths + volumes))

    # 5. export ------------------------------------------------------------
    if stems:
        gpkg = IO.write_all_layers_to_gpkg(
            stems, out_profile, os.path.join(out_dir, "conformance"))
        import pyogrio
        layers = sorted(r[0] for r in pyogrio.list_layers(gpkg))
        checks["gpkg_layers"] = layers == ["nodes", "stems", "vectors"]
        n = len(pyogrio.read_dataframe(gpkg, layer="stems"))
        checks["gpkg_stem_count_matches"] = n == len(stems)
    else:
        # a model may legitimately detect nothing (undertrained) — the
        # pipeline still conformed if we got here without crashing
        result["zero_detections"] = True

    result["ok"] = all(v is True for v in checks.values())
    return result


def main():
    model_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "result.json")
    try:
        result = run(model_path, out_dir)
    except Exception as exc:  # any crash = non-conformance, reported cleanly
        result = {
            "model": model_path, "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-3000:],
        }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "traceback"}, indent=1, sort_keys=True))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
