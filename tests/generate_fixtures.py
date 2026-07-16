#!/usr/bin/env python3
"""Generate the golden-master fixtures for the WINMOL pipeline test suite.

This is the ONLY producer of tests/fixtures/. It executes the CURRENT
pipeline on a small crop and records every stage's real output:

    crop_input.tif -> stem_map.tif -> stage_*.json.gz -> golden_*.gpkg

Usage:
    PYTHONHASHSEED=0 python tests/generate_fixtures.py \
        [--source <tif>] [--crop-size 1024] [--window ROW,COL] [--model <hdf5>]

Determinism: the script re-execs itself with PYTHONHASHSEED=0 (pipeline
output depends on set-iteration order — docs/CODE_REVIEW_2.md A-4), forces
serial workers, and runs the whole vector chain TWICE, refusing to write
fixtures unless both passes are canonically identical.

The crop window is auto-selected for overlapping trees: candidate 1024-px
windows are ranked by predicted stem density, and the top candidates are
quick-vectorized to count pairwise crossing skeleton paths; the window with
the most crossings wins.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

# --- determinism guards (before ANY heavy import) --------------------------
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)
# ONNX inference must be deterministic and match Linux CI: force CPU
# onnxruntime (CoreML/CUDA may differ). TF_USE_LEGACY_KERAS is only consulted
# if a .hdf5 model is passed via --model.
os.environ.setdefault("WINMOL_ONNX_FORCE_CPU", "1")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO_ROOT, TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                          # noqa: E402
import rasterio                             # noqa: E402
from rasterio.windows import Window         # noqa: E402

import helpers                              # noqa: E402
from classes.Config import Config           # noqa: E402

FIXTURES_DIR = helpers.FIXTURES_DIR
DEFAULT_SOURCE = os.path.join(
    REPO_ROOT, "standalone", "pred", "notebook_crop_input.tif")
DEFAULT_MODEL = os.path.join(
    REPO_ROOT, "standalone", "model_onnx", "General.onnx")

# Determinism overrides baked into the config snapshot. Serial workers pin
# result ORDER (imap_unordered / callback appends) and CONTENT (the refine
# serial path mutates a shared skeleton view — CODE_REVIEW V-2 — so worker
# count changes results; 1 worker + fixed hash seed is fully reproducible).
DETERMINISM_OVERRIDES = {
    "cpu_workers": 1,
    "vector_tile_workers": 1,
    "prediction_batch_autotune": False,
    "prediction_batch_size": 1,
    "prediction_batch_cpu": 1,
}


def build_config(snapshot=None):
    config = Config()
    for key, value in (snapshot or DETERMINISM_OVERRIDES).items():
        setattr(config, key, value)
    return config


def config_to_snapshot(config) -> dict:
    snap = {}
    for key in dir(config):
        if key.startswith("_"):
            continue
        value = getattr(config, key)
        if callable(value):
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        snap[key] = value
    return snap


def sha16(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Window selection: density ranking + crossing count on the top candidates
# ---------------------------------------------------------------------------

def select_window(source_path, model, config, crop_size, stride):
    from utils import IO
    from utils import Prediction as Pred
    from utils import Skeletonization as Skel

    img, profile = IO.load_orthomosaic(source_path, config)
    src_transform = profile["transform"]
    print(f"[select] predicting full source {img.shape[:2]} ...")
    pred_full, out_profile = Pred.predict_with_resampling_per_tile(
        img, dict(profile), model, config)
    out_transform = out_profile["transform"]

    src_h, src_w = img.shape[0], img.shape[1]
    if src_h < crop_size or src_w < crop_size:
        raise SystemExit(
            f"source {src_w}x{src_h} smaller than crop {crop_size}")

    def to_pred_rc(src_row, src_col):
        x, y = src_transform * (src_col, src_row)
        pc, pr = ~out_transform * (x, y)
        return int(round(pr)), int(round(pc))

    candidates = []
    for row in range(0, src_h - crop_size + 1, stride):
        for col in range(0, src_w - crop_size + 1, stride):
            r0, c0 = to_pred_rc(row, col)
            r1, c1 = to_pred_rc(row + crop_size, col + crop_size)
            sub = pred_full[max(r0, 0):r1, max(c0, 0):c1]
            density = float(sub.mean()) if sub.size else 0.0
            candidates.append((density, row, col))
    candidates.sort(reverse=True)
    top = candidates[:3]
    print("[select] top-3 by density:",
          [(round(d, 4), r, c) for d, r, c in top])

    from utils import Vectorization as Vec

    best = None
    for density, row, col in top:
        r0, c0 = to_pred_rc(row, col)
        r1, c1 = to_pred_rc(row + crop_size, col + crop_size)
        sub = np.ascontiguousarray(pred_full[max(r0, 0):r1, max(c0, 0):c1])
        sub_profile = {"transform": out_transform,
                       "width": sub.shape[1], "height": sub.shape[0]}
        # Crossings must be counted on CONNECTED stems: part paths are cut at
        # skeleton branchpoints, so raw parts never literally cross.
        parts = Skel.find_segments(sub, config, sub_profile)
        parts = Vec.restore_geoinformation(parts, config, sub_profile)
        stems = Vec.connect_stems(Vec.build_stem_parts(parts), config)
        lines = [s.path for s in stems if len(s.path.coords) >= 2]
        crossings = 0
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if lines[i].crosses(lines[j]):
                    crossings += 1
        print(f"[select] window ({row},{col}): density {density:.4f}, "
              f"{len(lines)} stems, {crossings} crossings")
        if best is None or crossings > best[0]:
            best = (crossings, density, row, col)

    crossings, density, row, col = best
    print(f"[select] chosen window row={row} col={col} "
          f"({crossings} crossings, density {density:.4f})")
    return row, col, crossings


def write_crop(source_path, row, col, crop_size, crop_path):
    with rasterio.open(source_path) as src:
        window = Window(col, row, crop_size, crop_size)
        bands = min(3, src.count)
        data = src.read(list(range(1, bands + 1)), window=window)
        profile = src.profile.copy()
        profile.update(
            width=crop_size, height=crop_size, count=bands,
            transform=src.window_transform(window), driver="GTiff",
            compress="deflate")
        profile.pop("nodata", None)
    if os.path.exists(crop_path):
        with rasterio.open(crop_path) as old:
            if (old.width, old.height) == (crop_size, crop_size) \
                    and np.array_equal(old.read(), data):
                print(f"[crop] unchanged: {crop_path}")
                return
    with rasterio.open(crop_path, "w", **profile) as dst:
        dst.write(data)
    print(f"[crop] wrote {crop_path}")


def predict_crop(crop_path, model, config, stem_map_path):
    from utils import IO
    from utils import Prediction as Pred

    img, profile = IO.load_orthomosaic(crop_path, config)
    pred, out_profile = Pred.predict_with_resampling_per_tile(
        img, dict(profile), model, config)
    pred = pred.astype(np.uint8)

    # Re-run stability: TF inference may not be bit-deterministic across
    # devices/runs. Keep the existing golden stem map when the fresh
    # prediction agrees within the inference-test tolerance, so regeneration
    # does not churn every downstream fixture.
    if os.path.exists(stem_map_path):
        with rasterio.open(stem_map_path) as old:
            if (old.height, old.width) == pred.shape:
                agreement = float(np.mean(old.read(1) == pred))
                if agreement >= 0.999:
                    print(f"[predict] unchanged within tolerance "
                          f"(agreement {agreement:.5f}): {stem_map_path}")
                    return
                print(f"[predict] REPLACING stem map "
                      f"(agreement {agreement:.5f} < 0.999)")
    out = {
        "driver": "GTiff", "height": pred.shape[0], "width": pred.shape[1],
        "count": 1, "dtype": "uint8", "crs": out_profile.get("crs"),
        "transform": out_profile["transform"], "compress": "deflate",
    }
    with rasterio.open(stem_map_path, "w", **out) as dst:
        dst.write(pred, 1)
    print(f"[predict] stem map {pred.shape} foreground "
          f"{int((pred > 0).sum())} px -> {stem_map_path}")


# ---------------------------------------------------------------------------
# The vector chain (mirrors standalone run_pipeline; each stage canonicalized
# BEFORE the next stage mutates the shared objects)
# ---------------------------------------------------------------------------

def run_vector_chain(stem_map_path, snapshot):
    from utils import IO  # noqa: F401  (env parity with the real pipeline)
    from utils import Quantification as Quant
    from utils import Skeletonization as Skel
    from utils import Vectorization as Vec

    config = build_config(snapshot)
    with rasterio.open(stem_map_path) as src:
        pred = src.read(1)
        profile = dict(src.profile)

    # Fixtures are stored in PIPELINE order (sort=False): downstream stages
    # are order-sensitive (set() rebuilds, greedy connect — review A-4), so
    # replaying a re-sorted fixture would diverge from the recorded run.
    # Comparisons in helpers sort both sides.
    stages = {}
    segments = Skel.find_segments(pred, config, profile)
    stages["stage_find_segments"] = \
        helpers.parts_to_canonical(segments, sort=False)

    segments = Vec.restore_geoinformation(segments, config, profile)
    stages["stage_restore_geo"] = \
        helpers.parts_to_canonical(segments, sort=False)

    stems = Vec.build_stem_parts(segments)
    stages["stage_build_stem_parts"] = \
        helpers.stems_to_canonical(stems, sort=False)

    stems = Vec.connect_stems(stems, config)
    Vec.rebuild_endnodes_from_stems(stems)  # no-op today (review V-8)
    stages["stage_connect_stems"] = \
        helpers.stems_to_canonical(stems, sort=False)

    stems = Quant.quantify_stems(stems, pred, profile, config)
    stages["stage_quantified_contour"] = \
        helpers.stems_to_canonical(stems, sort=False)

    # EDT variant from the connect fixture (fresh objects, fresh config)
    config_edt = build_config(snapshot)
    config_edt.diameter_method = "edt"
    stems_edt = helpers.stems_from_canonical(stages["stage_connect_stems"])
    stems_edt = Quant.quantify_stems(stems_edt, pred, profile, config_edt)
    stages["stage_quantified_edt"] = \
        helpers.stems_to_canonical(stems_edt, sort=False)

    return stages, stems, profile


def write_golden_gpkg(stems, profile, out_path):
    """write_all_layers_to_gpkg into a temp prefix, then move into fixtures
    only when content changed (gpkg bytes embed timestamps)."""
    from utils import IO
    tmp_dir = tempfile.mkdtemp(prefix="winmol_fixture_gpkg_")
    try:
        tmp_gpkg = IO.write_all_layers_to_gpkg(
            stems, profile, os.path.join(tmp_dir, "golden"))
        if os.path.exists(out_path):
            try:
                same = (helpers.gpkg_stems_canonical(tmp_gpkg)
                        == helpers.gpkg_stems_canonical(out_path))
            except Exception:
                same = False
            if same:
                print(f"[gpkg] unchanged: {out_path}")
                return
        shutil.copyfile(tmp_gpkg, out_path)
        print(f"[gpkg] wrote {out_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_tiled_variant(stem_map_path, snapshot, merged_out):
    from utils import IO
    from utils.Tiling import build_tile_grid, meters_to_pixels
    from utils.VectorTilePipeline import process_prediction_tiles

    config = build_config(snapshot)
    with rasterio.open(stem_map_path) as src:
        width, height = src.width, src.height
        px_x, px_y = src.transform.a, src.transform.e

    halo_px = meters_to_pixels(config.tile_overlap_m, px_x, px_y)
    jobs = build_tile_grid(width, height, 512, halo_px)
    grid_records = [{
        "tile_id": j.tile_id,
        "inner": [j.x0, j.y0, j.x1, j.y1],
        "halo": [j.hx0, j.hy0, j.hx1, j.hy1],
    } for j in jobs]

    work_dir = tempfile.mkdtemp(prefix="winmol_fixture_tiles_")
    try:
        tile_paths = []
        for job in jobs:
            tile, tile_profile = IO.load_raster_window_with_profile(
                stem_map_path, job.halo_window)
            if tile is None or tile.size == 0 or not (tile >= 1).any():
                continue
            tile_path = os.path.join(
                work_dir, f"{job.tile_id}_roi_stem_map.tif")
            IO.write_tile_raster(tile, tile_profile, tile_path)
            tile_paths.append(tile_path)
        print(f"[tiled] {len(tile_paths)}/{len(jobs)} tiles with foreground "
              f"(halo {halo_px} px)")

        process_prediction_tiles(tile_paths, config, "Trees", work_dir, 1)
        tmp_merged = os.path.join(work_dir, "merged.gpkg")
        IO.merge_and_filter_tiled_results(
            work_dir=work_dir, output_gpkg=tmp_merged,
            edge_buffer_m=config.tile_overlap_m, config=config)

        if os.path.exists(merged_out):
            try:
                same = (helpers.gpkg_stems_canonical(tmp_merged)
                        == helpers.gpkg_stems_canonical(merged_out))
            except Exception:
                same = False
            if same:
                print(f"[tiled] unchanged: {merged_out}")
                return grid_records
        shutil.copyfile(tmp_merged, merged_out)
        print(f"[tiled] wrote {merged_out}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return grid_records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--crop-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--window", default=None,
                    help="ROW,COL to skip the window search")
    args = ap.parse_args()

    if not os.path.exists(args.source):
        raise SystemExit(f"source not found: {args.source}")
    if not os.path.exists(args.model):
        raise SystemExit(f"model not found: {args.model}")
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    config = build_config()
    snapshot = config_to_snapshot(config)

    from utils import IO
    print(f"[model] loading {os.path.basename(args.model)}")
    model = IO.load_model_from_path(args.model)

    if args.window:
        row, col = (int(v) for v in args.window.split(","))
        crossings = -1
        print(f"[select] using explicit window row={row} col={col}")
    else:
        row, col, crossings = select_window(
            args.source, model, config, args.crop_size, args.stride)

    crop_path = os.path.join(FIXTURES_DIR, "crop_input.tif")
    stem_map_path = os.path.join(FIXTURES_DIR, "stem_map.tif")
    write_crop(args.source, row, col, args.crop_size, crop_path)
    predict_crop(crop_path, model, config, stem_map_path)

    print("[chain] pass A ...")
    stages_a, stems_a, profile_a = run_vector_chain(stem_map_path, snapshot)
    print("[chain] pass B (determinism check) ...")
    stages_b, _, _ = run_vector_chain(stem_map_path, snapshot)

    for name in stages_a:
        if (helpers.dumps_canonical(stages_a[name])
                != helpers.dumps_canonical(stages_b[name])):
            na, nb = len(stages_a[name]), len(stages_b[name])
            raise SystemExit(
                f"DETERMINISM FAILURE at {name}: pass A ({na} records) != "
                f"pass B ({nb} records). Fixtures NOT written. "
                f"Check PYTHONHASHSEED and worker overrides.")
    print("[chain] determinism check passed "
          f"({ {k: len(v) for k, v in stages_a.items()} })")

    for name, records in stages_a.items():
        path = os.path.join(FIXTURES_DIR, name + ".json.gz")
        changed = helpers.write_if_changed(path, records)
        print(f"[stage] {'wrote' if changed else 'unchanged'}: {path} "
              f"({len(records)} records)")

    write_golden_gpkg(
        stems_a, profile_a, os.path.join(FIXTURES_DIR, "golden_stems.gpkg"))

    grid_records = run_tiled_variant(
        stem_map_path, snapshot,
        os.path.join(FIXTURES_DIR, "golden_merged.gpkg"))
    helpers.write_if_changed(
        os.path.join(FIXTURES_DIR, "tile_grid.json"), grid_records)

    snap_path = os.path.join(FIXTURES_DIR, "config_snapshot.json")
    with open(snap_path, "w") as f:
        json.dump(snapshot, f, indent=1, sort_keys=True)

    import scipy
    import shapely
    import skimage
    manifest = {
        "pythonhashseed": "0",
        "source": os.path.relpath(args.source, REPO_ROOT),
        "window": {"row": row, "col": col, "size": args.crop_size},
        "crossings_in_window": crossings,
        "model": {"file": os.path.basename(args.model),
                  "sha256_16": sha16(args.model)},
        "counts": {k: len(v) for k, v in stages_a.items()},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "shapely": shapely.__version__,
            "rasterio": rasterio.__version__,
            "scikit-image": skimage.__version__,
            "scipy": scipy.__version__,
        },
    }
    with open(os.path.join(FIXTURES_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("[done] fixtures in", FIXTURES_DIR)


if __name__ == "__main__":
    main()
