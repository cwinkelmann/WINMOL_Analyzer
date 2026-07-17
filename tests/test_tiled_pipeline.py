"""Golden tests for the tiled vector path: tile grid math, per-tile
vectorization, and merge_and_filter_tiled_results.

The merge comparison pins current behavior INCLUDING its known issues
(outer-perimeter filtering M-1; seam handling M-2) — a fix there must
regenerate golden_merged.gpkg and will show an explicit stem-count diff.
"""

import os
import shutil
import tempfile

import pytest
import rasterio

import helpers
from utils import IO
from utils.Tiling import build_tile_grid, meters_to_pixels

pytestmark = pytest.mark.slow


def test_tile_grid_matches_golden(stem_map, pipeline_config, golden,
                                  fixtures_dir):
    pred, profile = stem_map
    t = profile["transform"]
    halo_px = meters_to_pixels(pipeline_config.tile_overlap_m, t.a, t.e)
    jobs = build_tile_grid(profile["width"], profile["height"], 512, halo_px)
    records = [{
        "tile_id": j.tile_id,
        "inner": [j.x0, j.y0, j.x1, j.y1],
        "halo": [j.hx0, j.hy0, j.hx1, j.hy1],
    } for j in jobs]
    golden_grid = helpers.read_json_gz(
        os.path.join(fixtures_dir, "tile_grid.json"))
    assert records == golden_grid


def test_tiled_vectorization_and_merge_matches_golden(
        stem_map, pipeline_config, fixtures_dir):
    from utils.VectorTilePipeline import process_prediction_tiles

    stem_map_path = os.path.join(fixtures_dir, "stem_map.tif")
    with rasterio.open(stem_map_path) as src:
        width, height = src.width, src.height
        t = src.transform

    halo_px = meters_to_pixels(pipeline_config.tile_overlap_m, t.a, t.e)
    jobs = build_tile_grid(width, height, 512, halo_px)

    work_dir = tempfile.mkdtemp(prefix="winmol_test_tiles_")
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
        assert tile_paths, "no foreground tiles — fixture corrupt?"

        process_prediction_tiles(
            tile_paths, pipeline_config, "Trees", work_dir, 1)
        merged = os.path.join(work_dir, "merged.gpkg")
        IO.merge_and_filter_tiled_results(
            work_dir=work_dir, output_gpkg=merged,
            edge_buffer_m=pipeline_config.tile_overlap_m,
            config=pipeline_config, stem_map_path=stem_map_path)

        helpers.assert_gpkg_stems_match(
            merged, os.path.join(fixtures_dir, "golden_merged.gpkg"))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_merged_golden_differs_from_untiled_golden(fixtures_dir):
    """Documents that the tiled+merged result is NOT identical to the
    untiled chain on the same stem map (edge filtering + seam handling).
    If this ever starts matching exactly, the merge semantics changed."""
    merged = helpers.gpkg_stems_canonical(
        os.path.join(fixtures_dir, "golden_merged.gpkg"))
    untiled = helpers.gpkg_stems_canonical(
        os.path.join(fixtures_dir, "golden_stems.gpkg"))
    assert len(merged) > 0
    assert merged != untiled
