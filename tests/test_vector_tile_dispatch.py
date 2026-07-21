"""Worker-side tile reads must be results-identical to the legacy
write-GeoTIFF-then-re-read handoff, in serial and in parallel dispatch.

The legacy round trip was not a no-op: build_safe_prediction_profile
defaults to dtype='float32', so workers always received float32 pixels
and the rewritten tile profile, never the uint8 source window. These
tests pin that _tile_handoff mirrors the round trip exactly, that spec
dispatch (worker reads its own window) equals path dispatch, and that
the ProcessPoolExecutor tile pool equals serial execution.
"""
import copy
import os
import shutil
import tempfile

import numpy as np
import pytest
import rasterio

import helpers
from utils import IO
from utils.Tiling import build_tile_grid, meters_to_pixels
from utils.VectorTilePipeline import (
    _tile_handoff,
    make_tile_spec,
    process_prediction_tiles,
)

pytestmark = pytest.mark.slow

TILE_INNER_PX = 760  # 2x2 grid over the 1520 px fixture


def _grid(stem_map_path, tile_overlap_m):
    with rasterio.open(stem_map_path) as src:
        width, height = src.width, src.height
        t = src.transform
    halo_px = meters_to_pixels(tile_overlap_m, t.a, t.e)
    return build_tile_grid(width, height, TILE_INNER_PX, halo_px)


def _canonical_tiles(work_dir):
    """{tile name: canonical stems} for every tile gpkg in work_dir."""
    out = {}
    for name in sorted(os.listdir(work_dir)):
        if name.endswith('.gpkg'):
            out[name] = helpers.gpkg_stems_canonical(
                os.path.join(work_dir, name))
    return out


def _run_spec_mode(stem_map_path, config, work_dir, cpu_workers):
    jobs = _grid(stem_map_path, config.tile_overlap_m)
    specs = [
        make_tile_spec(stem_map_path, job.tile_id, job.halo_window)
        for job in jobs
    ]
    return process_prediction_tiles(
        specs, config, 'Trees', work_dir, cpu_workers)


@pytest.fixture(scope='module')
def spec_serial(fixtures_dir, pipeline_config):
    """Serial spec-mode run: the reference for the other dispatch modes."""
    stem_map_path = os.path.join(fixtures_dir, 'stem_map.tif')
    work_dir = tempfile.mkdtemp(prefix='winmol_spec_serial_')
    config = copy.copy(pipeline_config)
    results = _run_spec_mode(stem_map_path, config, work_dir, 1)
    yield work_dir, results
    shutil.rmtree(work_dir, ignore_errors=True)


def test_handoff_matches_write_read_roundtrip(fixtures_dir):
    stem_map_path = os.path.join(fixtures_dir, 'stem_map.tif')
    job = _grid(stem_map_path, 12.0)[0]
    pred, tile_profile = IO.load_raster_window_with_profile(
        stem_map_path, job.halo_window)

    work_dir = tempfile.mkdtemp(prefix='winmol_handoff_')
    try:
        tile_path = os.path.join(work_dir, 'tile_roi_stem_map.tif')
        IO.write_tile_raster(pred, tile_profile, tile_path)
        legacy_pred, legacy_profile = IO.load_stem_map(tile_path)

        handoff_pred, handoff_profile = _tile_handoff(pred, tile_profile)

        assert handoff_pred.dtype == legacy_pred.dtype == np.float32
        assert np.array_equal(handoff_pred, legacy_pred)
        for key in ('width', 'height', 'count', 'dtype', 'nodata'):
            assert handoff_profile[key] == legacy_profile[key], key
        assert handoff_profile['transform'] == legacy_profile['transform']
        assert handoff_profile['crs'] == legacy_profile['crs']
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_spec_dispatch_matches_path_dispatch(
        fixtures_dir, pipeline_config, spec_serial):
    stem_map_path = os.path.join(fixtures_dir, 'stem_map.tif')
    spec_dir, spec_results = spec_serial

    path_dir = tempfile.mkdtemp(prefix='winmol_path_serial_')
    try:
        tile_paths = []
        for job in _grid(stem_map_path, pipeline_config.tile_overlap_m):
            tile, tile_profile = IO.load_raster_window_with_profile(
                stem_map_path, job.halo_window)
            if tile is None or tile.size == 0 or not (tile >= 1).any():
                continue
            tile_path = os.path.join(
                path_dir, f'{job.tile_id}_roi_stem_map.tif')
            IO.write_tile_raster(tile, tile_profile, tile_path)
            tile_paths.append(tile_path)
        assert tile_paths, 'no foreground tiles — fixture corrupt?'

        path_results = process_prediction_tiles(
            tile_paths, copy.copy(pipeline_config), 'Trees', path_dir, 1)

        # Foreground tiles agree (spec mode also queues empty tiles and
        # reports them as None).
        assert (len([r for r in spec_results if r is not None])
                == len([r for r in path_results if r is not None]))
        assert _canonical_tiles(spec_dir) == _canonical_tiles(path_dir)
        # Spec mode must keep writing the tile rasters the merge stage
        # and the failure-inspection contract rely on.
        spec_rasters = sorted(
            f for f in os.listdir(spec_dir) if f.endswith('.tif'))
        path_rasters = sorted(
            f for f in os.listdir(path_dir) if f.endswith('.tif'))
        assert spec_rasters == path_rasters
    finally:
        shutil.rmtree(path_dir, ignore_errors=True)


def test_parallel_dispatch_matches_serial(
        fixtures_dir, pipeline_config, spec_serial):
    stem_map_path = os.path.join(fixtures_dir, 'stem_map.tif')
    spec_dir, _ = spec_serial

    par_dir = tempfile.mkdtemp(prefix='winmol_spec_par_')
    try:
        config = copy.copy(pipeline_config)
        config.vector_tile_workers = 2
        _run_spec_mode(stem_map_path, config, par_dir, 2)
        assert _canonical_tiles(par_dir) == _canonical_tiles(spec_dir)
    finally:
        shutil.rmtree(par_dir, ignore_errors=True)


def test_empty_tile_spec_returns_none_and_writes_nothing(pipeline_config):
    work_dir = tempfile.mkdtemp(prefix='winmol_spec_empty_')
    try:
        empty_path = os.path.join(work_dir, 'empty.tif')
        transform = rasterio.transform.from_origin(0.0, 32.0, 1.0, 1.0)
        with rasterio.open(
            empty_path, 'w', driver='GTiff', dtype='uint8', count=1,
            width=32, height=32, transform=transform,
        ) as dst:
            dst.write(np.zeros((32, 32), dtype=np.uint8), 1)

        out_dir = os.path.join(work_dir, 'out')
        spec = make_tile_spec(
            empty_path, 'raster_r00000_c00000',
            rasterio.windows.Window(0, 0, 32, 32))
        results = process_prediction_tiles(
            [spec], copy.copy(pipeline_config), 'Trees', out_dir, 1)

        assert results == [None]
        assert os.listdir(out_dir) == []
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
