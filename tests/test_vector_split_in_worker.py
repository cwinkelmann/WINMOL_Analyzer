"""The tile split done by the workers must produce what the serial split
in the parent produced: the same tile raster for every window with
foreground, no raster for an empty window, and the same vector result
as running the pipeline on a pre-split tile.
"""
import glob
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils import IO  # noqa: E402
from utils import VectorTilePipeline as VTP  # noqa: E402
from utils.Tiling import build_tile_grid  # noqa: E402


@pytest.fixture
def stem_map(tmp_path):
    """A 300x300 binary stem map with one diagonal stem in the top-left
    quadrant; the bottom-right quadrant is empty."""
    arr = np.zeros((300, 300), dtype=np.uint8)
    for i in range(20, 120):
        arr[i:i + 3, i + 10:i + 13] = 1
    path = str(tmp_path / 'stem_map.tif')
    with rasterio.open(
        path, 'w', driver='GTiff', width=300, height=300, count=1,
        dtype='uint8', crs='EPSG:32633',
        transform=from_origin(500000, 5800000, 0.05, 0.05),
    ) as dst:
        dst.write(arr, 1)
    return path


def _config():
    from classes.Config import Config
    cfg = Config()
    cfg.cpu_workers = 1
    cfg.vector_tile_workers = 1
    cfg.progress_interval_s = 3600
    return cfg


def test_worker_split_writes_only_foreground_tiles(stem_map, tmp_path):
    jobs = build_tile_grid(300, 300, 150, 10)
    work = str(tmp_path / 'work')
    paths = [os.path.join(work, f'{j.tile_id}_roi_stem_map.tif')
             for j in jobs]
    sources = [(stem_map, j.halo_window) for j in jobs]
    results = VTP.process_prediction_tiles(
        paths, _config(), 'Nodes', work, 1, sources=sources)
    written = sorted(os.path.basename(p)
                     for p in glob.glob(os.path.join(work, '*_roi_stem_map.tif')))
    # exactly the windows with foreground got a raster
    expected = []
    with rasterio.open(stem_map) as src:
        for j in jobs:
            win = src.read(1, window=j.halo_window, boundless=True,
                           fill_value=0)
            if (win >= 1).any():
                expected.append(f'{j.tile_id}_roi_stem_map.tif')
    assert written == sorted(expected)
    assert 0 < len(written) < len(jobs)
    # empty windows are reported as empty tiles, not failures
    assert sum(r is None for r in results) == len(jobs) - len(written)


def test_worker_split_matches_parent_split(stem_map, tmp_path):
    jobs = build_tile_grid(300, 300, 150, 10)

    # the way it was: parent reads, filters and writes; workers read tiles
    serial = str(tmp_path / 'serial')
    serial_paths = []
    for j in jobs:
        tile, prof = IO.load_raster_window_with_profile(
            stem_map, j.halo_window)
        if not (tile >= 1).any():
            continue
        p = os.path.join(serial, f'{j.tile_id}_roi_stem_map.tif')
        IO.write_tile_raster(tile, prof, p)
        serial_paths.append(p)
    VTP.process_prediction_tiles(serial_paths, _config(), 'Nodes', serial, 1)

    # the way it is: workers do the split
    worker = str(tmp_path / 'worker')
    paths = [os.path.join(worker, f'{j.tile_id}_roi_stem_map.tif')
             for j in jobs]
    VTP.process_prediction_tiles(
        paths, _config(), 'Nodes', worker, 1,
        sources=[(stem_map, j.halo_window) for j in jobs])

    for p in serial_paths:
        q = os.path.join(worker, os.path.basename(p))
        with rasterio.open(p) as a, rasterio.open(q) as b:
            assert a.profile == b.profile
            assert np.array_equal(a.read(1), b.read(1))
    a_gpkgs = sorted(os.path.basename(p)
                     for p in glob.glob(os.path.join(serial, '*.gpkg')))
    b_gpkgs = sorted(os.path.basename(p)
                     for p in glob.glob(os.path.join(worker, '*.gpkg')))
    assert a_gpkgs == b_gpkgs
    assert a_gpkgs, 'the stem should have produced at least one tile gpkg'
