import numpy as np
import rasterio
from rasterio.transform import from_origin

from utils.PredictWorkers import _read_batch_jobs


def _make_raster(path):
    data = np.zeros((3, 300, 300), dtype=np.uint8)
    data[0] = 100
    data[1] = 150
    data[2] = 200
    data[:, :50, :50] = 0
    profile = dict(
        driver="GTiff", height=300, width=300, count=3, dtype="uint8",
        crs="EPSG:25833", transform=from_origin(0, 300, 1, 1))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _read(path, fill_invalid):
    job = {"src_col": 0, "src_row": 0, "src_width": 300, "src_height": 300}
    with rasterio.open(path) as src:
        tiles, masks, _ = _read_batch_jobs(
            src, [1, 2, 3], [job], fill_invalid=fill_invalid)
    return tiles[0], masks[0]


def test_multigpu_fills_black_corner_when_enabled(tmp_path):
    path = tmp_path / "r.tif"
    _make_raster(path)
    tile, mask = _read(path, fill_invalid=True)
    assert not mask[10, 10]
    assert tuple(tile[10, 10]) == (100, 150, 200)


def test_multigpu_leaves_black_corner_when_disabled(tmp_path):
    path = tmp_path / "r.tif"
    _make_raster(path)
    tile, mask = _read(path, fill_invalid=False)
    assert tuple(tile[10, 10]) == (0, 0, 0)
