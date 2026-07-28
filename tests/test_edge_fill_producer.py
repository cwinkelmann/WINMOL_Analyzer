import queue

import numpy as np
import rasterio
from rasterio.transform import from_origin

from utils.Prediction import TileBatchProducer


def _make_raster(path):
    # Uniform colour with a black (invalid) top-left corner.
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


def _run_producer(path, fill_invalid):
    q = queue.Queue()
    job = {"src_col": 0, "src_row": 0, "src_width": 300, "src_height": 300}
    p = TileBatchProducer(
        uav_path=str(path), chunk_size=1, jobs=[job], n_channels=3,
        out_queue=q, out_size=None, fill_invalid=fill_invalid)
    p.start()
    p.join()
    assert p.error is None, p.error
    msg = q.get_nowait()
    _, tile, valid_mask = msg["items"][0]
    return tile, valid_mask


def test_producer_fills_black_corner_when_enabled(tmp_path):
    path = tmp_path / "r.tif"
    _make_raster(path)
    tile, valid_mask = _run_producer(path, fill_invalid=True)
    assert not valid_mask[10, 10]                    # corner is invalid
    assert tuple(tile[10, 10]) == (100, 150, 200)    # filled from nearest


def test_producer_leaves_black_corner_when_disabled(tmp_path):
    path = tmp_path / "r.tif"
    _make_raster(path)
    tile, valid_mask = _run_producer(path, fill_invalid=False)
    assert tuple(tile[10, 10]) == (0, 0, 0)          # untouched cliff
