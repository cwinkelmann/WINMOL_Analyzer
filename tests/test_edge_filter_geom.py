"""Pure-geometry table for ``_raster_filter_geom``'s per-side buffering.

Shrinking a tile footprint inward by edge_buffer_m dedups stems shared with
an overlapping neighbour across an interior seam. But a tile side that lies
on the ortho's TRUE OUTER boundary has no neighbour, so shrinking it drops
real stems (worst at corners, where two boundary sides meet). When
ortho_bounds is known, only interior-seam sides should shrink; sides on the
ortho boundary must keep their true extent. With ortho_bounds=None the
legacy all-sides shrink must be unchanged.

No fixtures: each raster is a tiny on-the-fly GeoTIFF with exact,
pixel-aligned bounds so the resulting polygon's ``.bounds`` can be asserted
against exact numbers.
"""
import numpy as np
import pytest

from utils.IO import _raster_filter_geom

EB = 2.0


def _write_tile_raster(path, bounds, pixel_size=1.0):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    left, bottom, right, top = bounds
    width = round((right - left) / pixel_size)
    height = round((top - bottom) / pixel_size)
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1,
        "dtype": "uint8", "crs": rasterio.crs.CRS.from_epsg(32633),
        "transform": from_origin(left, top, pixel_size, pixel_size),
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(np.zeros((1, height, width), dtype="uint8"))
    return str(path)


def _ortho_bounds(left, bottom, right, top):
    rasterio = pytest.importorskip("rasterio")
    return rasterio.coords.BoundingBox(left, bottom, right, top)


def test_interior_tile_shrinks_all_four_sides(tmp_path):
    ortho = _ortho_bounds(0, 0, 1000, 1000)
    path = _write_tile_raster(tmp_path / "interior.tif", (100, 100, 200, 200))
    geom, _ = _raster_filter_geom(path, EB, ortho_bounds=ortho)
    assert geom.bounds == (102.0, 102.0, 198.0, 198.0)


def test_corner_tile_shrinks_only_its_two_interior_sides(tmp_path):
    # Bottom-left tile: left & bottom sit on the ortho boundary, right & top
    # are interior seams.
    ortho = _ortho_bounds(0, 0, 1000, 1000)
    path = _write_tile_raster(tmp_path / "corner.tif", (0, 0, 100, 100))
    geom, _ = _raster_filter_geom(path, EB, ortho_bounds=ortho)
    assert geom.bounds == (0.0, 0.0, 98.0, 98.0)


def test_edge_non_corner_tile_shrinks_three_sides(tmp_path):
    # Left edge tile: only left sits on the ortho boundary.
    ortho = _ortho_bounds(0, 0, 1000, 1000)
    path = _write_tile_raster(tmp_path / "edge.tif", (0, 300, 100, 400))
    geom, _ = _raster_filter_geom(path, EB, ortho_bounds=ortho)
    assert geom.bounds == (0.0, 302.0, 98.0, 398.0)


def test_tile_spanning_full_ortho_shrinks_nothing(tmp_path):
    ortho = _ortho_bounds(0, 0, 1000, 1000)
    path = _write_tile_raster(tmp_path / "full.tif", (0, 0, 1000, 1000))
    geom, _ = _raster_filter_geom(path, EB, ortho_bounds=ortho)
    assert geom.bounds == (0.0, 0.0, 1000.0, 1000.0)


def test_ortho_bounds_none_keeps_legacy_all_sides_shrink(tmp_path):
    # Without ortho_bounds, even a tile spanning the whole ortho is shrunk on
    # all sides -- the pre-fix behaviour, kept as the fallback for callers
    # that don't pass a stem map.
    path = _write_tile_raster(tmp_path / "legacy.tif", (0, 0, 1000, 1000))
    geom, _ = _raster_filter_geom(path, EB, ortho_bounds=None)
    assert geom.bounds == (2.0, 2.0, 998.0, 998.0)
