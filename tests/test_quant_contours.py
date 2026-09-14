"""Quantification polygonises the foreground on its bounding box and
places the vertices itself with GDAL's formula. That must equal what
rasterio.features.shapes returns for the whole tile with the transform
applied by GDAL -- coordinate for coordinate, bit for bit -- including
holes, thick blobs, and foreground touching the tile edge.
"""
import os
import sys

import numpy as np
import pytest
import rasterio.features
from affine import Affine
from shapely.geometry import shape

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.Quantification import _foreground_contours  # noqa: E402


def _tile(seed):
    rng = np.random.default_rng(seed)
    h, w = rng.integers(60, 300, 2)
    arr = np.zeros((h, w), dtype=np.uint8)
    for _ in range(rng.integers(3, 12)):
        y, x = rng.integers(0, h - 8), rng.integers(0, w - 8)
        arr[y:y + rng.integers(2, 8), x:x + rng.integers(2, 30)] = 1
    # a blob with a hole, and foreground on two edges
    arr[10:30, 10:30] = 1
    arr[15:25, 15:25] = 0
    arr[0, :5] = 1
    arr[:, w - 1] = 1
    if seed % 2:
        arr[:, :] = 0
        arr[h // 2, w // 2] = 1          # single pixel
    return arr


def _transform(seed):
    rng = np.random.default_rng(1000 + seed)
    a = rng.uniform(0.01, 2.0)
    if seed % 3:
        b, d = 0.0, 0.0
    else:
        b, d = rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)
    c = rng.uniform(-1e7, 1e7)
    e = -rng.uniform(0.01, 2.0)
    f = rng.uniform(-1e7, 1e7)
    return Affine(a, b, c, d, e, f)


@pytest.mark.parametrize('seed', range(12))
def test_cropped_contours_equal_gdal_full_tile(seed):
    arr = _tile(seed)
    transform = _transform(seed)
    expected = [
        shape(g) for g, v in rasterio.features.shapes(
            arr, mask=arr, transform=transform) if v == 1
    ]
    got = _foreground_contours(arr, arr, transform)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g.exterior.coords[:] == e.exterior.coords[:]
        assert len(g.interiors) == len(e.interiors)
        for gi, ei in zip(g.interiors, e.interiors):
            assert gi.coords[:] == ei.coords[:]


def test_empty_mask_gives_no_contours():
    arr = np.zeros((40, 40), dtype=np.uint8)
    assert _foreground_contours(arr, arr, Affine.identity()) == []


def test_self_check_passes_on_this_gdal_and_fallback_is_equivalent(
        monkeypatch):
    """The crop is only taken when the process-level self-check says this
    GDAL places vertices with the replicated formula; on a build where it
    does not, GDAL polygonises the whole tile. Both paths must agree."""
    import utils.Quantification as Q
    Q._VERTEX_FORMULA_HOLDS = None
    assert Q._gdal_vertex_formula_holds() is True
    arr = _tile(4)
    transform = _transform(4)
    cropped = _foreground_contours(arr, arr, transform)
    monkeypatch.setattr(Q, '_VERTEX_FORMULA_HOLDS', False)
    whole = _foreground_contours(arr, arr, transform)
    Q._VERTEX_FORMULA_HOLDS = None
    assert [g.wkb for g in cropped] == [g.wkb for g in whole]
