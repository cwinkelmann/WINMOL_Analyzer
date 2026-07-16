"""Pure unit tests — no fixtures, no model, no TF.

These pin the exact CURRENT behavior of small deterministic functions,
including known quirks from the review docs (IDs referenced inline). When a
quirk is fixed, the corresponding assertion here must be updated in the same
commit — that is the point: behavior changes must be explicit.
"""

import math

import numpy as np
import pytest

from classes.Part import Part
from utils.Geometry import ang, create_vector
from utils.Quantification import calc_l_v, clean_diameter
from utils.Tiling import build_tile_grid, meters_to_pixels


# ---------------------------------------------------------------------------
# Geometry.ang — returns |angle| in [0, 180]; the %380 and sign branches are
# dead (review V-10). Collinear segments give ~0.
# ---------------------------------------------------------------------------

def _line(p1, p2):
    return [p1, p2]


def test_ang_collinear_is_zero():
    # create_vector pads the norm with machine epsilon, so angles carry
    # ~1e-6 deg of noise — tolerance reflects that.
    a = _line((0.0, 0.0), (1.0, 0.0))
    b = _line((2.0, 0.0), (3.0, 0.0))
    assert abs(ang(a, b)) < 1e-4


def test_ang_perpendicular_is_ninety():
    a = _line((0.0, 0.0), (1.0, 0.0))
    b = _line((0.0, 0.0), (0.0, 1.0))
    assert math.isclose(abs(ang(a, b)), 90.0, abs_tol=1e-4)


def test_ang_antiparallel_is_180_not_zero():
    # Anti-parallel directions read as 180 deg, NOT collinear — this is why
    # connect_stems is orientation-sensitive (review A-11 / V-1).
    a = _line((0.0, 0.0), (1.0, 0.0))
    b = _line((3.0, 0.0), (2.0, 0.0))
    assert math.isclose(abs(ang(a, b)), 180.0, abs_tol=1e-4)


def test_create_vector_is_normalized():
    v = np.asarray(create_vector(((1.0, 2.0), (4.0, 6.0))))
    assert np.allclose(v, [0.6, 0.8])  # (3,4)/5


# ---------------------------------------------------------------------------
# calc_l_v — truncated-cone (frustum) volume; diameters are halved to radii
# (verified NOT the 4x bug — CODE_REVIEW_2 refuted list)
# ---------------------------------------------------------------------------

def test_calc_l_v_cylinder():
    length, volume = calc_l_v((0, 0), (0, 2.0), 0.4, 0.4)
    assert math.isclose(length, 2.0, rel_tol=1e-12)
    assert math.isclose(volume, math.pi * 0.2 ** 2 * 2.0, rel_tol=1e-12)


def test_calc_l_v_cone():
    length, volume = calc_l_v((0, 0), (3.0, 0), 0.5, 0.0)
    assert math.isclose(length, 3.0, rel_tol=1e-12)
    assert math.isclose(volume, math.pi * 0.25 ** 2 * 3.0 / 3.0,
                        rel_tol=1e-12)


# ---------------------------------------------------------------------------
# clean_diameter — pins A-13: stems with 3 or 4 diameters are never corrected
# (quantiles computed then discarded); second-to-last node never smoothed
# (first-pass V-5).
# ---------------------------------------------------------------------------

class _FakePath:
    def __init__(self, coords):
        self.coords = coords


class _FakeStem:
    def __init__(self, diameters, coords=None):
        self.segment_diameter_list = list(diameters)
        self.path = _FakePath(coords or
                              [(float(i), 0.0)
                               for i in range(len(diameters))])


def test_clean_diameter_three_nodes_is_noop():
    stem = _FakeStem([0.3, 99.0, 0.3])  # blatant outlier stays (A-13a)
    clean_diameter(stem)
    assert stem.segment_diameter_list == [0.3, 99.0, 0.3]


def test_clean_diameter_interior_outlier_interpolated():
    stem = _FakeStem([0.4, 0.4, 9.0, 0.4, 0.4, 0.4])
    clean_diameter(stem)
    assert stem.segment_diameter_list[2] == pytest.approx(0.4, rel=1e-9)


def test_clean_diameter_second_to_last_not_corrected():
    # V-5: range(1, n-2) skips index n-2 — outlier there survives.
    stem = _FakeStem([0.4, 0.4, 0.4, 0.4, 9.0, 0.4])
    clean_diameter(stem)
    assert stem.segment_diameter_list[4] == 9.0


# ---------------------------------------------------------------------------
# Part identity — pins A-4 mechanics: eq/hash ignore the path, and the hash
# tuple contains strings (=> PYTHONHASHSEED-dependent set ordering).
# ---------------------------------------------------------------------------

def test_part_eq_ignores_path():
    a = Part((0, 0), (5, 5), [(0, 0), (1, 1), (5, 5)], (0, 0), (5, 5))
    b = Part((0, 0), (5, 5), [(0, 0), (4, 1), (5, 5)], (0, 0), (5, 5))
    assert a == b               # different geometry, "equal" parts
    assert len({a, b}) == 1     # set() dedup drops one of them


# ---------------------------------------------------------------------------
# Tiling math — exact
# ---------------------------------------------------------------------------

def test_meters_to_pixels():
    assert meters_to_pixels(12.0, 0.04, -0.04) == 300
    assert meters_to_pixels(0.0, 0.04, 0.04) == 0


def test_build_tile_grid_covers_raster_exactly():
    jobs = build_tile_grid(1000, 700, 512, 100)
    # inner windows tile the raster without gaps or overlap
    cells = set()
    for j in jobs:
        assert 0 <= j.x0 < j.x1 <= 1000
        assert 0 <= j.y0 < j.y1 <= 700
        assert j.hx0 <= j.x0 and j.hx1 >= j.x1      # halo contains inner
        assert j.hx0 >= 0 and j.hy0 >= 0            # halo clamped
        assert j.hx1 <= 1000 and j.hy1 <= 700
        cells.add((j.x0, j.y0, j.x1, j.y1))
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in cells)
    assert area == 1000 * 700


# ---------------------------------------------------------------------------
# _to_float32_image — pins A-2: EVERY integer dtype is divided by 255
# (uint16 imagery comes out in [0, 257]); non-float32 floats pass unscaled.
# ---------------------------------------------------------------------------

def test_to_float32_image_dtype_behavior():
    from utils.Prediction import _to_float32_image
    u8 = np.array([0, 255], dtype=np.uint8)
    assert np.allclose(_to_float32_image(u8), [0.0, 1.0])
    u16 = np.array([0, 65535], dtype=np.uint16)
    out = _to_float32_image(u16)
    assert out.max() > 200.0            # A-2: not rescaled to [0, 1]
    f64 = np.array([0.0, 200.0])
    assert _to_float32_image(f64).max() == 200.0   # floats pass through


# ---------------------------------------------------------------------------
# _binarize_prediction_core — threshold semantics
# ---------------------------------------------------------------------------

def test_binarize_prediction_core():
    from utils.Prediction import _binarize_prediction_core
    pred = np.array([[0.49, 0.5], [0.51, 1.0]], dtype=np.float32)
    mask = np.ones_like(pred, dtype=bool)
    out = _binarize_prediction_core(pred, mask, threshold=0.5)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1], [1, 1]]        # >= threshold
    mask[1, 1] = False
    assert _binarize_prediction_core(pred, mask, 0.5).tolist() == \
        [[0, 1], [1, 0]]


# ---------------------------------------------------------------------------
# _merge_diameter_lists — the A-1 fix: merged stems carry merged diameters
# ---------------------------------------------------------------------------

def test_merge_diameter_lists_full_parents():
    from shapely.geometry import LineString, Point
    from classes.Stem import Stem
    from utils.Vectorization import _merge_diameter_lists

    a = Stem(Point(0, 0), Point(2, 0),
             LineString([(0, 0), (1, 0), (2, 0)]), [],
             [0.5, 0.6, 0.7], [], [])
    b = Stem(Point(3, 0), Point(5, 0),
             LineString([(3, 0), (4, 0), (5, 0)]), [],
             [0.8, 0.9, 1.0], [], [])
    # merged path = a[:-1] + b[1:] -> 4 coords -> 4 diameters
    merged = _merge_diameter_lists(a, b)
    assert merged == [0.5, 0.6, 0.9, 1.0]


def test_merge_diameter_lists_empty_parents_stay_empty():
    from shapely.geometry import LineString, Point
    from classes.Stem import Stem
    from utils.Vectorization import _merge_diameter_lists

    a = Stem(Point(0, 0), Point(1, 0),
             LineString([(0, 0), (1, 0)]), [], [], [], [])
    b = Stem(Point(2, 0), Point(3, 0),
             LineString([(2, 0), (3, 0)]), [], [], [], [])
    assert _merge_diameter_lists(a, b) == []  # in-tile case: unchanged
