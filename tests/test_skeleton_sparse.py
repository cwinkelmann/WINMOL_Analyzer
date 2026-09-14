"""The sparse skeleton stage must match the dense formulation pixel for
pixel, and find_segments must not depend on where the foreground sits.

Each sparse helper is checked against the plain dense expression it
replaced (the one that used to be in the module), on random masks with
the occupancy and structure of real stem maps: thin elongated blobs,
some touching, some crossing, some at the tile edge.
"""
import os
import sys

import numpy as np
import pytest
import scipy.ndimage
from skimage import morphology

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils import Skeletonization as Skel  # noqa: E402


def _stem_mask(seed, size=600, stems=25, edge=False):
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(stems):
        y0, x0 = rng.uniform(0, size, 2)
        if edge:
            y0 = rng.choice([0.0, size - 1.0])
        length = rng.uniform(40, 200)
        angle = rng.uniform(0, np.pi)
        width = rng.uniform(1.5, 6)
        dy, dx = np.sin(angle), np.cos(angle)
        t = (yy - y0) * dy + (xx - x0) * dx
        d = np.abs((yy - y0) * dx - (xx - x0) * dy)
        mask |= (t >= 0) & (t <= length) & (d <= width)
    return mask


def _dense_find_skeleton_nodes(skel):
    p = np.pad(skel, 1)
    p2, p3, p4, p5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
    p6, p7, p8, p9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
    tr = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + ((p3 == 0) & (p4 == 1)) \
        + ((p4 == 0) & (p5 == 1)) + ((p5 == 0) & (p6 == 1)) \
        + ((p6 == 0) & (p7 == 1)) + ((p7 == 0) & (p8 == 1)) \
        + ((p8 == 0) & (p9 == 1)) + ((p9 == 0) & (p2 == 1))
    mask = p[1:-1, 1:-1] == 1
    ends = [tuple(map(int, q)) for q in np.argwhere((tr == 1) & mask)]
    branches = [tuple(map(int, q)) for q in np.argwhere((tr >= 3) & mask)]
    return ends, branches


@pytest.mark.parametrize('seed', range(6))
def test_per_component_skeletonize_matches_whole_image(seed):
    mask = _stem_mask(seed, edge=(seed % 2 == 1))
    expected = morphology.skeletonize(mask)
    assert np.array_equal(Skel._skeletonize_sparse(mask), expected)


@pytest.mark.parametrize('seed', range(6))
def test_sparse_node_detection_matches_dense(seed):
    skel = morphology.skeletonize(_stem_mask(seed))
    assert Skel.find_skeleton_nodes(skel.copy()) == \
        _dense_find_skeleton_nodes(skel)


@pytest.mark.parametrize('seed', range(6))
def test_sparse_dense_node_removal_matches_erosion(seed):
    skel = morphology.skeletonize(_stem_mask(seed))
    # A few 2x2 blocks planted so the erosion has something to find.
    rng = np.random.default_rng(seed)
    for _ in range(20):
        y, x = rng.integers(1, skel.shape[0] - 2, 2)
        skel[y:y + 2, x:x + 2] = True
    dense = morphology.binary_erosion(
        np.pad(skel, 1), np.ones((2, 2)))[1:-1, 1:-1]
    expected = skel.copy()
    expected[dense] = False
    expected_count = scipy.ndimage.label(dense)[1]
    got, count = Skel.remove_dense_skeleton_nodes(skel.copy())
    assert np.array_equal(got, expected)
    assert count == expected_count


class _Cfg:
    min_length = 2.0
    max_tree_height = 32
    measuring_point_spacing_m = 0.5
    cpu_workers = 1


def _parts_as_tuples(parts):
    return [(p.start, p.stop, tuple(p.path), p.l_bound, p.u_bound)
            for p in parts]


@pytest.mark.parametrize('seed', range(3))
def test_find_segments_is_translation_invariant(seed):
    """Shifting the foreground inside the tile shifts every part by the
    same integer offset and changes nothing else -- which is what the
    crop-plus-offset construction relies on."""
    profile = {'transform': (0.05, 0, 0, 0, -0.05, 0)}
    mask = _stem_mask(seed, size=400, stems=12)
    canvas = np.zeros((700, 700), dtype=bool)
    canvas[50:450, 60:460] = mask
    shifted = np.zeros((700, 700), dtype=bool)
    shifted[200:600, 150:550] = mask
    a = _parts_as_tuples(Skel.find_segments(canvas, _Cfg(), profile))
    b = _parts_as_tuples(Skel.find_segments(shifted, _Cfg(), profile))
    assert len(a) > 0
    dy, dx = 150, 90

    def move(part):
        (s, e, path, lo, up) = part
        sh = lambda q: (q[0] + dy, q[1] + dx)  # noqa: E731
        return (sh(s), sh(e), tuple(sh(q) for q in path), sh(lo), sh(up))
    # As multisets: find_skeleton_segments hands its parts through a
    # set(), whose iteration order follows the coordinate hashes, so the
    # ORDER legitimately changes with the shift (it did before the crop
    # too). The bit-identity that matters is for the same input, which
    # the reference comparison on real tiles covers.
    assert sorted(move(p) for p in a) == sorted(b)


def test_find_segments_with_padding_below_the_margin():
    """A small max_tree_height gives a frame padding below the 8 px crop
    margin; with foreground on the tile edge the crop must still map
    inside the padded frame (it used to slice out of it)."""
    profile = {'transform': (0.05, 0, 0, 0, -0.05, 0)}

    class Cfg(_Cfg):
        max_tree_height = 0.2          # padding = int(0.2 / 0.05) + 1 = 5

    mask = np.zeros((300, 300), dtype=bool)
    mask[0:3, 0:120] = True            # touches row 0 and column 0
    mask[297:300, 180:300] = True      # touches the far edges
    parts = Skel.find_segments(mask, Cfg(), profile)
    assert len(parts) >= 2
    for p in parts:
        assert all(0 <= r < 310 and 0 <= c < 310 for r, c in p.path)


def test_dense_occupancy_path_matches_per_component_path(monkeypatch):
    """Above _DENSE_OCCUPANCY the skeleton is thinned on the dense array;
    both routes must give the same pixels."""
    mask = _stem_mask(7, size=200, stems=60)
    ys, xs = np.nonzero(mask)
    sparse = Skel._skeletonize_coords(ys, xs, mask.shape)
    monkeypatch.setattr(Skel, '_DENSE_OCCUPANCY', 0.0)
    dense = Skel._skeletonize_coords(ys, xs, mask.shape)
    assert np.array_equal(sparse[0], dense[0])
    assert np.array_equal(sparse[1], dense[1])
    assert np.array_equal(sparse[2], dense[2])
    assert np.array_equal(sparse[0], morphology.skeletonize(mask))


def test_find_segments_empty_mask_returns_no_parts():
    profile = {'transform': (0.05, 0, 0, 0, -0.05, 0)}
    assert Skel.find_segments(
        np.zeros((300, 300), dtype=bool), _Cfg(), profile) == []
