"""The hoisted spatial indexes must not change a single result.

Each test compares the current implementation against an inline brute-force
reference implementing the PRE-hoist semantics exactly. The tests pass both
before the hoist (proving the reference matches the old behavior) and after
(proving the hoisted version is bit-identical to it).
"""
import numpy as np
from shapely.geometry import LineString, Point

import geopandas as gpd

from classes.Stem import Stem
from utils import Quantification as Quant
from utils import Skeletonization as Skel
from utils.Vectorization import remove_duplicates


def _stem(path: LineString) -> Stem:
    return Stem(start=Point(path.coords[0]), stop=Point(path.coords[-1]),
                path=path, vector=[], segment_diameter_list=[],
                segment_length_list=[], segment_volume_list=[])


def _random_stems(n=200, seed=1234):
    rng = np.random.default_rng(seed)
    stems = []
    for _ in range(n):
        x, y = rng.uniform(0.0, 25.0, 2)
        a = rng.uniform(0.0, 2.0 * np.pi)
        length = rng.uniform(0.2, 5.0)
        path = LineString([(x, y),
                           (x + np.cos(a) * length, y + np.sin(a) * length)])
        stems.append(_stem(path))
        if rng.random() < 0.4:
            # A sub-segment sits inside the parent's 0.3 m buffer, so real
            # duplicates exist and the containment branch is exercised.
            t0, t1 = sorted(rng.uniform(0.05, 0.95, 2))
            if t1 - t0 > 0.05:
                sub = LineString([path.interpolate(t0, normalized=True),
                                  path.interpolate(t1, normalized=True)])
                stems.append(_stem(sub))
    return stems


def _remove_duplicates_reference(stems):
    """The original O(n^2) pop-the-longest / drop-the-contained loop."""
    stems = list(stems)
    stems.sort(key=lambda s: s.length, reverse=True)
    count = 0
    kept = []
    while stems:
        base = stems.pop(0)
        buffer_geom = base.path.buffer(0.3)
        survivors = []
        for s in stems:
            try:
                if buffer_geom.contains(s.path):
                    count += 1
                    continue
            except Exception:
                pass
            survivors.append(s)
        kept.append(base)
        stems = survivors
    return kept, count


def test_remove_duplicates_matches_brute_force():
    stems = _random_stems()
    expected, expected_count = _remove_duplicates_reference(stems)
    kept, count = remove_duplicates(stems)

    assert expected_count > 0  # the containment branch really ran
    assert count == expected_count
    # Same objects in the same order, not merely equal geometries.
    assert [id(s) for s in kept] == [id(s) for s in expected]


def _calc_d_scene(seed=99, n_polys=60, n_cases=120):
    rng = np.random.default_rng(seed)
    polys = []
    for _ in range(n_polys):
        x, y = rng.uniform(0.0, 40.0, 2)
        polys.append(Point(x, y).buffer(rng.uniform(0.1, 1.2),
                                        quad_segs=int(rng.integers(2, 8))))
    gdf = gpd.GeoDataFrame(geometry=polys)

    cases = []
    for i in range(n_cases):
        if i % 2 == 0:
            # Through a polygon centroid: node lies on the intersection,
            # so the distance-gated max-length branch is taken.
            c = polys[int(rng.integers(0, n_polys))].centroid
            node = (c.x, c.y)
        else:
            node = tuple(rng.uniform(0.0, 40.0, 2))
        a = rng.uniform(0.0, 2.0 * np.pi)
        half = rng.uniform(0.5, 2.0)
        dx, dy = np.cos(a) * half, np.sin(a) * half
        line = LineString([(node[0] - dx, node[1] - dy),
                           (node[0] + dx, node[1] + dy)])
        cases.append((node, line))
    return gdf, cases


def _calc_d_reference(node, line, contours):
    """The original full-scan calc_d: intersect against EVERY contour."""
    node = Point(node)
    d = 0
    intersects = contours.geometry.intersection(line)
    intersects = intersects[~intersects.is_empty]
    for i in intersects:
        if node.distance(i) < 0.01:
            if i.geom_type == 'MultiLineString':
                for i_ in i.geoms:
                    if node.distance(i_) < 0.01:
                        d = max(d, i_.length)
            else:
                d = max(d, i.length)
    return d


def test_calc_d_matches_brute_force():
    gdf, cases = _calc_d_scene()
    contour_index = getattr(Quant, "ContourIndex", None)
    hits = 0
    for node, line in cases:
        expected = _calc_d_reference(node, line, gdf)
        hits += expected > 0
        # GeoDataFrame path (the pre-hoist signature / post-hoist fallback).
        assert Quant.calc_d(node, line, gdf) == expected
        if contour_index is not None:
            assert Quant.calc_d(node, line, contour_index(gdf)) == expected
    assert hits > 0  # nonzero diameters were actually measured


def _get_neighbors_reference(x, y, skel):
    """The original numpy implementation."""
    offsets = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 1],
        [1, -1], [1, 0], [1, 1],
    ])
    coords = offsets + [x, y]
    h, w = skel.shape
    mask = (
        (coords[:, 0] >= 0) & (coords[:, 0] < h) &
        (coords[:, 1] >= 0) & (coords[:, 1] < w)
    )
    valid_coords = coords[mask]
    is_skeleton = skel[valid_coords[:, 0], valid_coords[:, 1]]
    result = valid_coords[is_skeleton != 0]
    return [tuple(pt) for pt in result]


def test_get_neighbors_matches_numpy_reference():
    rng = np.random.default_rng(7)
    skel = (rng.random((16, 16)) < 0.4).astype(np.uint8)
    for x in range(skel.shape[0]):
        for y in range(skel.shape[1]):
            assert Skel.get_neighbors(x, y, skel) == \
                _get_neighbors_reference(x, y, skel)
