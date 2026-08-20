"""Regression tests for issue #41: IndexError in the tiled edge-merge.

connect_stems fuses two stems' paths as ``first[:-1] + second[1:]``. It must
merge their per-node diameter lists the same way, or a later quantify_stem in
the cross-tile merge indexes ``segment_diameter_list[i + 1]`` off the end and
the whole run aborts at the very last step. The single-tile pipeline hides
this because it re-measures diameters after connect_stems; the tiled merge
quantifies the connect_stems output directly.
"""
from shapely import LineString, Point

import pytest

from classes.Config import Config
from classes.Stem import Stem
from utils.Quantification import quantify_stem
from utils.Vectorization import _merge_diameter_lists, connect_stems


def _stem(coords, diameters):
    return Stem(
        start=Point(coords[0]),
        stop=Point(coords[-1]),
        path=LineString(coords),
        vector=[],
        segment_diameter_list=list(diameters),
        segment_length_list=[],
        segment_volume_list=[],
    )


def test_merge_diameter_lists_matches_merged_path_length():
    first = _stem([(0, 0), (1, 0), (2, 0)], [0.30, 0.28, 0.26])
    second = _stem([(2, 0), (3, 0), (4, 0)], [0.26, 0.24, 0.22])

    merged = _merge_diameter_lists(first, second)

    # merged path = first[:-1] + second[1:]: the shared junction node is
    # dropped, so the diameters are d_first[:-1] + d_second[1:], one per node
    merged_coords = list(first.path.coords)[:-1] + list(second.path.coords)[1:]
    assert merged == [0.30, 0.28, 0.24, 0.22]
    assert len(merged) == len(merged_coords)


def test_merge_diameter_lists_empty_when_a_parent_is_unmeasured():
    """In-tile connect_stems runs before quantification (lists empty); the
    merge must stay a no-op there rather than fabricate a mismatched list."""
    first = _stem([(0, 0), (1, 0), (2, 0)], [0.30, 0.28, 0.26])
    second = _stem([(2, 0), (3, 0), (4, 0)], [])

    assert _merge_diameter_lists(first, second) == []


def test_quantify_stem_survives_a_consistent_merged_stem():
    merged_coords = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    stem = _stem(merged_coords, [0.30, 0.28, 0.26, 0.24, 0.22])

    out = quantify_stem(stem)

    assert len(out.segment_length_list) == len(merged_coords) - 1
    assert len(out.segment_volume_list) == len(merged_coords) - 1


def test_connect_stems_merges_diameter_lists_end_to_end():
    """The actual #41 fix: connect_stems must merge the parents' per-node
    diameter lists alongside their paths, so the merged stem stays
    quantifiable. Before the fix the merged stem kept the base's shorter list
    (path 4 nodes, diameters 3) and quantify_stem IndexError'd at merge."""
    a = _stem([(0, 0), (1, 0), (2, 0)], [0.30, 0.28, 0.26])
    b = _stem([(3, 0), (4, 0), (5, 0)], [0.24, 0.22, 0.20])

    merged = connect_stems([a, b], Config())

    assert len(merged) == 1
    stem = merged[0]
    assert len(stem.segment_diameter_list) == len(stem.path.coords)
    quantify_stem(stem)  # would raise IndexError without the fix


def test_quantify_stem_indexerrors_on_short_diameter_list():
    """The exact precondition that crashed at merge time. The guard in
    _reconstruct_edge_stems_for_tiled_merge filters these out with the same
    len() check asserted here, so quantify is never reached in that state."""
    stem = _stem([(0, 0), (1, 0), (2, 0)], [0.30, 0.28])  # 3 nodes, 2 diam

    assert len(stem.segment_diameter_list) != len(stem.path.coords)
    with pytest.raises(IndexError):
        quantify_stem(stem)
