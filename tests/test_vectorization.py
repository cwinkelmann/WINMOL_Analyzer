"""Golden tests for the vector stages. Each stage consumes the PREVIOUS
stage's fixture (not a fresh upstream run), so a failure localizes to exactly
one stage.

Pins current behavior including review-known quirks:
- build_stem_parts' identical if/else branches (V-1): orientation is flipped
  unconditionally — the fixtures record the flipped orientation.
- connect_stems only tries start<->stop pairings (A-11) and dedups through
  0.3 m corridors (A-12).
"""

import helpers
from utils import Vectorization as Vec


def test_restore_geoinformation_matches_golden(pipeline_config, stem_map,
                                               golden):
    _, profile = stem_map
    parts = helpers.parts_from_canonical(golden("stage_find_segments"))
    restored = Vec.restore_geoinformation(parts, pipeline_config, profile)
    helpers.assert_world_parts_match_golden(
        restored, golden("stage_restore_geo"))


def test_build_stem_parts_matches_golden(golden):
    parts = helpers.parts_from_canonical(golden("stage_restore_geo"))
    stems = Vec.build_stem_parts(parts)
    helpers.assert_stems_match_golden(
        stems, golden("stage_build_stem_parts"))


def test_connect_stems_matches_golden(pipeline_config, golden):
    stems = helpers.stems_from_canonical(golden("stage_build_stem_parts"))
    connected = Vec.connect_stems(stems, pipeline_config)
    Vec.rebuild_endnodes_from_stems(connected)  # no-op today (V-8)
    helpers.assert_stems_match_golden(
        connected, golden("stage_connect_stems"))


def test_connect_stems_reduces_fragment_count(golden):
    raw = golden("stage_build_stem_parts")
    connected = golden("stage_connect_stems")
    assert len(connected) < len(raw), (
        "connect_stems is expected to merge/filter fragments on this crop")


def test_connect_stems_empty_input(pipeline_config):
    assert Vec.connect_stems([], pipeline_config) == []


def _mk_stem(coords):
    from shapely.geometry import LineString, Point

    from classes.Stem import Stem
    return Stem(Point(coords[0]), Point(coords[-1]), LineString(coords),
                [], [], [], [])


def _vote_inputs(base, max_distance=3.0):
    line_start, line_stop = Vec._stem_end_lines(base)
    start_buffer = base.start.buffer(max_distance, resolution=32)
    end_buffer = base.stop.buffer(max_distance, resolution=32)
    return line_start, line_stop, start_buffer, end_buffer


def test_score_then_build_appends_collinear_follower():
    """Deferred-geometry path, branch 1: a collinear stem starting just
    past base.stop is scored without geometry, then built into the same
    merged stem the old inline construction produced."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge

    base = _mk_stem([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    follower = _mk_stem([(6, 0), (7, 0), (8, 0), (9, 0), (10, 0)])
    line_start, line_stop, start_buffer, end_buffer = _vote_inputs(base)

    scored = Vec._score_connectivity(
        base, line_start, line_stop, start_buffer, end_buffer,
        3.0, 40.0, 5.0, follower)
    assert scored is not None and scored[1] == 1

    changed, vote, cand, slave = Vec.calc_connectivity_votes(
        base, line_start, line_stop, start_buffer, end_buffer,
        3.0, 40.0, 5.0, follower)
    assert changed and vote == scored[0] and slave is follower

    expected = linemerge([
        LineString(base.path.coords[:-1]),
        LineString([base.path.coords[-2], follower.path.coords[1]]),
        LineString(follower.path.coords[1:]),
    ])
    assert list(cand.path.coords) == list(expected.coords)
    assert cand.start.coords[0] == (0.0, 0.0)
    assert cand.stop.coords[0] == (10.0, 0.0)


def test_score_then_build_prepends_collinear_leader():
    """Branch 2 (mirrored): the candidate ends just before base.start."""
    base = _mk_stem([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    leader = _mk_stem([(-8, 0), (-7, 0), (-6, 0), (-3, 0), (-2, 0)])
    line_start, line_stop, start_buffer, end_buffer = _vote_inputs(base)

    scored = Vec._score_connectivity(
        base, line_start, line_stop, start_buffer, end_buffer,
        3.0, 40.0, 5.0, leader)
    assert scored is not None and scored[1] == 2

    cand = Vec._build_connection(base, leader, scored[1])
    assert cand.start.coords[0] == (-8.0, 0.0)
    assert cand.stop.coords[0] == (4.0, 0.0)
    assert cand.path.coords[0] == (-8.0, 0.0)
    assert cand.path.coords[-1] == (4.0, 0.0)


def test_score_connectivity_identical_stem_scores_none():
    base = _mk_stem([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    twin = _mk_stem([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    line_start, line_stop, start_buffer, end_buffer = _vote_inputs(base)
    assert Vec._score_connectivity(
        base, line_start, line_stop, start_buffer, end_buffer,
        3.0, 40.0, 5.0, twin) is None

    changed, vote, cand, slave = Vec.calc_connectivity_votes(
        base, line_start, line_stop, start_buffer, end_buffer,
        3.0, 40.0, 5.0, twin)
    assert changed is False and cand is None and slave is None


def test_score_connectivity_rejects_far_candidate():
    base = _mk_stem([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    far = _mk_stem([(50, 0), (51, 0), (52, 0), (53, 0), (54, 0)])
    line_start, line_stop, start_buffer, end_buffer = _vote_inputs(base)
    assert Vec._score_connectivity(
        base, line_start, line_stop, start_buffer, end_buffer,
        3.0, 40.0, 5.0, far) is None
