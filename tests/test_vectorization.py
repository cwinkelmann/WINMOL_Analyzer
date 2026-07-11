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
