"""Golden test: stem_map.tif -> Skel.find_segments == recorded fixture.

Exact structural comparison (pixel-space integers). Deterministic only under
PYTHONHASHSEED=0 + serial workers — both enforced by conftest/config
snapshot. NOTE this pins currently-buggy behavior on purpose (e.g. the
min_length/16 gate, review V-3, and 10/30-degree hardcoded splits, V-7);
fixing those requires regenerating fixtures in the same commit.
"""

import helpers
from utils import Skeletonization as Skel


def test_find_segments_matches_golden(stem_map, pipeline_config, golden):
    pred, profile = stem_map
    parts = Skel.find_segments(pred, pipeline_config, profile)
    helpers.assert_parts_match_golden(parts, golden("stage_find_segments"))


def test_find_segments_empty_mask_yields_no_parts(stem_map, pipeline_config):
    import numpy as np
    pred, profile = stem_map
    empty = np.zeros_like(pred)
    parts = Skel.find_segments(empty, pipeline_config, profile)
    assert len(parts) == 0


def test_neighbors_from_bytes_matches_get_neighbors():
    """The bytes-snapshot scan used by the trace walks must return exactly
    what get_neighbors returns, in the same order, for every pixel
    (including corners and edges)."""
    import numpy as np
    rng = np.random.default_rng(42)
    skel = rng.random((13, 17)) < 0.4
    skel = np.ascontiguousarray(skel)
    h, w = skel.shape
    flat = skel.tobytes()
    for x in range(h):
        for y in range(w):
            assert Skel._neighbors_from_bytes(x, y, flat, h, w) == \
                Skel.get_neighbors(x, y, skel)
