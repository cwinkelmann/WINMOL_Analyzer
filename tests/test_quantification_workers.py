"""Quantification must give the same answer however it is parallelised.

The stem-level stages were farmed out to a PROCESS pool, which pickled every
Stem (shapely geometry + diameter lists) to a worker and back. Measured on one
4096-px tile with 1567 stems, that made quantification ~10x SLOWER than doing
it serially (45.2 s vs 4.6 s) — the transport cost dwarfed 4.6 s of actual
work. These tests pin the invariant that matters while that is changed: the
result must not depend on the worker count.
"""
import copy

import pytest

import helpers
from utils import Quantification as Quant


@pytest.fixture()
def connect_stage_stems(golden):
    return helpers.stems_from_canonical(golden("stage_connect_stems"))


def _quantify_with_workers(stems, pred, profile, config, workers):
    cfg = copy.copy(config)
    cfg.cpu_workers = workers
    return Quant.quantify_stems(list(stems), pred, profile, cfg)


def test_single_and_multi_worker_agree(connect_stage_stems, stem_map,
                                       pipeline_config, golden):
    """Whatever the pool does, one worker and several must agree."""
    pred, profile = stem_map
    one = _quantify_with_workers(connect_stage_stems, pred, profile,
                                 pipeline_config, 1)
    many = _quantify_with_workers(connect_stage_stems, pred, profile,
                                  pipeline_config, 4)

    assert len(one) == len(many)
    # imap_unordered does not preserve order, so compare order-independently.
    key = sorted(s.path.wkb_hex for s in one)
    assert key == sorted(s.path.wkb_hex for s in many)


def test_multi_worker_still_matches_the_golden_fixture(
        connect_stage_stems, stem_map, pipeline_config, golden):
    """The golden test runs at the default worker count; pin the multi-worker
    path against the same fixture so a pooling change cannot silently drift."""
    pred, profile = stem_map
    stems = _quantify_with_workers(connect_stage_stems, pred, profile,
                                   pipeline_config, 4)
    helpers.assert_stems_match_golden(
        stems, golden("stage_quantified_contour"))
