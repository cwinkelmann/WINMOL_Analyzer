"""Inner-pool budget rules and the persistent refine pool.

_worker_count used to return 1 in EVERY child process, which made
tile-level and inner-stage parallelism mutually exclusive. The new rule:
daemonic pool workers stay serial (they may not spawn children);
non-daemonic children (the vector stage's tile workers) use the explicit
per-tile budget from config.cpu_workers; children with no explicit budget
stay serial. And refine's process pool must persist across calls instead
of being respawned per tile (~1.6 s per spawn on macOS/Windows).
"""
import copy
from types import SimpleNamespace

import pytest

import helpers
from utils import Quantification as Quant
from utils import Skeletonization as Skel


class _Cfg:
    def __init__(self, cpu_workers):
        self.cpu_workers = cpu_workers


@pytest.mark.parametrize("mod", [Skel, Quant])
def test_main_process_uses_config_budget(mod):
    assert mod._worker_count(_Cfg(5)) == 5


@pytest.mark.parametrize("mod", [Skel, Quant])
def test_daemonic_worker_is_always_serial(mod, monkeypatch):
    fake = SimpleNamespace(name="SpawnPoolWorker-1", daemon=True)
    monkeypatch.setattr(mod.mp, "current_process", lambda: fake)
    assert mod._worker_count(_Cfg(5)) == 1


@pytest.mark.parametrize("mod", [Skel, Quant])
def test_non_daemonic_child_composes_with_explicit_budget(mod, monkeypatch):
    fake = SimpleNamespace(name="SpawnProcess-2", daemon=False)
    monkeypatch.setattr(mod.mp, "current_process", lambda: fake)
    assert mod._worker_count(_Cfg(3)) == 3


@pytest.mark.parametrize("mod", [Skel, Quant])
def test_non_daemonic_child_without_budget_stays_serial(mod, monkeypatch):
    fake = SimpleNamespace(name="SpawnProcess-2", daemon=False)
    monkeypatch.setattr(mod.mp, "current_process", lambda: fake)
    assert mod._worker_count(None) == 1
    assert mod._worker_count(_Cfg(None)) == 1


def test_refine_pool_is_reused_until_size_changes():
    try:
        pool_a = Skel._refine_pool(2)
        assert Skel._refine_pool(2) is pool_a
        pool_b = Skel._refine_pool(3)
        assert pool_b is not pool_a
        assert Skel._refine_pool(3) is pool_b
    finally:
        Skel.close_refine_pool()
    assert Skel._REFINE_POOL is None


@pytest.mark.slow
def test_parallel_refine_matches_serial_refine(stem_map, pipeline_config):
    """find_segments through the persistent pool == serial, twice in a row
    (the second call reuses the pool)."""
    pred, profile = stem_map

    def parts_key(parts):
        return sorted(
            (p.start, p.stop, tuple(map(tuple, p.path))) for p in parts)

    serial_cfg = copy.copy(pipeline_config)
    serial_cfg.cpu_workers = 1
    expected = parts_key(Skel.find_segments(pred, serial_cfg, profile))

    pooled_cfg = copy.copy(pipeline_config)
    pooled_cfg.cpu_workers = 2
    try:
        first = parts_key(Skel.find_segments(pred, pooled_cfg, profile))
        again = parts_key(Skel.find_segments(pred, pooled_cfg, profile))
    finally:
        Skel.close_refine_pool()

    assert first == expected
    assert again == expected


@pytest.mark.slow
def test_quantification_budget_composes_in_fake_tile_worker(
        stem_map, pipeline_config, golden, monkeypatch):
    """Inside a non-daemonic 'tile worker', quantify with a budget must
    match the golden fixture exactly."""
    fake = SimpleNamespace(name="SpawnProcess-9", daemon=False)
    monkeypatch.setattr(Quant.mp, "current_process", lambda: fake)
    pred, profile = stem_map
    stems = helpers.stems_from_canonical(golden("stage_connect_stems"))
    cfg = copy.copy(pipeline_config)
    cfg.cpu_workers = 4
    result = Quant.quantify_stems(list(stems), pred, profile, cfg)
    helpers.assert_stems_match_golden(
        result, golden("stage_quantified_contour"))
