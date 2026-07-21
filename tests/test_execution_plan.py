"""Scheduler unit tests: the vector-phase worker split must COMPOSE.

The old _vector_worker_split returned either (N, 1) or (1, N) — tile-level
and inner-stage parallelism were mutually exclusive — and gated the tile
branch on cpu_workers // 4 and on the estimated PREDICTION tile count.
Measured effect: vector wall time equalled the serial sum of per-tile
totals. These tests pin the composed scheduling contract.
"""

from types import SimpleNamespace

from classes.Config import Config
from classes.ExecutionPlan import (
    _estimate_vector_tiles,
    _vector_worker_split,
    build_execution_plan,
)


def split(cpu_workers, tiles, hw_cpu=16, process_type='Trees',
          huge=False, config=None):
    return _vector_worker_split(
        config or Config(), hw_cpu, cpu_workers, tiles, process_type, huge)


def test_stems_keeps_full_inner_budget():
    assert split(12, 100, process_type='Stems') == (1, 12)


def test_single_vector_tile_stays_serial():
    assert split(12, 1) == (1, 12)


def test_small_machines_stay_serial():
    assert split(3, 9, hw_cpu=4) == (1, 3)


def test_tile_and_inner_workers_compose():
    # 16 workers over 9 tiles: 4 tile workers x 4 inner workers.
    assert split(16, 9) == (4, 4)


def test_cpu_div_4_gate_is_gone():
    # 7 CPU workers used to yield 7 // 4 = 1 tile worker -> fully serial
    # tiles. Now the budget prefers tile-level overlap.
    tile_w, inner_w = split(7, 9)
    assert tile_w == 4
    assert inner_w == 1


def test_tile_workers_clamped_by_tile_count():
    assert split(16, 2) == (2, 8)


def test_budget_never_oversubscribed():
    for cpu in range(1, 33):
        for tiles in (1, 2, 3, 5, 9, 40):
            tile_w, inner_w = split(cpu, tiles)
            assert tile_w >= 1 and inner_w >= 1
            assert tile_w * inner_w <= max(cpu, 1)


def test_max_vector_tile_workers_is_respected():
    config = Config()
    config.max_vector_tile_workers = 2
    assert split(16, 9, config=config) == (2, 8)


def test_huge_job_serial_path_keeps_trimmed_inner_pool():
    # Memory guard unchanged: hw_cpu // 5 inner workers when serial.
    assert split(7, 9, hw_cpu=10, huge=True) == (1, 2)


def test_huge_job_parallel_path_keeps_old_conservative_split():
    # cpu_workers // 4 cap and (N, 1) shape are preserved for huge jobs.
    assert split(16, 9, huge=True) == (4, 1)


def test_estimate_vector_tiles_uses_prediction_grid():
    # 1520 px at 0.02936 m/px is ~44.6 m -> ~1523 px on the prediction
    # grid (15 m / 512 px): one 4096 px vector tile, nine 512 px tiles.
    config = Config()
    raster = SimpleNamespace(
        width=1520, height=1520, bands=1, dtype='uint8',
        pixel_size_x=0.02936221178130864,
        pixel_size_y=0.02936221178130864,
        estimated_input_gb=0.002,
    )
    assert _estimate_vector_tiles(config, raster) == 1
    config.tile_inner_px = 512
    assert _estimate_vector_tiles(config, raster) == 9


def test_plan_reports_vector_tiles_and_coherent_split():
    config = Config()
    config.tile_inner_px = 512
    hardware = SimpleNamespace(
        cpu_count=16, total_ram_gb=32.0, gpu_count=0,
        gpu_names=[], gpu_memory_gb=[],
    )
    raster = dict(
        width=1520, height=1520, bands=1, dtype='uint8',
        pixel_size_x=0.02936221178130864,
        pixel_size_y=0.02936221178130864,
        estimated_input_gb=0.002,
    )
    plan = build_execution_plan(config, hardware, raster, 'Trees')
    assert plan.estimated_vector_tiles == 9
    assert plan.vector_tile_workers >= 2
    assert (plan.vector_tile_workers * plan.vector_inner_workers
            <= plan.cpu_workers)
