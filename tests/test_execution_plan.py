"""Planner unit tests: composed vector worker split + vector_mode knob.

Union of two independently developed test sets:

1. Scheduler tests — the vector-phase worker split must COMPOSE.
   The old _vector_worker_split returned either (N, 1) or (1, N) —
   tile-level and inner-stage parallelism were mutually exclusive — and
   gated the tile branch on cpu_workers // 4 and on the estimated
   PREDICTION tile count. Measured effect: vector wall time equalled the
   serial sum of per-tile totals. These tests pin the composed
   scheduling contract (which applies to the TILED vector mode).

2. vector_mode tests — Config.vector_processing ('auto' | 'tiled' |
   'untiled') is the user-facing knob; build_execution_plan resolves it
   to plan.vector_mode. 'auto' may pick 'untiled' only when the
   estimated stem-map working set clearly fits in RAM (<= 50% of
   total_ram_gb at ~48 bytes/pixel of the PREDICTION grid) and must
   fall back to 'tiled' when in doubt (unknown RAM, degenerate raster).
"""

from types import SimpleNamespace

from classes.Config import Config
from classes.ExecutionPlan import (
    _estimate_vector_tiles,
    _vector_worker_split,
    build_execution_plan,
)
from classes.HardwareInfo import HardwareInfo


# ---------------------------------------------------------------------------
# Composed tile x inner worker split (tiled vector mode)
# ---------------------------------------------------------------------------

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
    # Pin the tiled vector mode: the composed tile x inner split is a
    # tiled-mode contract. (In 'auto', this small raster with ample RAM
    # now legitimately resolves to 'untiled', where the split is 1 x N.)
    config.vector_processing = 'tiled'
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


# ---------------------------------------------------------------------------
# vector_mode resolution (Config.vector_processing knob)
# ---------------------------------------------------------------------------

# Prediction-grid pixel size is tile_size/img_width = 15/512 ~ 0.0293 m/px.
# Rasters below are described at that same resolution, so the stem map has
# (roughly) the same dimensions as the input raster.
PRED_PX = 15.0 / 512.0


def _hardware(ram_gb, cpus=8, gpus=0):
    return HardwareInfo(
        cpu_count=cpus,
        total_ram_gb=ram_gb,
        gpu_count=gpus,
        gpu_names=[f"gpu{i}" for i in range(gpus)],
        gpu_memory_gb=[16.0] * gpus,
    )


def _raster(width, height, px=PRED_PX, est_gb=None):
    if est_gb is None:
        est_gb = width * height / (1024.0 ** 3)
    return {
        'width': width,
        'height': height,
        'bands': 1,
        'dtype': 'uint8',
        'pixel_size_x': px,
        'pixel_size_y': px,
        'estimated_input_gb': est_gb,
    }


SMALL = _raster(1520, 1520)            # fixture-sized: ~2.3 Mpx -> ~0.1 GB
FULL_ORTHO = _raster(14624, 10088)     # ~147.5 Mpx -> ~6.6 GB working set


def _plan(config, hardware, raster, process_type='Trees'):
    return build_execution_plan(config, hardware, raster, process_type)


def test_config_default_knob_is_tiled():
    # Deliberate: default preserves the pre-existing tiled behavior (and
    # its published numbers); 'auto'/'untiled' are opt-in (docs/CONFIG.md).
    assert Config().vector_processing == 'tiled'


def test_stems_process_type_has_no_vector_mode():
    plan = _plan(Config(), _hardware(64.0), SMALL, process_type='Stems')
    assert plan.vector_mode == 'none'


def test_knob_tiled_forces_tiled_even_with_plenty_of_ram():
    config = Config()
    config.vector_processing = 'tiled'
    plan = _plan(config, _hardware(256.0), SMALL)
    assert plan.vector_mode == 'tiled'


def test_knob_untiled_forces_untiled_even_with_tiny_ram():
    config = Config()
    config.vector_processing = 'untiled'
    plan = _plan(config, _hardware(2.0), FULL_ORTHO)
    assert plan.vector_mode == 'untiled'


def test_auto_picks_untiled_for_small_raster_with_ample_ram():
    config = Config()
    config.vector_processing = 'auto'
    plan = _plan(config, _hardware(16.0), SMALL)
    assert plan.vector_mode == 'untiled'


def _auto_config():
    config = Config()
    config.vector_processing = 'auto'
    return config


def test_auto_picks_untiled_for_full_ortho_with_big_ram():
    # 6.6 GB working set <= 50% of 64 GB -> clearly safe.
    plan = _plan(_auto_config(), _hardware(64.0), FULL_ORTHO)
    assert plan.vector_mode == 'untiled'


def test_auto_falls_back_to_tiled_when_ram_is_small():
    # 6.6 GB working set > 50% of 8 GB -> tiled.
    plan = _plan(_auto_config(), _hardware(8.0), FULL_ORTHO)
    assert plan.vector_mode == 'tiled'


def test_auto_falls_back_to_tiled_when_ram_is_unknown():
    plan = _plan(_auto_config(), _hardware(0.0), SMALL)
    assert plan.vector_mode == 'tiled'


def test_auto_falls_back_to_tiled_for_degenerate_raster():
    plan = _plan(_auto_config(), _hardware(64.0), _raster(0, 0))
    assert plan.vector_mode == 'tiled'


def test_default_config_stays_tiled_even_with_ample_ram():
    plan = _plan(Config(), _hardware(64.0), FULL_ORTHO)
    assert plan.vector_mode == 'tiled'


def test_unrecognized_knob_value_behaves_like_auto():
    config = Config()
    config.vector_processing = 'banana'
    assert _plan(config, _hardware(16.0), SMALL).vector_mode == 'untiled'
    assert _plan(config, _hardware(0.0), SMALL).vector_mode == 'tiled'


def test_untiled_plan_never_fans_out_tile_workers():
    config = Config()
    config.vector_processing = 'untiled'
    plan = _plan(config, _hardware(64.0, cpus=32), FULL_ORTHO)
    assert plan.vector_tile_workers == 1
    assert plan.vector_inner_workers == plan.cpu_workers


def test_default_config_on_gpu_hardware_still_resolves_prediction_modes():
    # The new knob must not disturb prediction-mode resolution.
    plan = _plan(Config(), _hardware(64.0, gpus=1), FULL_ORTHO)
    assert plan.prediction_mode == 'stream'
    plan = _plan(Config(), _hardware(64.0, gpus=0), FULL_ORTHO)
    assert plan.prediction_mode == 'cpu_stream'


def test_stem_map_pixels_estimated_on_prediction_grid_not_input_grid():
    # A coarse 1 m/px input over a big extent yields a much larger
    # prediction grid: 3000x3000 px at 1 m -> ~102k x 102k stem-map px
    # (~500 GB working set) -> must be tiled even with lots of RAM.
    coarse = _raster(3000, 3000, px=1.0)
    plan = _plan(Config(), _hardware(64.0), coarse)
    assert plan.vector_mode == 'tiled'
