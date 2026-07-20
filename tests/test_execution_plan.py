"""Unit tests for the planner's vector_mode decision (untiled vector phase).

Config.vector_processing ('auto' | 'tiled' | 'untiled') is the user-facing
knob; build_execution_plan resolves it to plan.vector_mode. 'auto' may pick
'untiled' only when the estimated stem-map working set clearly fits in RAM
(<= 50% of total_ram_gb at ~48 bytes/pixel of the PREDICTION grid) and must
fall back to 'tiled' when in doubt (unknown RAM, degenerate raster).
"""

from classes.Config import Config
from classes.ExecutionPlan import build_execution_plan
from classes.HardwareInfo import HardwareInfo

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


def test_config_default_knob_is_auto():
    assert Config().vector_processing == 'auto'


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


def test_auto_picks_untiled_for_full_ortho_with_big_ram():
    # 6.6 GB working set <= 50% of 64 GB -> clearly safe.
    plan = _plan(Config(), _hardware(64.0), FULL_ORTHO)
    assert plan.vector_mode == 'untiled'


def test_auto_falls_back_to_tiled_when_ram_is_small():
    # 6.6 GB working set > 50% of 8 GB -> tiled.
    plan = _plan(Config(), _hardware(8.0), FULL_ORTHO)
    assert plan.vector_mode == 'tiled'


def test_auto_falls_back_to_tiled_when_ram_is_unknown():
    plan = _plan(Config(), _hardware(0.0), SMALL)
    assert plan.vector_mode == 'tiled'


def test_auto_falls_back_to_tiled_for_degenerate_raster():
    plan = _plan(Config(), _hardware(64.0), _raster(0, 0))
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
