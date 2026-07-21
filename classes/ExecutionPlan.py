from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class RasterInfo:
    width: int
    height: int
    bands: int
    dtype: str
    pixel_size_x: float
    pixel_size_y: float
    estimated_input_gb: float

    @classmethod
    def from_any(cls, value: Any) -> "RasterInfo":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                width=int(value.get('width', 0)),
                height=int(value.get('height', 0)),
                bands=int(value.get('bands', 0)),
                dtype=str(value.get('dtype', 'unknown')),
                pixel_size_x=float(value.get('pixel_size_x', 0.0) or 0.0),
                pixel_size_y=float(value.get('pixel_size_y', 0.0) or 0.0),
                estimated_input_gb=float(
                    value.get('estimated_input_gb', 0.0) or 0.0
                ),
            )
        return cls(
            width=int(getattr(value, 'width', 0)),
            height=int(getattr(value, 'height', 0)),
            bands=int(getattr(value, 'bands', 0)),
            dtype=str(getattr(value, 'dtype', 'unknown')),
            pixel_size_x=float(getattr(value, 'pixel_size_x', 0.0) or 0.0),
            pixel_size_y=float(getattr(value, 'pixel_size_y', 0.0) or 0.0),
            estimated_input_gb=float(
                getattr(value, 'estimated_input_gb', 0.0) or 0.0
            ),
        )


@dataclass
class ExecutionPlan:
    process_type: str
    scenario: str
    prediction_mode: str
    vector_mode: str
    tile_inner_px: int
    tile_overlap_m: float
    halo_px: int
    estimated_prediction_tiles: int
    estimated_vector_tiles: int
    prediction_batch_size: int
    producer_queue_batches: int
    producer_workers: int
    progress_interval_s: float
    gpu_workers: int
    cpu_workers: int
    vector_tile_workers: int
    vector_inner_workers: int
    keep_temp: bool


CPU_ONLY = 'cpu_only'
SINGLE_GPU = 'gpu'
MULTI_GPU = 'multi_gpu_dgx'

# Untiled vector phase RAM heuristic (Config.vector_processing = 'auto').
# The legacy whole-raster chain holds the uint8 stem map plus skeletonize /
# labeling / EDT working copies at once — roughly uint8 pred (1) + bool
# skeleton (1) + int32 labels (4) + float64 EDT (8) + float64 temporaries
# (8) ≈ 22 bytes/pixel; doubled for safety margin and rounded up.
UNTILED_BYTES_PER_PIXEL = 48.0
# Never plan more than half of physical RAM for the untiled vector phase.
UNTILED_RAM_BUDGET_FRACTION = 0.5


def _cfg(config: Any, key: str, default: Any) -> Any:
    return getattr(config, key, default)


def _meters_to_pixels(
    tile_overlap_m: float,
    pixel_size_x: float,
    pixel_size_y: float,
) -> int:
    px = max(abs(pixel_size_x) or 0.0, abs(pixel_size_y) or 0.0)
    if px <= 0.0:
        return 0
    return int(math.ceil(tile_overlap_m / px))


def _estimate_prediction_tiles(config: Any, raster: RasterInfo) -> int:
    if raster.width <= 0 or raster.height <= 0:
        return 0
    px_x = abs(raster.pixel_size_x) or 0.0
    px_y = abs(raster.pixel_size_y) or 0.0
    if px_x <= 0.0 or px_y <= 0.0:
        return 0
    tile_size = float(_cfg(config, 'tile_size', 15.0))
    px_per_tile_x = int(math.ceil(tile_size / px_x))
    px_per_tile_y = int(math.ceil(tile_size / px_y))
    overlap_pred = int(_cfg(config, 'overlap_pred', 8))
    img_width = int(_cfg(config, 'img_width', 512))
    overlap_img_x = overlap_pred * px_per_tile_x / max(img_width, 1)
    overlap_img_y = overlap_pred * px_per_tile_y / max(img_width, 1)
    step_x = max(px_per_tile_x - overlap_img_x, 1.0)
    step_y = max(px_per_tile_y - overlap_img_y, 1.0)
    x_tiles = int(math.ceil(raster.width / step_x))
    y_tiles = int(math.ceil(raster.height / step_y))
    return max(1, x_tiles * y_tiles)


def _estimate_vector_tiles(config: Any, raster: RasterInfo) -> int:
    """Estimate the VECTOR-stage tile count (tile_inner_px grid).

    The vector grid is laid over the PREDICTED stem map, whose pixel size is
    tile_size / img_width meters (the prediction resamples to the model's
    native resolution), not over the input orthomosaic's grid. The previous
    scheduler gated tile parallelism on the estimated PREDICTION tile count
    (~512 px grid, hundreds of tiles), which made the "few tiles" gate
    meaningless: a raster with 2 vector tiles and 600 prediction tiles was
    scheduled as if it had hundreds of independent vector work items.
    """
    if raster.width <= 0 or raster.height <= 0:
        return 0
    tile_inner_px = max(1, int(_cfg(config, 'tile_inner_px', 4096)))
    px_x = abs(raster.pixel_size_x) or 0.0
    px_y = abs(raster.pixel_size_y) or 0.0
    tile_size = float(_cfg(config, 'tile_size', 15.0))
    img_width = int(_cfg(config, 'img_width', 512))
    pred_px = tile_size / max(img_width, 1)
    if px_x > 0.0 and px_y > 0.0 and pred_px > 0.0:
        est_width = raster.width * px_x / pred_px
        est_height = raster.height * px_y / pred_px
    else:
        est_width = float(raster.width)
        est_height = float(raster.height)
    x_tiles = int(math.ceil(est_width / tile_inner_px))
    y_tiles = int(math.ceil(est_height / tile_inner_px))
    return max(1, x_tiles * y_tiles)


def _scenario(hardware: Any) -> str:
    gpu_count = int(getattr(hardware, 'gpu_count', 0) or 0)
    if gpu_count <= 0:
        return CPU_ONLY
    if gpu_count == 1:
        return SINGLE_GPU
    return MULTI_GPU


def _gpu_memory_gb(hardware: Any) -> float:
    values = list(getattr(hardware, 'gpu_memory_gb', []) or [])
    return max(values) if values else 0.0


def _vector_worker_split(
    config: Any,
    hw_cpu: int,
    cpu_workers: int,
    vector_tiles: int,
    process_type: str,
    huge_nodes_job: bool,
) -> tuple[int, int]:
    """Split the CPU budget into tile-level x inner-stage workers.

    Tile workers and inner workers COMPOSE (tile_workers * inner_workers
    <= cpu_workers) instead of excluding each other: measured vector wall
    time equalled the serial sum of per-tile totals because the old split
    returned either (N, 1) or (1, N), and the (N, 1) branch was additionally
    gated behind cpu_workers // 4, which is <= 1 on most workstations.

    ``vector_tiles`` is the estimated VECTOR tile count (see
    _estimate_vector_tiles), not the prediction tile count the old code
    received.
    """
    if process_type == 'Stems':
        return 1, max(1, cpu_workers)

    max_tile_workers = max(
        1,
        int(_cfg(config, 'max_vector_tile_workers', 4)),
    )

    if huge_nodes_job:
        # Memory guard (unchanged): dense multi-tile jobs keep the old
        # conservative split so at most max_tile_workers tiles are resident
        # and inner pools stay trimmed.
        if vector_tiles < 2 or hw_cpu < 8:
            return 1, max(1, hw_cpu // 5)
        tile_workers = min(
            max_tile_workers, max(1, cpu_workers // 4), vector_tiles)
        if tile_workers <= 1:
            return 1, max(1, hw_cpu // 5)
        return tile_workers, 1

    if vector_tiles < 2 or hw_cpu < 8:
        return 1, max(1, cpu_workers)

    tile_workers = min(max_tile_workers, cpu_workers, vector_tiles)
    if tile_workers <= 1:
        return 1, max(1, cpu_workers)
    return tile_workers, max(1, cpu_workers // tile_workers)


def _estimate_stem_map_pixels(config: Any, raster: RasterInfo) -> int:
    """Estimate the pixel count of the PREDICTION-GRID stem map.

    The vector phase consumes the resampled stem map whose pixel size is
    tile_size / img_width (m/px), not the input orthomosaic grid. When the
    input has no usable georeferencing, fall back to the input grid size.
    """
    if raster.width <= 0 or raster.height <= 0:
        return 0
    px_x = abs(raster.pixel_size_x) or 0.0
    px_y = abs(raster.pixel_size_y) or 0.0
    tile_size = float(_cfg(config, 'tile_size', 15.0))
    img_width = int(_cfg(config, 'img_width', 512))
    pred_px = tile_size / max(img_width, 1)
    if px_x <= 0.0 or px_y <= 0.0 or pred_px <= 0.0:
        return raster.width * raster.height
    stem_w = int(math.ceil(raster.width * px_x / pred_px))
    stem_h = int(math.ceil(raster.height * px_y / pred_px))
    return stem_w * stem_h


def _untiled_fits_in_ram(config: Any, hardware: Any,
                         raster: RasterInfo) -> bool:
    """True when the whole-raster vector chain clearly fits in RAM.

    Conservative on purpose: unknown RAM or a degenerate raster means False,
    so 'auto' keeps the current tiled behavior when in doubt.
    """
    total_ram_gb = float(getattr(hardware, 'total_ram_gb', 0.0) or 0.0)
    if total_ram_gb <= 0.0:
        return False
    pixels = _estimate_stem_map_pixels(config, raster)
    if pixels <= 0:
        return False
    needed_gb = pixels * UNTILED_BYTES_PER_PIXEL / (1024.0 ** 3)
    return needed_gb <= total_ram_gb * UNTILED_RAM_BUDGET_FRACTION


def _resolve_vector_mode(config: Any, hardware: Any, raster: RasterInfo,
                         process_type: str) -> str:
    if process_type == 'Stems':
        return 'none'
    pref = str(_cfg(config, 'vector_processing', 'auto')).lower()
    if pref == 'untiled':
        return 'untiled'
    if pref == 'tiled':
        return 'tiled'
    # 'auto' (and any unrecognized value): untiled only when clearly safe.
    if _untiled_fits_in_ram(config, hardware, raster):
        return 'untiled'
    return 'tiled'


def _resolve_prediction_mode(config: Any, scen: str) -> str:
    pref = str(_cfg(config, 'prediction_backend', 'auto')).lower()
    if pref == 'cpu':
        return 'cpu_stream'
    if pref == 'single_gpu':
        return 'stream' if scen != CPU_ONLY else 'cpu_stream'
    if pref == 'multi_gpu':
        if scen == MULTI_GPU:
            return 'multi_gpu_stream'
        return 'stream' if scen == SINGLE_GPU else 'cpu_stream'
    if scen == CPU_ONLY:
        return 'cpu_stream'
    if scen == SINGLE_GPU:
        return 'stream'
    return 'multi_gpu_stream'


def build_execution_plan(
    config: Any,
    hardware: Any,
    raster_info: Any,
    process_type: str,
) -> ExecutionPlan:
    raster = RasterInfo.from_any(raster_info)
    tile_inner_px = int(_cfg(config, 'tile_inner_px', 4096))
    tile_overlap_m = float(_cfg(config, 'tile_overlap_m', 12.0))
    halo_px = _meters_to_pixels(
        tile_overlap_m,
        raster.pixel_size_x,
        raster.pixel_size_y,
    )
    keep_temp = bool(_cfg(config, 'keep_temp_tiles', False))
    max_gpu_workers = int(_cfg(config, 'max_gpu_workers', 8))
    hw_cpu = max(1, int(getattr(hardware, 'cpu_count', 1) or 1))
    max_cpu_workers = min(
        int(_cfg(config, 'max_cpu_workers', max(hw_cpu - 1, 1))),
        max(hw_cpu - 1, 1),
    )
    tiles = _estimate_prediction_tiles(config, raster)
    vector_tiles = _estimate_vector_tiles(config, raster)
    scen = _scenario(hardware)
    gpu_mem_gb = _gpu_memory_gb(hardware)
    large_prediction_job = raster.estimated_input_gb >= 4.0 or tiles >= 800
    huge_nodes_job = (
        process_type in {'Trees', 'Nodes'}
        and (tiles >= 1200 or large_prediction_job)
    )

    if scen == CPU_ONLY:
        cpu_workers = max(
            1,
            min(max_cpu_workers, int(round(max_cpu_workers * 0.75)) or 1),
        )
        gpu_workers = 0
        prediction_batch_size = max(
            1,
            int(_cfg(config, 'prediction_batch_cpu', 1)),
        )
        producer_queue_batches = max(
            2,
            int(_cfg(config, 'producer_queue_batches', 4)),
        )
        producer_workers = max(
            1,
            int(_cfg(config, 'prediction_producer_workers_cpu', 1)),
        )
        progress_interval_s = float(
            _cfg(config, 'progress_interval_s_cpu', 45.0)
        )
    elif scen == SINGLE_GPU:
        cpu_workers = max(
            1,
            min(
                max_cpu_workers,
                int(
                    _cfg(
                        config,
                        'single_gpu_cpu_workers',
                        min(12, max(8, hw_cpu // 2)),
                    )
                ),
            ),
        )
        gpu_workers = 1
        if gpu_mem_gb >= 20:
            default_batch = 8 if huge_nodes_job or large_prediction_job else 4
        elif gpu_mem_gb >= 12:
            default_batch = 4
        else:
            default_batch = 2
        prediction_batch_size = max(
            1,
            min(
                int(_cfg(config, 'prediction_batch_max_gpu', 12)),
                int(_cfg(config, 'prediction_batch_gpu', default_batch)),
            ),
        )
        producer_queue_batches = max(
            4,
            int(_cfg(config, 'producer_queue_batches', 8)),
        )
        if huge_nodes_job:
            producer_queue_batches = max(producer_queue_batches, 8)
        producer_workers = int(
            _cfg(config, 'prediction_producer_workers_gpu', 3)
        )
        if cpu_workers < 10:
            producer_workers = min(producer_workers, 2)
        producer_workers = max(
            1,
            min(producer_workers, max(1, cpu_workers // 3)),
        )
        progress_interval_s = float(
            _cfg(config, 'progress_interval_s_gpu', 60.0)
        )
    else:
        gpu_available = int(getattr(hardware, 'gpu_count', 0) or 0)
        gpu_workers = max(1, min(max_gpu_workers, gpu_available))
        if tiles < 1000:
            gpu_workers = min(gpu_workers, 2)
        elif tiles < 2500:
            gpu_workers = min(gpu_workers, 4)
        else:
            gpu_workers = min(gpu_workers, 8)
        cpu_workers = max(
            1,
            min(
                max_cpu_workers,
                int(
                    _cfg(
                        config,
                        'multi_gpu_cpu_workers',
                        max(24, int(max_cpu_workers * 0.75)),
                    )
                ),
            ),
        )
        if gpu_mem_gb >= 70:
            default_batch = 8
        elif gpu_mem_gb >= 40:
            default_batch = 6
        elif gpu_mem_gb >= 20:
            default_batch = 4
        else:
            default_batch = 3
        multi_gpu_batch = _cfg(config, 'prediction_batch_multi_gpu', None)
        if multi_gpu_batch is None:
            multi_gpu_batch = default_batch

        prediction_batch_size = max(
            1,
            min(
                int(_cfg(config, 'prediction_batch_max_gpu', 12)),
                int(multi_gpu_batch),
            ),
        )
        producer_queue_batches = max(
            4,
            int(_cfg(config, 'producer_queue_batches', 8)),
        )
        producer_workers = max(
            1,
            int(_cfg(config, 'prediction_producer_workers_multi_gpu', 2)),
        )
        progress_interval_s = float(
            _cfg(config, 'progress_interval_s_multi_gpu', 20.0)
        )

    vector_tile_workers, vector_inner_workers = _vector_worker_split(
        config,
        hw_cpu,
        cpu_workers,
        vector_tiles,
        process_type,
        huge_nodes_job,
    )
    prediction_mode = _resolve_prediction_mode(config, scen)
    vector_mode = _resolve_vector_mode(config, hardware, raster, process_type)
    if vector_mode == 'untiled':
        # One whole-raster job: all vector parallelism goes to the inner
        # stages (quantification workers), never to tile fan-out.
        vector_tile_workers = 1
        vector_inner_workers = max(1, cpu_workers)

    return ExecutionPlan(
        process_type=process_type,
        scenario=scen,
        prediction_mode=prediction_mode,
        vector_mode=vector_mode,
        tile_inner_px=tile_inner_px,
        tile_overlap_m=tile_overlap_m,
        halo_px=halo_px,
        estimated_prediction_tiles=tiles,
        estimated_vector_tiles=vector_tiles,
        prediction_batch_size=prediction_batch_size,
        producer_queue_batches=producer_queue_batches,
        producer_workers=producer_workers,
        progress_interval_s=progress_interval_s,
        gpu_workers=gpu_workers,
        cpu_workers=cpu_workers,
        vector_tile_workers=vector_tile_workers,
        vector_inner_workers=vector_inner_workers,
        keep_temp=keep_temp,
    )
