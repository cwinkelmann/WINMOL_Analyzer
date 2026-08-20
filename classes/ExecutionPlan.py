from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    prediction_batch_size: int
    producer_queue_batches: int
    producer_workers: int
    progress_interval_s: float
    gpu_workers: int
    cpu_workers: int
    vector_tile_workers: int
    vector_inner_workers: int
    keep_temp: bool
    #: Human-readable notes for every knob the planner reduced below what
    #: the config asked for. Empty when nothing was overridden.
    capped: list = field(default_factory=list)


CPU_ONLY = 'cpu_only'
SINGLE_GPU = 'gpu'
MULTI_GPU = 'multi_gpu_dgx'


def _cfg(config: Any, key: str, default: Any) -> Any:
    return getattr(config, key, default)


def _apply_caps(label: str, requested, caps, capped: list) -> int:
    """Reduce ``requested`` to the tightest cap, recording WHY it bit.

    Silent capping is how this pipeline hid two real limits: a configured
    ``prediction_producer_workers_gpu = 6`` was cut to 3 by
    ``cpu_workers // 3``, and the vector pool ran on 2 of 12 cores. In
    both cases the config value looked honoured and never was, so nobody
    thought to question it. Anything that lowers a user-set knob has to
    say so out loud.

    ``caps`` is an iterable of ``(value, reason)``; the note names the
    constraint that actually bound, not just the final number.
    """
    resolved = int(requested)
    binder = None
    for value, reason in caps:
        value = int(value)
        if value < resolved:
            resolved, binder = value, reason
    resolved = max(1, resolved)
    if resolved < int(requested):
        capped.append(
            f"{label}: config asked for {int(requested)}, using {resolved} "
            f"(limited by {binder or 'a floor of 1'})"
        )
    return resolved


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


def _scenario(hardware: Any) -> str:
    gpu_count = int(getattr(hardware, 'gpu_count', 0) or 0)
    if gpu_count <= 0:
        return CPU_ONLY
    if gpu_count == 1:
        return SINGLE_GPU
    return MULTI_GPU


#: Private resident bytes one vector tile worker needs. Measured at
#: ~1.38 GB (Private_Dirty, smaps_rollup, Tegel R13/R12); rounded up for
#: headroom since peak differs from the sampled instant.
VECTOR_WORKER_PRIVATE_BYTES = 1.75 * (1024 ** 3)

#: Sentinel for "this cap does not apply" inside the min() below.
_UNCONSTRAINED = 1 << 30


def _gpu_memory_gb(hardware: Any) -> float:
    values = list(getattr(hardware, 'gpu_memory_gb', []) or [])
    return max(values) if values else 0.0


def _vector_available_bytes(hardware: Any) -> float:
    """Free RAM in bytes, or 0.0 when it cannot be determined.

    psutil ships in requirements/cpu.txt and gpu.txt, so this degrades
    rather than raises when it is missing (e.g. a minimal CI image); the
    fallback assumes half of total RAM is usable.
    """
    try:
        # Clamped to the cgroup limit when one is set: a limited
        # container otherwise reads the HOST's free memory and
        # over-provisions the vector pool exactly as badly as reading
        # the host's total.
        from plugin_utils import container
        return float(container.available_memory_bytes())
    except Exception:
        total_gb = getattr(hardware, 'total_ram_gb', None)
        if total_gb:
            return float(total_gb) * (1024 ** 3) * 0.5
    return 0.0


def _vector_memory_cap(config: Any, hardware: Any) -> int:
    """How many vector tile workers free RAM can actually hold.

    Sized on PRIVATE resident memory, not RSS. Measured with
    smaps_rollup on a live Tegel vector phase, per worker:

        RSS 2734 MB | Shared_Dirty 1257 MB | Private_Dirty 1381 MB

    The Shared_Dirty part is copy-on-write state inherited from the
    parent at fork; it is shared with the parent and every sibling, so
    it costs physical RAM ONCE, not once per worker. Sizing off RSS
    triple-counts it and badly under-provisions the pool.
    """
    per_worker = float(_cfg(
        config, 'vector_worker_bytes', VECTOR_WORKER_PRIVATE_BYTES,
    ) or VECTOR_WORKER_PRIVATE_BYTES)
    if per_worker <= 0:
        return _UNCONSTRAINED

    total_gb = getattr(hardware, 'total_ram_gb', None)
    fraction = float(_cfg(config, 'vector_ram_fraction', 0.4) or 0.4)
    floor_bytes = (float(total_gb) * (1024 ** 3) * fraction
                   if total_gb else 0.0)

    # Deliberately the LARGER of "free right now" and a fraction of total.
    # The plan is computed at startup, so the instantaneous reading is
    # unrepresentative of the vector phase in BOTH directions: the model
    # and prediction buffers are not allocated yet (so free RAM reads
    # high), while anything else running on the box makes it read low --
    # a transient dip must not permanently pin the pool to one worker.
    # The fraction of total is the stable term; free RAM only ever raises
    # the budget above it.
    budget = max(_vector_available_bytes(hardware), floor_bytes)
    if budget <= 0:
        return _UNCONSTRAINED       # unknown -> let the CPU cap decide
    # Floor of one worker: refusing to run is worse than swapping.
    return max(1, int(budget // per_worker))


def _vector_worker_split(
    config: Any,
    hw_cpu: int,
    cpu_workers: int,
    tiles: int,
    process_type: str,
    huge_nodes_job: bool,
    hardware: Any = None,
    capped: list | None = None,
) -> tuple[int, int]:
    capped = [] if capped is None else capped

    if process_type == 'Stems':
        return 1, max(1, cpu_workers)

    if tiles < 8 or hw_cpu < 8:
        inner = cpu_workers if not huge_nodes_job else max(1, hw_cpu // 5)
        return 1, max(1, inner)

    max_tile_workers = max(
        1,
        int(_cfg(config, 'max_vector_tile_workers', 16)),
    )
    # Was `cpu_workers // 4`. That divisor reserved three quarters of the
    # budget for inner workers -- but this branch returns inner=1, so it
    # reserved cores for parallelism it never created. On a 12-core box
    # (cpu_workers=11) it yielded 2, and the vector phase, which is ~73%
    # of a large run, used 2 of 12 cores. Bound by cores and free RAM
    # instead; both are real constraints, the divisor was not.
    tile_workers = _apply_caps(
        'vector_tile_workers',
        max_tile_workers,
        [
            (max(1, cpu_workers), 'cpu_workers'),
            (_vector_memory_cap(config, hardware), 'free RAM / worker size'),
            (tiles, 'number of vector tiles'),
        ],
        capped,
    )
    if tile_workers <= 1:
        inner = cpu_workers if not huge_nodes_job else max(1, hw_cpu // 5)
        return 1, max(1, inner)

    return max(1, tile_workers), 1


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
    # Every knob the planner reduces below its configured value lands
    # here, so the run can print WHY instead of silently disagreeing
    # with config.json.
    capped: list = []
    keep_temp = bool(_cfg(config, 'keep_temp_tiles', False))
    max_gpu_workers = int(_cfg(config, 'max_gpu_workers', 8))
    hw_cpu = max(1, int(getattr(hardware, 'cpu_count', 1) or 1))
    max_cpu_workers = min(
        int(_cfg(config, 'max_cpu_workers', max(hw_cpu - 1, 1))),
        max(hw_cpu - 1, 1),
    )
    tiles = _estimate_prediction_tiles(config, raster)
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
        requested_producers = int(
            _cfg(config, 'prediction_producer_workers_gpu', 3)
        )
        producer_caps = [(max(1, cpu_workers // 3), 'cpu_workers // 3')]
        if cpu_workers < 10:
            producer_caps.append((2, 'cpu_workers < 10'))
        producer_workers = _apply_caps(
            'prediction_producer_workers_gpu',
            requested_producers,
            producer_caps,
            capped,
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
        tiles,
        process_type,
        huge_nodes_job,
        hardware,
        capped,
    )
    prediction_mode = _resolve_prediction_mode(config, scen)
    vector_mode = 'none' if process_type == 'Stems' else 'tiled'

    return ExecutionPlan(
        process_type=process_type,
        scenario=scen,
        prediction_mode=prediction_mode,
        vector_mode=vector_mode,
        tile_inner_px=tile_inner_px,
        tile_overlap_m=tile_overlap_m,
        halo_px=halo_px,
        estimated_prediction_tiles=tiles,
        prediction_batch_size=prediction_batch_size,
        producer_queue_batches=producer_queue_batches,
        producer_workers=producer_workers,
        progress_interval_s=progress_interval_s,
        gpu_workers=gpu_workers,
        cpu_workers=cpu_workers,
        vector_tile_workers=vector_tile_workers,
        vector_inner_workers=vector_inner_workers,
        keep_temp=keep_temp,
        capped=capped,
    )
