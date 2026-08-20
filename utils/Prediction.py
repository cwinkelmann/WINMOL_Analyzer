#!/usr/bin/env python

################################################################################
"""Imports"""
import os
import queue
import subprocess
import threading
import time
import contextlib
import sys

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window
from skimage.transform import resize

from classes.Timer import Timer
from utils import IO


@contextlib.contextmanager
def _suppress_native_stderr(enabled=True):
    if not enabled:
        yield
        return

    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    saved_fd = os.dup(fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(saved_fd)


################################################################################
"""Prediction of the semantic stem map with U-Net"""


def _to_float32_image(arr):
    if arr.dtype == np.float32:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return (arr / 255.0).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _resampling_layout(shape, profile, config):
    height, width = int(shape[0]), int(shape[1])
    px_per_tile_x = int(np.ceil(config.tile_size /
                                abs(profile['transform'][0])))
    px_per_tile_y = int(np.ceil(config.tile_size /
                                abs(profile['transform'][4])))
    overlap_img_x = config.overlap_pred * px_per_tile_x / config.img_width
    overlap_img_y = config.overlap_pred * px_per_tile_y / config.img_width
    x_tiles = int(np.ceil(width / max(px_per_tile_x - overlap_img_x, 1)))
    y_tiles = int(np.ceil(height / max(px_per_tile_y - overlap_img_y, 1)))
    img_width_inner = config.img_width - config.overlap_pred
    out_width = int(x_tiles * img_width_inner + config.overlap_pred)
    out_height = int(y_tiles * img_width_inner + config.overlap_pred)
    out_transform = Affine(
        profile['transform'][0] * px_per_tile_x / config.img_width, 0.0,
        profile['transform'][2], 0.0,
        profile['transform'][4] * px_per_tile_y / config.img_width,
        profile['transform'][5]
    )
    return {
        'px_per_tile_x': px_per_tile_x,
        'px_per_tile_y': px_per_tile_y,
        'overlap_img_x': overlap_img_x,
        'overlap_img_y': overlap_img_y,
        'x_tiles': x_tiles,
        'y_tiles': y_tiles,
        'img_width_inner': img_width_inner,
        'out_width': out_width,
        'out_height': out_height,
        'out_transform': out_transform,
    }


def _iter_tile_jobs(layout, config):
    core = layout['img_width_inner']
    src_width = max(1, layout['px_per_tile_x'] - 1)
    src_height = max(1, layout['px_per_tile_y'] - 1)
    tile_index = 0
    for i in range(layout['y_tiles']):
        src_row = int(np.floor(i * (layout['px_per_tile_y']
                                    - layout['overlap_img_y'])))
        for j in range(layout['x_tiles']):
            src_col = int(np.floor(j * (layout['px_per_tile_x']
                                        - layout['overlap_img_x'])))
            dst_row = config.overlap_pred // 2 + i * core
            dst_col = config.overlap_pred // 2 + j * core
            yield {
                'tile_index': tile_index,
                'src_row': src_row,
                'src_col': src_col,
                'src_width': src_width,
                'src_height': src_height,
                'dst_row': dst_row,
                'dst_col': dst_col,
            }
            tile_index += 1


def _raw_tile_to_batchable(tile_img):
    tile_img = _to_float32_image(tile_img)
    if tile_img.ndim == 2:
        tile_img = tile_img[:, :, None]
    if tile_img.shape[2] < 3:
        pad = np.zeros(
            (tile_img.shape[0], tile_img.shape[1], 3 - tile_img.shape[2]),
            dtype=np.float32)
        tile_img = np.concatenate([tile_img, pad], axis=2)
    return tile_img[:, :, :3]


def _default_valid_mask(tile_img):
    tile_img = _raw_tile_to_batchable(tile_img)
    return np.any(tile_img != 0, axis=2)


def _resize_batch(batch_nhwc, size, order):
    """Resize an NHWC float32 batch to (H, W). order=3 ~ bicubic (imagery),
    order=0 = nearest (masks). Pure skimage/numpy -- no TensorFlow.

    Fast path: when the batch is already at the target size (e.g. tiles were
    resampled during the GDAL read in stream mode), this is a no-op -- so the
    per-tile CPU resize disappears entirely."""
    n, h, w, c = batch_nhwc.shape
    if (h, w) == (int(size[0]), int(size[1])):
        return np.ascontiguousarray(batch_nhwc, dtype=np.float32)
    out = np.empty((n, size[0], size[1], c), dtype=np.float32)
    for i in range(n):
        out[i] = resize(
            batch_nhwc[i], (size[0], size[1]),
            order=order, mode="edge",
            anti_aliasing=False, preserve_range=True,
        ).astype(np.float32)
    return out


def _prepare_inference_batch(raw_tiles, raw_masks, config):
    batch = np.stack([_raw_tile_to_batchable(t) for t in raw_tiles], axis=0)
    size = (config.img_height, config.img_width)
    tile_batch = _resize_batch(batch, size, order=3)

    if raw_masks is None:
        raw_masks = [_default_valid_mask(t) for t in raw_tiles]

    mask_batch = np.stack(
        [m.astype(np.float32)[:, :, None] for m in raw_masks],
        axis=0,
    )
    mask_resized = _resize_batch(mask_batch, size, order=0)
    return tile_batch, mask_resized


def _binarize_prediction_core(pred_core, mask_core, threshold: float = 0.5):
    return np.ascontiguousarray(
        ((pred_core >= threshold) & mask_core).astype(np.uint8)
    )


def _predict_batch_core(raw_tiles, raw_masks, model, config):
    tile_tensor, mask_resized = _prepare_inference_batch(
        raw_tiles, raw_masks, config)
    pred = np.asarray(model.predict_on_batch(tile_tensor))
    crop = config.overlap_pred // 2
    threshold = float(getattr(config, 'stem_binary_threshold', 0.5))
    pred_cores = []
    for idx in range(pred.shape[0]):
        pred_core = pred[idx, crop:(
            config.img_width - crop), crop:(config.img_width - crop), 0]
        mask_core = mask_resized[idx, crop:(
            config.img_width - crop), crop:(config.img_width - crop), 0] > 0.5
        pred_cores.append(_binarize_prediction_core(
            pred_core, mask_core, threshold=threshold))
    return pred_cores


def predict_tile_array(tile_img, model, config, tile_mask=None):
    return _predict_batch_core([tile_img], [tile_mask]
                               if tile_mask is not None
                               else None, model, config)[0]


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return 'unknown'
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _prediction_batch_candidates(config, initial_batch: int) -> list[int]:
    initial = max(1, int(initial_batch))
    max_batch_attr = getattr(config, 'prediction_batch_max_gpu', initial)
    max_batch = max(
        initial,
        int(max_batch_attr if max_batch_attr is not None else initial),
    )
    return list(range(initial, max_batch + 1))


#: How far above the initial batch the sweep may go when free memory could
#: not be determined at all. Deliberately tiny: an unbounded sweep on an
#: unknown machine is what can take a box down -- host RAM exhaustion
#: raises nothing at all, it just swaps and dies.
AUTOTUNE_BLIND_HEADROOM = 2

_GB = float(1024 ** 3)


def _batch_override(config):
    """The user's manual pin as a positive int, or None.

    ``Config.prediction_batch_size`` cannot serve this purpose: the
    planner (classes/ExecutionPlan.py) overwrites it on every run.
    """
    raw = getattr(config, 'prediction_batch_override', None)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _available_ram_bytes():
    """Free host RAM in bytes, or None when psutil is unavailable.

    psutil ships in requirements/cpu.txt and requirements/gpu.txt, so this
    must degrade rather than raise when it is missing (e.g. a minimal CI
    image).
    """
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:
        return None


def _free_gpu_memory_gb():
    """Free VRAM per visible GPU in GiB via ``nvidia-smi``, or ``[]`` when
    it is unavailable or fails. Never raises."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            # A driver stuck in an uninterruptible ioctl used to hang this
            # call forever (rr NVIDIA_SMI_TIMEOUT); the except catches
            # TimeoutExpired and falls back to the host-RAM bound.
            timeout=8.0,
        )
        if result.returncode != 0:
            return []
        values = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line) / 1024.0)
            except ValueError:
                continue
        return values
    except Exception:
        return []


def _free_memory_bytes(model, config):
    """``(bytes, source)`` describing the memory the sweep may spend from.

    Returns ``(None, reason)`` when it cannot be determined; the caller
    then falls back to :data:`AUTOTUNE_BLIND_HEADROOM`.
    """
    accelerator = str(getattr(model, 'accelerator', '') or '').lower()

    if accelerator == 'cuda':
        try:
            free = _free_gpu_memory_gb()
        except Exception:                               # pragma: no cover
            free = []
        usable = [f for f in free if f and f > 0]
        if usable:
            # The smallest visible device bounds the run: the same batch
            # size is used on all of them.
            gpu_free = min(usable) * _GB
            host = _available_ram_bytes()
            if host is not None and 0 < host < gpu_free:
                # Bound by BOTH. A run planned for CUDA whose session
                # silently fell back to the CPU provider allocates on the
                # host -- plenty of VRAM free, none in use, and the arena
                # eating RAM instead.
                return host, 'psutil available RAM (below free VRAM)'
            return gpu_free, 'nvidia-smi memory.free'
        return None, 'nvidia-smi did not report free GPU memory'

    available = _available_ram_bytes()
    if available is None or available <= 0:
        return None, 'psutil unavailable'
    return available, 'psutil available RAM'


def _estimated_bytes_per_tile(config) -> float:
    """Device memory one tile costs, activations included."""
    height = int(getattr(config, 'img_height', 512) or 512)
    width = int(getattr(config, 'img_width', 512) or 512)
    channels = int(getattr(config, 'n_channels', 3) or 3)
    classes = int(getattr(config, 'num_classes', 1) or 1)
    raw = float(height * width * (channels + classes) * 4)
    factor = float(getattr(
        config, 'prediction_batch_autotune_activation_factor', 32) or 32)
    return max(1.0, raw * max(1.0, factor))


def _memory_batch_ceiling(model, config, initial_batch: int) -> dict:
    """The largest batch the sweep may TRY, decided before anything is
    timed.

    This is the safety cap the whole feature hangs on: a GPU OOM is caught
    and halved (``_predict_batch_adaptive``), but host RAM exhaustion
    raises nothing at all -- the box swaps and dies. Only a pre-emptive
    ceiling helps. Unlike the configured caps (``prediction_batch_max_gpu``
    and friends), this one can also pull the ACTUAL batch used below
    ``initial_batch``: the planner's batch is a considered floor under
    normal conditions, but not something to trust blindly on a box that is
    already nearly out of memory.
    """
    initial = max(1, int(initial_batch))
    free, source = _free_memory_bytes(model, config)
    per_tile = _estimated_bytes_per_tile(config)

    if not free or free <= 0:
        return {
            'ceiling': initial + AUTOTUNE_BLIND_HEADROOM,
            'free_bytes': None,
            'source': source,
            'fraction': None,
            'bytes_per_tile': per_tile,
            'blind': True,
        }

    fraction = float(getattr(
        config, 'prediction_batch_autotune_memory_fraction', 0.6) or 0.6)
    fraction = min(0.95, max(0.05, fraction))
    ceiling = int((free * fraction) // per_tile)
    return {
        'ceiling': max(1, ceiling),
        'free_bytes': free,
        'source': source,
        'fraction': fraction,
        'bytes_per_tile': per_tile,
        'blind': False,
    }


def _describe_memory_budget(budget: dict) -> str:
    """Where the memory ceiling came from, without the label."""
    if budget.get('blind'):
        return (f"free memory unknown ({budget['source']}), "
                f"blind headroom +{AUTOTUNE_BLIND_HEADROOM}")
    return (
        f"{budget['free_bytes'] / _GB:.1f} GB free per {budget['source']}, "
        f"{budget['fraction'] * 100:.0f}% budget, "
        f"~{budget['bytes_per_tile'] / (1024 ** 2):.0f} MB/tile"
    )


class TileBatchProducer(threading.Thread):
    def __init__(self, uav_path, chunk_size, jobs, n_channels,
                 out_queue, producer_id=0, out_size=None):
        super().__init__(daemon=True)
        self.uav_path = uav_path
        self.chunk_size = max(1, int(chunk_size))
        self.jobs = jobs
        self.n_channels = n_channels
        self.out_queue = out_queue
        self.producer_id = producer_id
        # (H, W) to resample each tile to *during* the GDAL read (fast, in
        # C, and able to use overviews). None keeps the native-resolution
        # read, leaving resizing to the (slow) skimage path downstream.
        self.out_size = tuple(out_size) if out_size else None
        self.error = None

    def run(self):
        try:
            batch_items = []
            batch_read_s = 0.0
            with rasterio.open(self.uav_path) as src:
                indexes = list(range(1, min(self.n_channels, src.count) + 1))
                for job in self.jobs:
                    t0 = time.perf_counter()
                    window = Window(job['src_col'], job['src_row'],
                                    job['src_width'], job['src_height'])
                    # Resample onto the model grid during the read when
                    # out_size is set: GDAL does it in C (cubic for
                    # imagery -- bilinear measurably thins the predicted
                    # mask at full-ortho scale -- nearest for the validity
                    # mask), replacing the slow per-tile skimage resize in
                    # the consumer (_resize_batch's identity fast path
                    # then short-circuits). boundless+fill_value=0 keeps
                    # the requested (oh, ow) shape even when the window
                    # runs past the raster edge.
                    if self.out_size is not None:
                        oh, ow = self.out_size
                        tile = src.read(
                            indexes,
                            window=window,
                            out_shape=(len(indexes), oh, ow),
                            resampling=Resampling.cubic,
                            boundless=True,
                            fill_value=0,
                        ).transpose(1, 2, 0)
                        gdal_mask = src.read_masks(
                            1,
                            window=window,
                            out_shape=(oh, ow),
                            resampling=Resampling.nearest,
                            boundless=True,
                        ) > 0
                    else:
                        tile = src.read(
                            indexes,
                            window=window,
                            boundless=True,
                            fill_value=0,
                        ).transpose(1, 2, 0)
                        gdal_mask = src.read_masks(
                            1,
                            window=window,
                            boundless=True,
                        ) > 0

                    pixel_mask = np.any(tile != 0, axis=2)

                    # If GDAL mask is effectively all valid, it is not helping.
                    # Fall back to pixel-based validity for
                    # black background suppression.
                    if np.all(gdal_mask):
                        valid_mask = pixel_mask
                    else:
                        valid_mask = gdal_mask & pixel_mask

                    batch_read_s += time.perf_counter() - t0
                    batch_items.append((job, tile, valid_mask))
                    if len(batch_items) >= self.chunk_size:
                        self.out_queue.put({'items': batch_items,
                                            'read_s': batch_read_s,
                                            'producer_id': self.producer_id})
                        batch_items = []
                        batch_read_s = 0.0
                if batch_items:
                    self.out_queue.put({'items': batch_items,
                                        'read_s': batch_read_s,
                                        'producer_id': self.producer_id})
        except Exception as exc:  # pragma: no cover
            self.error = exc
        finally:
            self.out_queue.put(
                {'producer_done': True, 'producer_id': self.producer_id})


def _write_prediction_core(dst, pred_core, job, layout):
    out_row = int(job['dst_row'])
    out_col = int(job['dst_col'])
    write_h = min(pred_core.shape[0], layout['out_height'] - out_row)
    write_w = min(pred_core.shape[1], layout['out_width'] - out_col)
    if write_h <= 0 or write_w <= 0:
        return 0.0
    pred_write = np.ascontiguousarray(
        pred_core[:write_h, :write_w], dtype=np.uint8)
    out_window = Window(
        col_off=out_col, row_off=out_row, width=write_w, height=write_h)
    t0 = time.perf_counter()
    dst.write(pred_write, 1, window=out_window)
    return time.perf_counter() - t0


def _predict_batch_adaptive(
    raw_tiles, raw_masks, model, config, batch_size
):
    try:
        return _predict_batch_core(
            raw_tiles, raw_masks, model, config), batch_size
    except (RuntimeError, MemoryError) as exc:
        msg = str(exc).lower()
        # onnx_runtime normalizes an onnxruntime OOM to MemoryError; this
        # string check is the fallback for a raw RuntimeError. Match the CUDA
        # BFC-arena wording too ("Failed to allocate memory for requested
        # buffer ..."), which carries neither 'oom' nor 'out of memory' and so
        # slipped past the back-off before, aborting the run (issue #40).
        is_oom = isinstance(exc, MemoryError) or (
            'oom' in msg or 'out of memory' in msg
            or 'failed to allocate memory' in msg)
        if batch_size <= 1 or not is_oom:
            raise
        reduced = max(1, batch_size // 2)
        print(f"Prediction batch too large; reducing micro-batch size from "
              f"{batch_size} to {reduced}", flush=True)
        return _predict_batch_adaptive(raw_tiles[:reduced], raw_masks[:reduced]
                                       if raw_masks is not None
                                       else None, model, config, reduced)


def _time_batch_candidate(
    sample_tiles,
    sample_masks,
    model,
    config,
    candidate_batch: int,
    repeats: int = 2,
):
    repeats = max(1, int(repeats))
    cand = max(1, int(candidate_batch))

    tiles = sample_tiles[:cand]
    masks = sample_masks[:cand] if sample_masks is not None else None

    # Warm this exact candidate once so graph/kernel setup is not charged
    # to the measured run.
    # _, warm_used = _predict_batch_adaptive(
    #     tiles,
    #     masks,
    #     model,
    #     config,
    #     cand,
    # )
    quiet = bool(getattr(config, "prediction_batch_autotune_quiet", True))

    with _suppress_native_stderr(quiet):
        _, warm_used = _predict_batch_adaptive(
            tiles,
            masks,
            model,
            config,
            cand,
        )
    oomed = warm_used < cand

    measure_batch = warm_used
    timings = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        with _suppress_native_stderr(quiet):
            _, used = _predict_batch_adaptive(
                sample_tiles[:measure_batch],
                sample_masks[
                    :measure_batch] if sample_masks is not None else None,
                model,
                config,
                measure_batch,
            )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed / max(used, 1))
        measure_batch = used

    per_tile = float(np.median(timings))
    return warm_used, per_tile, oomed


def _autotune_batch_size(
    sample_tiles,
    sample_masks,
    model,
    config,
    initial_batch,
    label='Prediction micro-batch',
):
    initial = max(1, int(initial_batch))

    # A manual pin beats everything: no probing, no timing, no memory
    # check.
    override = _batch_override(config)
    if override is not None:
        print(
            f"{label} autotune: bound by the user pin "
            f"(Config.prediction_batch_override); skipped, batch pinned "
            f"to b{override} by the user.",
            flush=True,
        )
        return override

    autotune = bool(getattr(config, 'prediction_batch_autotune', True))
    if not autotune:
        return initial
    if len(sample_tiles) < 2:
        return initial

    patience = max(
        1,
        int(getattr(config, 'prediction_batch_autotune_patience', 2)),
    )
    min_improve = max(
        0.0,
        float(getattr(config, 'prediction_batch_autotune_min_improve', 0.02)),
    )
    min_improve_s = max(
        0.0,
        float(getattr(
            config, 'prediction_batch_autotune_min_improve_s', 0.2)),
    )
    stop_on_oom = bool(getattr(
        config, 'prediction_batch_autotune_stop_on_oom', True,
    ))
    repeats = max(
        1,
        int(getattr(config, 'prediction_batch_autotune_repeats', 2)),
    )

    # Bound the sweep by FREE memory BEFORE timing anything: the candidate
    # list below is derived from this ceiling, so a candidate past it is
    # never even attempted.
    budget = _memory_batch_ceiling(model, config, initial)
    ceiling = max(1, int(budget['ceiling']))

    if ceiling < initial:
        print(
            f"{label} autotune: memory ceiling b{ceiling} is below the "
            f"planned batch b{initial} ({_describe_memory_budget(budget)}); "
            f"skipping the sweep, using b{ceiling}.",
            flush=True,
        )
        return ceiling

    candidates = [
        c for c in _prediction_batch_candidates(config, initial)
        if c <= len(sample_tiles) and c <= ceiling
    ]
    if len(candidates) <= 1:
        return initial

    best_batch = candidates[0]
    best_per_tile = float('inf')
    stale_steps = 0
    results = []
    stop_reason = None
    # Working ceiling tightened by an OOM fallback during this sweep: once
    # set, no later candidate at or above it is attempted, whatever
    # stop_on_oom says -- the next candidate is by definition further past
    # the memory cliff that was just hit.
    oom_ceiling = None

    for cand in candidates:
        if oom_ceiling is not None and cand >= oom_ceiling:
            stop_reason = (
                f"stopped after OOM fallback: working ceiling lowered to "
                f"b{oom_ceiling}"
            )
            break

        used, per_tile, oomed = _time_batch_candidate(
            sample_tiles,
            sample_masks,
            model,
            config,
            cand,
            repeats=repeats,
        )

        results.append((cand, used, per_tile, oomed))

        # A candidate only counts as progress if it clears BOTH bars: the
        # existing relative one (min_improve) AND a new absolute floor
        # (min_improve_s). 0.337 vs 0.340 s/tile is jitter, not a win, and
        # treating it as one just chases noise to the top of the range.
        improved = (
            not np.isfinite(best_per_tile)
            or (per_tile < best_per_tile * (1.0 - min_improve)
                and per_tile <= best_per_tile - min_improve_s)
        )

        if improved:
            best_per_tile = per_tile
            best_batch = used
            stale_steps = 0
        else:
            stale_steps += 1

        if oomed:
            oom_ceiling = cand
            if stop_on_oom:
                stop_reason = (
                    f"stopped after OOM fallback at candidate {cand}"
                )
                break

        if stale_steps >= patience and cand > best_batch:
            stop_reason = (
                f"stopped after {stale_steps} non-improving step(s)"
            )
            break

    if results:
        summary_parts = []
        for cand, used, per_tile, oomed in results:
            if used == cand:
                txt = f"b{used}={per_tile:.3f}s/tile"
            else:
                txt = f"b{cand}->b{used}={per_tile:.4f}s/tile"
            if oomed:
                txt += " OOM"
            summary_parts.append(txt)

        summary = ', '.join(summary_parts)
        msg = f"{label} autotune: {summary} -> selected {best_batch}"
        if stop_reason is not None:
            msg = f"{msg} ({stop_reason})"
        print(msg, flush=True)

    return best_batch


def _split_jobs_for_producers(jobs, producer_workers: int):
    workers = max(1, int(producer_workers))
    if workers <= 1 or len(jobs) <= 1:
        return [jobs]
    total = len(jobs)
    out = []
    start = 0
    for worker_idx in range(workers):
        end = int(round((worker_idx + 1) * total / workers))
        shard = jobs[start:end]
        if shard:
            out.append(shard)
        start = end
    return out or [jobs]


def predict_stream_to_raster(
    uav_path: str,
    output_stem_map: str,
    model,
    config,
    tile_jobs=None,
):
    t = Timer()
    t.start()
    print("#######################################################")
    print("Prediction of the semantic stem map")
    print("Resampling tiles while analyzing (stream mode)")

    os.makedirs(os.path.dirname(output_stem_map) or '.', exist_ok=True)

    with rasterio.open(uav_path) as src:
        profile = src.profile.copy()
        layout = _resampling_layout((src.height, src.width), profile, config)

    out_profile = IO.build_safe_prediction_profile(
        src_profile=profile,
        width=layout['out_width'],
        height=layout['out_height'],
        transform=layout['out_transform'],
        compress='DEFLATE' if getattr(
            config, 'compress_output', True) else None,
        dtype='uint8',
    )

    total_tiles = layout['x_tiles'] * layout['y_tiles']
    initial_batch_size = max(1, int(getattr(
        config, 'prediction_batch_size', None) or getattr(
            config, 'prediction_batch_gpu', 1)))
    chunk_size = \
        max(initial_batch_size,
            int(getattr(config, 'prediction_batch_max_gpu', initial_batch_size)
                or initial_batch_size))
    queue_depth = max(2, int(getattr(
        config, 'producer_queue_batches', getattr(
            config, 'prediction_prefetch', 2))))
    progress_interval_s = float(getattr(config, 'progress_interval_s', 30.0))
    producer_workers = max(1, int(getattr(
        config, 'prediction_producer_workers', getattr(
            config, 'prediction_producer_workers_gpu', 1)) or 1))
    jobs_iter = list(_iter_tile_jobs(layout, config)) \
        if tile_jobs is None else list(tile_jobs)

    q = queue.Queue(maxsize=queue_depth)
    producer_job_lists = _split_jobs_for_producers(
        jobs_iter, producer_workers)
    producers = [
        TileBatchProducer(
            uav_path=uav_path,
            chunk_size=chunk_size,
            jobs=producer_job_lists[idx],
            n_channels=config.n_channels,
            out_queue=q,
            producer_id=idx,
            # Tiles arrive on the model grid already; _resize_batch's
            # identity fast path then makes _prepare_inference_batch a
            # no-op for the resize step.
            out_size=(config.img_height, config.img_width),
        )
        for idx in range(len(producer_job_lists))
    ]

    tmp_path = IO.atomic_tmp_path(output_stem_map)
    done = 0
    last_report = time.monotonic()
    start = time.monotonic()
    total_read_s = 0.0
    total_prep_s = 0.0
    total_infer_s = 0.0
    total_write_s = 0.0
    active_batch_size = initial_batch_size
    pending_items = []
    finished_producers = 0

    for producer in producers:
        producer.start()

    with rasterio.open(tmp_path, 'w', **out_profile) as dst:
        while finished_producers < len(producers) or pending_items:
            while (finished_producers < len(producers)
                   and len(pending_items) < chunk_size
                   ):
                payload = q.get()
                if (
                    isinstance(payload, dict)
                    and payload.get('producer_done')
                ):
                    finished_producers += 1
                    continue
                if payload is None:
                    finished_producers += 1
                    continue
                pending_items.extend(payload['items'])
                total_read_s += float(payload.get('read_s', 0.0))

            if not pending_items:
                continue

            if done == 0:
                sample_tiles = [tile for _, tile, _ in
                                pending_items[:chunk_size]]
                sample_masks = [mask for _, _, mask in
                                pending_items[:chunk_size]]
                # active_batch_size = _autotune_batch_size(
                #     sample_tiles, sample_masks, model,
                #     config, initial_batch_size)
                active_batch_size = _autotune_batch_size(
                    sample_tiles,
                    sample_masks,
                    model,
                    config,
                    initial_batch_size,
                    label='Prediction micro-batch',
                )

            current_n = min(active_batch_size, len(pending_items))
            items = pending_items[:current_n]
            pending_items = pending_items[current_n:]
            raw_tiles = [tile for _, tile, _ in items]
            raw_masks = [mask for _, _, mask in items]

            prep0 = time.perf_counter()
            tile_tensor, mask_resized = _prepare_inference_batch(
                raw_tiles, raw_masks, config)
            total_prep_s += time.perf_counter() - prep0

            infer0 = time.perf_counter()
            pred = model.predict_on_batch(tile_tensor)
            total_infer_s += time.perf_counter() - infer0

            crop = config.overlap_pred // 2
            write_batch_s = 0.0
            for idx, (job, _, _) in enumerate(items):
                pred_core = pred[idx, crop:(
                    config.img_width - crop), crop:(
                        config.img_width - crop), 0]
                mask_core = mask_resized[idx, crop:(
                    config.img_width - crop), crop:(
                        config.img_width - crop), 0] > 0.5
                pred_core = _binarize_prediction_core(
                    pred_core,
                    mask_core,
                    threshold=float(getattr(
                        config, 'stem_binary_threshold', 0.5)),
                )
                write_batch_s += _write_prediction_core(
                    dst, pred_core, job, layout)
                done += 1
            total_write_s += write_batch_s

            now = time.monotonic()
            if (
                done == 1
                or done == total_tiles
                or (now - last_report) >= progress_interval_s
            ):
                elapsed = max(now - start, 1e-9)
                rate = done / elapsed
                eta_s = (total_tiles - done) / rate if rate > 0 \
                    else float('inf')
                avg_read = total_read_s / max(done, 1)
                avg_prep = total_prep_s / max(done, 1)
                avg_infer = total_infer_s / max(done, 1)
                avg_write = total_write_s / max(done, 1)
                queue_fill = (q.qsize() / max(queue_depth, 1)) \
                    if queue_depth > 0 else 0.0
                print(
                    f"Written tile {done}/{total_tiles} | "
                    f"{done / total_tiles:.1%} | "
                    f"{rate * 60:.1f} tiles/min | ETA {_format_eta(eta_s)} | "
                    f"avg read {avg_read:.3f}s prep {avg_prep:.3f}s infer "
                    f"{avg_infer:.3f}s write {avg_write:.3f}s | "
                    f"batch {active_batch_size} | queue {queue_fill:.0%} full"
                    f" | producers {len(producers)} | "
                    f"src {layout['px_per_tile_x']}x{layout['px_per_tile_y']} "
                    f"-> out {config.img_width - config.overlap_pred}x"
                    f"{config.img_width - config.overlap_pred}",
                    flush=True,
                )
                last_report = now

    for producer in producers:
        producer.join()
        if producer.error is not None:
            raise producer.error
    IO.finalize_raster(tmp_path, output_stem_map)

    print(total_tiles, " tiles analyzed")
    t.stop()
    print("#######################################################")
    print("")
    return out_profile


def predict_stream_single_gpu(
    uav_path: str,
    output_stem_map: str,
    model,
    config,
):
    return predict_stream_to_raster(uav_path, output_stem_map, model, config)


def predict_stream_cpu(
    uav_path: str,
    output_stem_map: str,
    model,
    config,
):
    return predict_stream_to_raster(uav_path, output_stem_map, model, config)


def predict_with_resampling_stream_to_raster(
    uav_path, output_stem_path, model, config
):
    return predict_stream_to_raster(uav_path, output_stem_path, model, config)


def predict_with_resampling_per_tile(img, profile, model, config):
    t = Timer()
    t.start()
    print("#######################################################")
    print("Prediction of the semantic stem map")
    print("Resampling tiles while analyzing")

    layout = _resampling_layout(img.shape[:2], profile, config)
    sy = int(np.ceil(layout['y_tiles']
                     * (layout['px_per_tile_y'] - layout['overlap_img_y'])
                     + layout['overlap_img_y']))
    sx = int(np.ceil(layout['x_tiles']
                     * (layout['px_per_tile_x'] - layout['overlap_img_x'])
                     + layout['overlap_img_x']))
    img_pd = np.full(
        (sy, sx, config.n_channels),
        fill_value=0,
        dtype=np.float32
    )
    img_pd[0:img.shape[0], 0:img.shape[1], ] = img

    img_width_ = layout['img_width_inner']
    prediction = np.zeros((layout['out_height'], layout['out_width']),
                          dtype=np.uint8)
    mask = np.where(img_pd[:, :, 0:3] == (0, 0, 0), False, True)[:, :, 0]
    mask = resize(mask, prediction.shape, order=0, preserve_range=True,
                  anti_aliasing=False).astype(bool)

    for i in range(layout['y_tiles']):
        x = int(np.floor(i * (layout['px_per_tile_y'] -
                              layout['overlap_img_y'])))
        for j in range(layout['x_tiles']):
            y = int(np.floor(j * (layout['px_per_tile_x'] -
                                  layout['overlap_img_x'])))
            tile = img_pd[x:x + layout['px_per_tile_x'] - 1, y:
                          y + layout['px_per_tile_y'] - 1, 0:3]
            pred2 = predict_tile_array(tile, model, config)
            prediction[(config.overlap_pred // 2 + i * img_width_):
                       ((config.img_width - config.overlap_pred // 2)
                       + i * img_width_),
                       (config.overlap_pred // 2 + j * img_width_):
                       ((config.img_width - config.overlap_pred // 2)
                       + j * img_width_),
                       ] = pred2
    prediction = np.ascontiguousarray((prediction > 0) & mask, dtype=np.uint8)

    profile['transform'] = layout['out_transform']
    print(layout['x_tiles'] * layout['y_tiles'], " tiles analyzed")
    t.stop()
    print("#######################################################")
    print("")
    return prediction, profile
