#!/usr/bin/env python

################################################################################
"""Imports"""
import os
import queue
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
from plugin_utils import autotune_cache
from plugin_utils.gpu_probe import run_nvidia_smi_query
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


#: Tile-read/resample strategies, selectable per run through
#: ``Config.prediction_read_strategy`` (default `graph`) or, for A/B
#: benchmarking, the WINMOL_BENCH_READ environment variable. They produce
#: DIFFERENT pixels, so the choice is an accuracy question, not just a
#: speed one -- compare them on real stem output (docs/resize-mechanics.md,
#: benchmark/bench_resize_parity.py):
#:   graph     : native uint8 reads; normalize + Catmull-Rom resize run
#:               INSIDE the ONNX graph. v0.5.0-equivalent on every
#:               execution provider. THE DEFAULT.
#:   graph_aa  : `graph` with ONNX Resize antialias=1 -- GDAL-like AA
#:               semantics, but deterministic and portable. For settling
#:               the AA accuracy question, not v0.5-equivalent.
#:   overview  : out_shape+cubic in the GDAL read (overview-served).
#:               The flag-gated fast path: ~25% faster end-to-end on
#:               R13-scale orthos but an anti-aliased kernel — measured
#:               +20% stems / +26% volume vs v0.5 semantics at 2.29x,
#:               validity unresolved. NOT the default for that reason.
#:               (The `fullres`/`boundless` bench variants of this kernel
#:               were removed 2026-08-11 after the investigation closed.)
#:   native    : read at native resolution, resize in skimage downstream
#:   native_producer : `native` pixels EXACTLY, resized in the producers
#:   cupy      : rc12's CUDA/CuPy preprocessing (guarded until the port
#:               is validated on a CUDA box)
_READ_STRATEGIES = ("graph", "graph_aa", "overview",
                    "native", "native_producer", "cupy")
#: The in-graph path predates its promotion under the bench name
#: `onnx_gpu`; keep the alias so existing bench scripts keep working.
_STRATEGY_ALIASES = {"onnx_gpu": "graph"}


def resolve_read_strategy(config=None):
    """The feature-flag resolution: WINMOL_BENCH_READ (benching override)
    beats ``config.prediction_read_strategy`` beats the `graph` default."""
    raw = (os.environ.get("WINMOL_BENCH_READ") or "").lower()
    if not raw and config is not None:
        raw = str(getattr(config, "prediction_read_strategy", "")
                  or "").lower()
    raw = _STRATEGY_ALIASES.get(raw, raw) or "graph"
    if raw not in _READ_STRATEGIES:
        raise ValueError(
            f"unknown read strategy {raw!r}; expected one of "
            f"{_READ_STRATEGIES}")
    if raw == "cupy":
        # Recognized but unported. Raising HERE, at the single chokepoint
        # every consumer calls, means no entry point can silently degrade
        # `cupy` into a different strategy's code path.
        raise RuntimeError(
            "the CuPy read strategy (rc12's CUDA preprocessing) is not "
            "ported yet -- it needs cupy-cuda12x and a CUDA device, and "
            "validation on such a box. Use 'graph' for v0.5-equivalent "
            "output; see docs/resize-mechanics.md.")
    return raw


def strategy_wraps_graph(strategy):
    """True when the strategy loads a graph-wrapped model that takes raw
    uint8 NHWC tiles. Model loading (utils.IO) and batch preparation
    below MUST agree on this set, or the wrapped model's uint8 input
    rejects every batch -- hence one shared predicate."""
    return strategy in ("graph", "graph_aa")


def _to_float32_image(arr):
    if arr.dtype == np.float32:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        # NOT `(arr / 255.0).astype(np.float32)`: that promotes uint8 to
        # float64 (a 6.3 MB temporary per 512x512x3 tile) only to round it
        # back down, at 4.9x the cost. Dividing straight into float32 is
        # one pass and BIT-IDENTICAL across all 256 uint8 values.
        return np.divide(arr, np.float32(255.0), dtype=np.float32)
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


def _resize_like_consumer(tile, valid_mask, out_size):
    """Producer-side twin of _prepare_inference_batch's resize.

    Reproduces the consumer's operation order exactly -- uint8 -> float32
    /255 via _to_float32_image, then skimage order=3 for imagery and
    order=0 for the mask -- so a tile resized here is bit-comparable to
    one resized there. The consumer's fast paths then short-circuit:
    _to_float32_image passes float32 through untouched and _resize_batch
    sees the batch already at target size.
    """
    size = (int(out_size[0]), int(out_size[1]))
    tile_f = _raw_tile_to_batchable(tile)
    tile_r = _resize_batch(tile_f[None, ...], size, order=3)[0]
    mask_f = valid_mask.astype(np.float32)[:, :, None]
    mask_r = _resize_batch(mask_f[None, ...], size, order=0)[0, :, :, 0]
    return tile_r, mask_r > 0.5


def _prepare_inference_batch(raw_tiles, raw_masks, config,
                             read_strategy=None):
    """read_strategy: pass the already-resolved strategy on hot paths (the
    stream loop resolves once); None resolves from config. Callers that
    load an UNWRAPPED model (wrap_preprocess=False) must pass a non-graph
    strategy explicitly -- see PredictWorkers."""
    if read_strategy is None:
        read_strategy = resolve_read_strategy(config)
    if raw_masks is None:
        raw_masks = [_default_valid_mask(t) for t in raw_tiles]
    mask_batch = np.stack(
        [m.astype(np.float32)[:, :, None] for m in raw_masks],
        axis=0,
    )

    if strategy_wraps_graph(read_strategy):
        # The wrapped model normalizes and resizes in-graph: hand it the
        # native uint8 batch untouched. Producers already resized the
        # masks to the model grid, so only stacking remains. EVERY caller
        # -- the consumer loop and the autotune probes alike -- must feed
        # the model this way, or the uint8 graph input rejects the batch.
        from utils.onnx_preprocess import as_uint8_nhwc
        return as_uint8_nhwc(raw_tiles), mask_batch

    batch = np.stack([_raw_tile_to_batchable(t) for t in raw_tiles], axis=0)
    size = (config.img_height, config.img_width)
    tile_batch = _resize_batch(batch, size, order=3)
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
    # Sweep from 1, not from the planner's `initial`. Starting at `initial`
    # made the low end unreachable: a machine that cannot fit the planner's
    # batch had no way to discover 1 or 2 and simply OOMed at run time.
    # `initial` still seeds the planner and the cache; it is no longer the
    # floor of the search.
    return list(range(1, max_batch + 1))


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
    it is unavailable or fails. Never raises. The query is bounded: a
    driver stuck in an uninterruptible ioctl used to hang this call
    forever (rr NVIDIA_SMI_TIMEOUT); on timeout the caller falls back
    to the host-RAM bound."""
    lines = run_nvidia_smi_query("memory.free", timeout=8.0, nounits=True)
    values = []
    for line in lines or []:
        try:
            values.append(float(line) / 1024.0)
        except ValueError:
            continue
    return values


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
                 out_queue, producer_id=0, out_size=None,
                 read_strategy="overview"):
        super().__init__(daemon=True)
        self.uav_path = uav_path
        self.chunk_size = max(1, int(chunk_size))
        self.jobs = jobs
        self.n_channels = n_channels
        self.out_queue = out_queue
        self.producer_id = producer_id
        # (H, W) of the model grid. For the GDAL strategies the tile is
        # resampled to it *during* the read; for `graph`/`cupy` the tile
        # stays native and only the validity mask is resized to it here.
        self.out_size = tuple(out_size) if out_size else None
        self.read_strategy = read_strategy
        self.error = None

    def run(self):
        try:
            strat = self.read_strategy
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
                    # then short-circuits).
                    # Only an edge window needs boundless -- it is what
                    # keeps the returned shape correct there. Paying for it
                    # on interior windows costs 3.2x on a native read
                    # (47.3 -> 15.0 ms) for pixel-IDENTICAL output, and
                    # on the out_shape path it also blocks overview use.
                    # Applies to EVERY read strategy.
                    bl = not (
                        window.col_off >= 0
                        and window.row_off >= 0
                        and window.col_off + window.width <= src.width
                        and window.row_off + window.height <= src.height
                    )
                    read_kw = {"boundless": bl,
                               "fill_value": 0 if bl else None}
                    mask_kw = {"boundless": bl}
                    # `overview` resamples during the read; every other
                    # strategy reads native and resizes downstream. The
                    # native read is now the DEFAULT path, so the rule
                    # above matters more here than on the overview branch.
                    if self.out_size is not None and strat == "overview":
                        oh, ow = self.out_size
                        read_kw.update(out_shape=(len(indexes), oh, ow),
                                       resampling=Resampling.cubic)
                        mask_kw.update(out_shape=(oh, ow),
                                       resampling=Resampling.nearest)
                    tile = src.read(
                        indexes, window=window, **read_kw).transpose(1, 2, 0)
                    gdal_mask = src.read_masks(
                        1, window=window, **mask_kw) > 0

                    pixel_mask = np.any(tile != 0, axis=2)

                    # If GDAL mask is effectively all valid, it is not helping.
                    # Fall back to pixel-based validity for
                    # black background suppression.
                    if np.all(gdal_mask):
                        valid_mask = pixel_mask
                    else:
                        valid_mask = gdal_mask & pixel_mask

                    if strategy_wraps_graph(strat) and self.out_size:
                        # The graph resizes the IMAGE on device; the mask
                        # is only needed at model resolution for the
                        # binarize step, and nearest on one channel is
                        # cheap enough to keep here (and parallel).
                        mk = valid_mask.astype(np.float32)[:, :, None]
                        valid_mask = _resize_batch(
                            mk[None, ...], (int(self.out_size[0]),
                                            int(self.out_size[1])),
                            order=0)[0, :, :, 0] > 0.5
                    elif strat == "native_producer" and self.out_size:
                        # Same skimage resize the consumer would do, but
                        # run HERE so it parallelises across producers
                        # instead of serialising on the GIL-holding
                        # consumer. Order of operations matches
                        # _prepare_inference_batch exactly -- float32
                        # scale FIRST, then resize -- so the pixels are
                        # identical to `native`, not merely similar.
                        tile, valid_mask = _resize_like_consumer(
                            tile, valid_mask, self.out_size)

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


def _is_oom_error(exc) -> bool:
    """True when ``exc`` is an out-of-memory failure from the backend.

    onnx_runtime normalizes an onnxruntime OOM to MemoryError; the string
    check is the fallback for a raw RuntimeError. The CUDA BFC-arena
    wording ("Failed to allocate memory for requested buffer ...") carries
    neither 'oom' nor 'out of memory' and so slipped past the back-off
    before, aborting the run (issue #40).
    """
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return ('oom' in msg or 'out of memory' in msg
            or 'failed to allocate memory' in msg)


def _predict_tensor_adaptive(tile_tensor, model, batch_size):
    """Run the model over an already-prepared batch, halving on OOM.

    Returns ``(pred, used)``. The ENTIRE tensor is always predicted: a
    reduction re-runs it in slices and concatenates, so the caller gets one
    row per input tile no matter how far the batch had to come down. This
    is what the streaming loop calls -- it used to call
    ``model.predict_on_batch`` directly, so the first steady-state OOM
    killed the whole run even though the back-off below already existed
    (it was reachable only from the autotune).
    """
    try:
        return np.asarray(model.predict_on_batch(tile_tensor)), batch_size
    except (RuntimeError, MemoryError) as exc:
        if batch_size <= 1 or not _is_oom_error(exc):
            raise
        reduced = max(1, batch_size // 2)
        print(f"Prediction batch too large; reducing micro-batch size from "
              f"{batch_size} to {reduced}", flush=True)
        # Carry the working size forward across slices: once a slice has
        # backed off to N, start the next one at N rather than re-probing
        # from `reduced` and paying another failed allocation. On a
        # 99k-tile ortho that difference is the whole cost of the fallback.
        parts = []
        used = reduced
        start = 0
        while start < len(tile_tensor):
            part, part_used = _predict_tensor_adaptive(
                tile_tensor[start:start + used], model, used)
            parts.append(part)
            start += len(part)
            used = min(used, part_used)
        return np.concatenate(parts, axis=0), used


def _predict_batch_adaptive(
    raw_tiles, raw_masks, model, config, batch_size
):
    try:
        return _predict_batch_core(
            raw_tiles, raw_masks, model, config), batch_size
    except (RuntimeError, MemoryError) as exc:
        if batch_size <= 1 or not _is_oom_error(exc):
            raise
        reduced = max(1, batch_size // 2)
        print(f"Prediction batch too large; reducing micro-batch size from "
              f"{batch_size} to {reduced}", flush=True)
        # Re-run the WHOLE input in slices of `reduced` -- never just
        # raw_tiles[:reduced]. Truncating here silently dropped every tile
        # past the first slice: harmless while only the autotune called this
        # (it keeps timings, not predictions), but the streaming loop writes
        # what comes back, so a truncated result is a hole in the stem map
        # with no error anywhere.
        cores = []
        used = reduced
        start = 0
        while start < len(raw_tiles):
            stop = start + used
            slice_masks = (
                raw_masks[start:stop] if raw_masks is not None else None)
            slice_cores, slice_used = _predict_batch_adaptive(
                raw_tiles[start:stop], slice_masks, model, config, used)
            cores.extend(slice_cores)
            start += len(slice_cores)
            # Report the SMALLEST size that worked: a later slice may have
            # had to back off further, and the caller latches this value for
            # the rest of the run.
            used = min(used, slice_used)
        return cores, used


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


def _autotune_cache_key(model, config, label):
    """``(key, cache_file)`` for the sweep's persistent cache, or
    ``(None, None)`` when the key cannot be derived -- never fatal, a miss
    just means the sweep below runs. See plugin_utils/autotune_cache.py.
    """
    try:
        key = autotune_cache.cache_key(
            model, config, getattr(config, 'hardware', None))
        return key, autotune_cache.cache_path()
    except Exception as exc:                                # pragma: no cover
        print(f"{label} autotune: cache key unavailable ({exc}); tuning.",
              flush=True)
        return None, None


def _autotune_cache_lookup(
    mode, key, cache_file, initial, max_reachable, label,
):
    """The cached batch to reuse, or ``None`` to fall through to the sweep.

    "auto" tunes ONCE per (hardware, model, execution provider, tile
    geometry) and reuses the persisted answer forever after; "force" never
    looks here, it always re-sweeps and refreshes the entry (see the
    caller). A cached value outside ``[initial, max_reachable]`` -- what
    THIS run can actually try, given the sample and the memory ceiling --
    is re-tuned rather than clamped: it was never measured under the
    current constraint.
    """
    if mode != 'auto' or key is None:
        return None
    cached = autotune_cache.load(key, path=cache_file)
    if cached is None:
        return None
    if initial <= cached <= max_reachable:
        print(
            f"{label} autotune: using cached batch {cached} "
            f"(key {key[:8]}, {cache_file})",
            flush=True,
        )
        return cached
    print(
        f"{label} autotune: ignoring out-of-range cached batch "
        f"{cached} (valid {initial}-{max_reachable}); re-tuning.",
        flush=True,
    )
    return None


def _autotune_cache_persist(
    key, cache_file, best_batch, best_per_tile, candidates, label,
):
    if key is None:
        return
    meta = {
        'per_tile_s': (None if not np.isfinite(best_per_tile)
                       else round(float(best_per_tile), 6)),
        'candidates': [int(c) for c in candidates],
        'label': str(label),
    }
    if autotune_cache.store(key, best_batch, meta=meta, path=cache_file):
        print(
            f"{label} autotune: cached batch {best_batch} "
            f"(key {key[:8]}, {cache_file})",
            flush=True,
        )
    else:
        print(
            f"{label} autotune: could not write {cache_file}; "
            "the result will be re-measured next run.",
            flush=True,
        )


def _persist_autotune_batch(key, cache_file, batch):
    """Lower the cached batch after a steady-state OOM.

    The sweep caches the size that fit a warm sample; once the real run
    OOMs and backs off, that cached value is known-fatal for this
    model/machine pair. Leaving it in place makes the NEXT run load it and
    die exactly the same way without even re-probing -- which is what the
    reported crash did.
    """
    if key is None:
        return
    stored = autotune_cache.store(
        key, int(batch), meta={'lowered_after_oom': True}, path=cache_file)
    if stored:
        print(f"Prediction micro-batch autotune: lowered cached batch to "
              f"{int(batch)} after an out-of-memory back-off "
              f"(key {key[:8]}, {cache_file})", flush=True)


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

    # "off": never sweep, never touch the cache (no read, no write) -- the
    # $WINMOL_BATCH_AUTOTUNE env var wins over Config.prediction_batch_autotune,
    # see plugin_utils.autotune_cache.resolve_mode.
    mode = autotune_cache.resolve_mode(config)
    if mode == 'off':
        return initial
    if len(sample_tiles) < 2:
        return initial

    # CoreML recompiles the model for every distinct batch shape, so the
    # sweep pays a recompile per candidate (~20s total on an M2). Measured
    # on this model: batch 2 is only ~3% faster per image than batch 1,
    # and batch>=4 is 2-3x SLOWER. That best-case ~3% is smaller than the
    # recompile cost of the sweep that would find it, so batch 1 is the
    # right default -- skip the sweep entirely (a user pin still wins).
    accel = str(getattr(model, 'accelerator', '') or '').lower()
    if accel == 'coreml':
        print(
            f"{label} autotune: CoreML gains <=3% from batching and is "
            f"much slower at batch>=4, not worth the recompile sweep — "
            f"using batch {initial}.",
            flush=True,
        )
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

    # Tune once, reuse forever (mode 'auto'); mode 'force' skips straight
    # to the sweep and refreshes the entry afterwards. See
    # _autotune_cache_key / _autotune_cache_lookup above.
    key, cache_file = _autotune_cache_key(model, config, label)
    max_reachable = candidates[-1] if candidates else initial
    cached = _autotune_cache_lookup(
        mode, key, cache_file, initial, max_reachable, label)
    if cached is not None:
        return cached

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
        # relative one (min_improve) AND a noise floor. 0.337 vs 0.340
        # s/tile is jitter, not a win, and treating it as one just chases
        # noise to the top of the range.
        #
        # The floor is RELATIVE to the measured baseline. A fixed
        # min_improve_s (0.2 s/tile by default) is unreachable on a GPU,
        # where per-tile times are 0.01-0.05 s: candidates[0] always wins
        # on the isfinite() branch, every later candidate then needs
        # per_tile <= best - 0.2 (negative), and the sweep could only ever
        # return its own starting point -- timing 5 candidates x 5 repeats
        # to re-derive the planner's value.
        #
        # min_improve_s stays as an upper bound, so the floor never gets
        # LOOSER than the tuned value; on slow CPU tiles (~0.8 s) it does
        # get tighter (0.2 -> ~0.04), which is the point: 5% of measured is
        # a real win at any speed, 0.2 s absolute is not a scale-free test.
        noise_floor = min(min_improve_s, max(0.002, 0.05 * best_per_tile))
        improved = (
            not np.isfinite(best_per_tile)
            or (per_tile < best_per_tile * (1.0 - min_improve)
                and per_tile <= best_per_tile - noise_floor)
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

    _autotune_cache_persist(
        key, cache_file, best_batch, best_per_tile, candidates, label)

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

    read_strategy = resolve_read_strategy(config)
    print(f"Tile read strategy: {read_strategy}")

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
            # For the GDAL strategies tiles arrive on the model grid
            # already (identity fast path downstream); for `graph` the
            # producers use it only to resize the validity mask.
            out_size=(config.img_height, config.img_width),
            read_strategy=read_strategy,
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
    # Resolved once so a steady-state OOM can lower the cached batch: the
    # sweep caches what fit a warm sample, and if the real run then backs
    # off, that cached value is known-fatal for this model/machine.
    autotune_key, autotune_cache_file = _autotune_cache_key(
        model, config, 'Prediction micro-batch')
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
                raw_tiles, raw_masks, config, read_strategy=read_strategy)
            total_prep_s += time.perf_counter() - prep0

            infer0 = time.perf_counter()
            pred, used_batch = _predict_tensor_adaptive(
                tile_tensor, model, current_n)
            total_infer_s += time.perf_counter() - infer0
            # Latch the reduction for the REST of the run. Without this every
            # subsequent batch re-hits the same memory cliff, pays the failed
            # allocation, and backs off again -- and a batch that OOMs at the
            # very first tile would never make progress at all.
            if used_batch < active_batch_size:
                active_batch_size = used_batch
                _persist_autotune_batch(autotune_key, autotune_cache_file,
                                        used_batch)

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
