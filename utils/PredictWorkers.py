from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.windows import Window

from classes.Config import Config
from utils import IO
from utils.Prediction import (_predict_batch_core, _resize_batch,
                              resolve_read_strategy, strategy_wraps_graph)


def _config_from_dict(config_dict: dict) -> Config:
    cfg = Config()
    for key, value in config_dict.items():
        try:
            setattr(cfg, key, value)
        except Exception:
            pass
    return cfg


def _resampling_layout(shape, profile, config):
    height, width = int(shape[0]), int(shape[1])
    px_per_tile_x = int(np.ceil(config.tile_size
                                / abs(profile['transform'][0])))
    px_per_tile_y = int(np.ceil(config.tile_size
                                / abs(profile['transform'][4])))
    overlap_img_x = config.overlap_pred * px_per_tile_x / config.img_width
    overlap_img_y = config.overlap_pred * px_per_tile_y / config.img_width
    x_tiles = int(np.ceil(width / max(px_per_tile_x - overlap_img_x, 1)))
    y_tiles = int(np.ceil(height / max(px_per_tile_y - overlap_img_y, 1)))
    img_width_inner = config.img_width - config.overlap_pred
    out_width = int(x_tiles * img_width_inner + config.overlap_pred)
    out_height = int(y_tiles * img_width_inner + config.overlap_pred)
    from rasterio import Affine

    out_transform = Affine(
        profile['transform'][0] * px_per_tile_x / config.img_width, 0.0,
        profile['transform'][2], 0.0,
        profile['transform'][4] * px_per_tile_y / config.img_width,
        profile['transform'][5],
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
            src_col = \
                int(np.floor(j * (layout['px_per_tile_x']
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


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return 'unknown'
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _group_jobs(jobs, batch_size):
    batch = []
    for job in jobs:
        batch.append(job)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _graph_out_size(config):
    """Model grid when the wrapped graph will do the resize, else None.

    None keeps the pre-existing behaviour for the non-graph strategies:
    masks stay native and `_prepare_inference_batch` resizes both tile
    and mask on the CPU, as it always did.
    """
    if not strategy_wraps_graph(resolve_read_strategy(config)):
        return None
    return (int(config.img_height), int(config.img_width))


def _read_batch_jobs(src, indexes, batch_jobs, out_size=None):
    """Read a batch of tiles at native resolution.

    ``out_size`` is the model grid. When it is set the validity mask is
    resized onto it here and the IMAGE is left native, which is the
    contract the wrapped graph expects (it resizes the image on device).
    Nearest on one channel is cheap; bicubic on three at 1250^2 is not,
    which is the whole reason this path was slow.
    """
    raw_tiles = []
    raw_masks = []
    stats = {
        'read_s': 0.0,
        'read_data_s': 0.0,
        'read_mask_s': 0.0,
        'prep_s': 0.0,
        'jobs': len(batch_jobs),
    }
    for job in batch_jobs:
        t0 = time.perf_counter()
        window = Window(job['src_col'], job['src_row'],
                        job['src_width'], job['src_height'])
        tile = src.read(
            indexes,
            window=window,
            boundless=True,
            fill_value=0,
        )
        stats['read_data_s'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        gdal_mask = src.read_masks(
            1,
            window=window,
            boundless=True,
        ) > 0
        stats['read_mask_s'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        tile = tile.transpose(1, 2, 0)
        pixel_mask = np.any(tile != 0, axis=2)
        valid_mask = \
            pixel_mask if np.all(gdal_mask) else (gdal_mask & pixel_mask)
        stats['prep_s'] += time.perf_counter() - t0

        if out_size is not None:
            t0 = time.perf_counter()
            mk = valid_mask.astype(np.float32)[:, :, None]
            valid_mask = _resize_batch(
                mk[None, ...],
                (int(out_size[0]), int(out_size[1])),
                order=0)[0, :, :, 0] > 0.5
            stats['prep_s'] += time.perf_counter() - t0

        raw_tiles.append(tile)
        raw_masks.append(valid_mask)
    stats['read_s'] = \
        stats['read_data_s'] + stats['read_mask_s'] + stats['prep_s']
    return raw_tiles, raw_masks, stats


def _watch_parent(sentinel, exit_fn=lambda: os._exit(1)):
    """Daemon-thread body: hard-exit this process the moment the parent
    (the coordinator, or ultimately the CLI process the QGIS plugin
    started) is gone.

    The plugin cancels a run by terminating the CLI process only
    (`tasks_threads.py:104-113`). On `main` the single-GPU path ran
    in-process, so that killed everything. Now the spawned worker(s)
    survive -- they keep the GPU, and, having inherited fd 1, keep the
    plugin's log `readline()` (and this repo's harness's
    `for line in proc.stdout`) from ever seeing EOF, because the pipe's
    write end is still open in the orphaned worker.

    `sentinel` is `multiprocessing.parent_process().sentinel`: an fd that
    becomes ready to read the moment the parent process exits. `wait()`
    blocks until then and returns; nothing here polls.
    """
    if sentinel is None:
        return
    try:
        from multiprocessing.connection import wait
        wait([sentinel])
    except Exception:
        return
    exit_fn()


def prediction_worker(
    gpu_id: Optional[int],
    model_path: str,
    input_raster: str,
    jobs: List[dict],
    results,
    config_dict: dict,
):
    """One process per accelerator. gpu_id=None means CPU.

    Reads are NOT done here any more: a ReaderPool of threads decodes
    batches ahead of us into a bounded queue, so the GPU never waits on
    a 27 ms decode. Any failure -- ours or a reader's -- is reported on
    `results` as {'error': ...}; the coordinator fails the run on it.
    """
    # See `_watch_parent`. `parent_process()` is None when this function is
    # called directly (e.g. from a test), not via `mp.Process` -- that is
    # not a bug, there is no parent to watch for.
    parent = mp.parent_process()
    if parent is not None:
        threading.Thread(
            target=_watch_parent, args=(parent.sentinel,),
            daemon=True).start()
    try:
        if gpu_id is not None:
            # Honour a CUDA_VISIBLE_DEVICES the user already set (main's
            # in-process path did): treat gpu_id as an INDEX into it
            # rather than overwriting it outright, so a user pinned to
            # e.g. "1" does not silently get card 0. Unset stays as
            # before. An out-of-range index fails loudly -- surfaced
            # below as {'error': ...} -- rather than falling back to an
            # unintended device.
            existing = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if existing:
                os.environ['CUDA_VISIBLE_DEVICES'] = \
                    existing.split(',')[gpu_id]
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        from utils.IO import load_model_from_path
        from utils.reader_pool import ReaderPool

        cfg = _config_from_dict(config_dict)
        # Wrapped, so the normalize + resize run on THIS worker's device
        # instead of on the CPU inside the timed inference block.
        model = load_model_from_path(model_path, cfg)
        # The dispatch used to print this from winmol_run.py; the model now
        # loads in this child instead, so print it here -- the plugin's
        # GPU-verdict UX parses this exact line.
        from utils.onnx_runtime import last_active_report
        report = last_active_report()
        if report:
            print(
                f"Execution providers (active): "
                f"{report['active_providers']} "
                f"(device: {report['accelerator_label']})")
        out_size = _graph_out_size(cfg)
        batch_size = max(1, int(getattr(cfg, 'prediction_batch_size', None)
                                or getattr(cfg, 'prediction_batch_gpu', 4)))
        n_readers = max(1, int(config_dict.get(
            'prediction_producer_workers', 1) or 1))
        depth = max(1, int(config_dict.get('producer_queue_batches', 4) or 4))

        with rasterio.open(input_raster) as probe:
            indexes = list(range(1, min(cfg.n_channels, probe.count) + 1))

        # Group readers at the plan's reader chunk, not at `batch_size`:
        # autotune samples the FIRST batch handed to it, and capping that
        # sample at `batch_size` (4 on a single GPU) also caps the
        # candidates `_prediction_batch_candidates` can sweep
        # (`c <= len(sample_tiles)`), so a 16-candidate times "correctly"
        # against a 4-tile sample, reports 4x too fast, wins, and latches
        # `active_batch` at a value the reader batch then silently caps
        # forever. CPU (batch 1) never sweeps at all (`len(sample) < 2`).
        # The chunk itself is plan-derived (ExecutionPlan.reader_chunk):
        # 12 on multi-GPU, but smaller on a single-worker machine, where a
        # deep chunk only spends RAM the pool no longer needs it to.
        chunk = max(batch_size,
                    int(getattr(cfg, 'prediction_reader_chunk', 12)))
        pool = ReaderPool(
            input_raster, list(_group_jobs(jobs, chunk)),
            lambda src, b: _read_batch_jobs(src, indexes, b, out_size),
            n_readers=n_readers, queue_depth=depth)
        from utils.Prediction import (_autotune_batch_size,
                                      _autotune_cache_key,
                                      _persist_autotune_batch,
                                      _predict_batch_adaptive)
        autotune_key, autotune_cache_file = _autotune_cache_key(
            model, cfg, 'Prediction micro-batch')
        active_batch = None
        try:
            pool.start()
            while True:
                item = pool.get()
                if item is None:
                    break
                batch_jobs, raw_tiles, raw_masks, read_stats = item
                if active_batch is None:
                    # Once per worker, on real tiles -- the stream path
                    # did exactly this; without it a 12 GB card OOMs where
                    # it used to back off.
                    active_batch = _autotune_batch_size(
                        raw_tiles, raw_masks, model, cfg, batch_size,
                        label='Prediction micro-batch', max_batch=chunk)
                # Readers group at the (now larger) chunk size; re-chunk
                # here so a reduced micro-batch stays reduced, as the
                # stream loop did -- _predict_batch_adaptive does not
                # slice its input on the success path, so handing it the
                # whole (possibly oversized) reader batch every time would
                # silently re-OOM and only recover via its own internal
                # recursion.
                i = 0
                while i < len(batch_jobs):
                    sl = slice(i, i + active_batch)
                    infer0 = time.perf_counter()
                    pred_cores, new_batch = _predict_batch_adaptive(
                        raw_tiles[sl], raw_masks[sl], model, cfg,
                        active_batch)
                    if new_batch < active_batch:
                        # Latch the reduction for the REST of the run, and
                        # persist it -- exactly what the stream path does
                        # (utils/Prediction.py:1197-1200) -- so the NEXT
                        # run does not load the stale (too-high) cached
                        # batch and OOM again before even re-probing.
                        _persist_autotune_batch(
                            autotune_key, autotune_cache_file, new_batch)
                    active_batch = new_batch
                    infer_s = time.perf_counter() - infer0
                    chunk_jobs = batch_jobs[sl]
                    n = max(len(chunk_jobs), 1)
                    for job, pred_core in zip(chunk_jobs, pred_cores):
                        results.put({
                            'row_off': job['dst_row'],
                            'col_off': job['dst_col'],
                            'array': pred_core,
                            'read_s': read_stats['read_s']
                            / max(len(batch_jobs), 1),
                            'infer_s': infer_s / n,
                        })
                    # Advance by what actually came back, not by
                    # `active_batch`: the returned size may have shrunk
                    # mid-chunk.
                    i += len(pred_cores)
        finally:
            pool.close()
    except BaseException as exc:      # noqa: BLE001 -- report, never vanish
        results.put({'error': f"{type(exc).__name__}: {exc}",
                     'gpu_id': gpu_id})
        return
    results.put({'done': True, 'gpu_id': gpu_id})


def prediction_service_worker(
    gpu_id: int,
    model_path: str,
    input_raster: str,
    tasks,
    results,
    config_dict: dict,
):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    from utils.IO import load_model_from_path

    cfg = _config_from_dict(config_dict)
    # Same reasoning as prediction_worker: wrapped so the preprocessing
    # runs on device.
    model = load_model_from_path(model_path, cfg)
    out_size = _graph_out_size(cfg)
    batch_size = max(1, int(getattr(cfg, 'prediction_batch_size', None)
                            or getattr(cfg, 'prediction_batch_gpu', 4)))

    with rasterio.open(input_raster) as src:
        indexes = list(range(1, min(cfg.n_channels, src.count) + 1))
        while True:
            task = tasks.get()
            if task is None or task.get('cmd') == 'stop':
                break

            request_id = int(task['request_id'])
            jobs = list(task.get('jobs') or [])
            outputs = []
            stats = {
                'jobs': len(jobs),
                'read_s': 0.0,
                'read_data_s': 0.0,
                'read_mask_s': 0.0,
                'prep_s': 0.0,
                'infer_s': 0.0,
            }

            for batch_jobs in _group_jobs(jobs, batch_size):
                raw_tiles, raw_masks, read_stats = \
                    _read_batch_jobs(src, indexes, batch_jobs, out_size)
                infer0 = time.perf_counter()
                pred_cores = _predict_batch_core(
                    raw_tiles, raw_masks, model, cfg)
                stats['infer_s'] += time.perf_counter() - infer0
                stats['read_s'] += float(read_stats.get('read_s', 0.0))
                stats['read_data_s'] \
                    += float(read_stats.get('read_data_s', 0.0))
                stats['read_mask_s'] \
                    += float(read_stats.get('read_mask_s', 0.0))
                stats['prep_s'] += float(read_stats.get('prep_s', 0.0))
                for job, pred_core in zip(batch_jobs, pred_cores):
                    outputs.append({
                        'request_index': int(job['__request_index__']),
                        'job': {
                            k: v for k, v in job.items()
                            if k != '__request_index__'
                        },
                        'array': pred_core,
                    })

            results.put({
                'request_id': request_id,
                'gpu_id': gpu_id,
                'outputs': outputs,
                'stats': stats,
            })


def start_multi_gpu_prediction_service(
    model_path: str,
    input_raster: str,
    gpu_ids: list[int],
    config,
):
    if not gpu_ids:
        raise ValueError(
            'start_multi_gpu_prediction_service requires at least one GPU id')

    ctx = mp.get_context('spawn')
    task_q = ctx.Queue(maxsize=max(8, len(gpu_ids) * 4))
    result_q = ctx.Queue(maxsize=max(8, len(gpu_ids) * 4))
    cfg_dict = dict(getattr(config, 'to_dict', lambda: {})())
    workers = []
    for gpu_id in gpu_ids:
        p = ctx.Process(
            target=prediction_service_worker,
            args=(gpu_id, model_path, input_raster, task_q, result_q, cfg_dict),
        )
        p.start()
        workers.append(p)
    return {
        'ctx': ctx,
        'task_q': task_q,
        'result_q': result_q,
        'workers': workers,
        'gpu_ids': list(gpu_ids),
        'next_request_id': 1,
    }


def stop_multi_gpu_prediction_service(service):
    if not service:
        return
    task_q = service.get('task_q')
    workers = list(service.get('workers') or [])
    for _ in workers:
        try:
            task_q.put({'cmd': 'stop'})
        except Exception:
            pass
    for proc in workers:
        try:
            proc.join(timeout=10.0)
        except Exception:
            pass
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=2.0)
            except Exception:
                pass


def predict_jobs_multi_gpu(service, jobs):
    jobs = list(jobs or [])
    stats = {
        'jobs': len(jobs),
        'read_s': 0.0,
        'read_data_s': 0.0,
        'read_mask_s': 0.0,
        'prep_s': 0.0,
        'infer_s': 0.0,
    }
    if not jobs:
        return [], stats

    gpu_ids = list(service.get('gpu_ids') or [])
    if not gpu_ids:
        raise ValueError(
            'predict_jobs_multi_gpu requires at least one GPU id in service')

    request_id = int(service.get('next_request_id', 1))
    service['next_request_id'] = request_id + 1

    shards = []
    start = 0
    for idx in range(len(gpu_ids)):
        end = int(round((idx + 1) * len(jobs) / len(gpu_ids)))
        shard = []
        for req_idx, job in enumerate(jobs[start:end], start=start):
            item = dict(job)
            item['__request_index__'] = int(req_idx)
            shard.append(item)
        if shard:
            shards.append(shard)
        start = end

    if not shards:
        return [], stats

    task_q = service['task_q']
    result_q = service['result_q']
    for shard in shards:
        task_q.put({
            'request_id': request_id,
            'jobs': shard,
        })

    outputs = [None] * len(jobs)
    pending = len(shards)
    while pending > 0:
        result = result_q.get()
        if int(result.get('request_id', -1)) != request_id:
            continue
        pending -= 1
        shard_stats = result.get('stats') or {}
        stats['read_s'] += float(shard_stats.get('read_s', 0.0))
        stats['read_data_s'] += float(shard_stats.get('read_data_s', 0.0))
        stats['read_mask_s'] += float(shard_stats.get('read_mask_s', 0.0))
        stats['prep_s'] += float(shard_stats.get('prep_s', 0.0))
        stats['infer_s'] += float(shard_stats.get('infer_s', 0.0))
        for item in result.get('outputs') or []:
            idx = int(item['request_index'])
            outputs[idx] = item['array']

    ordered = [arr for arr in outputs if arr is not None]
    return ordered, stats


class PredictionWorkerFailed(RuntimeError):
    """A prediction worker reported an error or died. The run fails."""


def _drain_results(result_q, workers, write_tile, total_tiles,
                   progress_interval_s, timeout_s=30.0, print_fn=print):
    """Consume worker results until every worker has said 'done'.

    Assembly is order-independent: each result carries its own output
    window, and windows are the tiles' non-overlapping cores. That is
    what lets readers deliver a shard's tiles in any order.

    Failure is loud. An 'error' message, a worker that is dead without
    'done', or a timeout with nobody left alive all raise; the old loop
    blocked on result_q.get() forever in each of those cases.
    """
    finished = 0
    done = 0
    start = time.monotonic()
    last_report = start
    total_read_s = total_infer_s = total_write_s = 0.0

    while finished < len(workers):
        try:
            result = result_q.get(timeout=timeout_s)
        except queue.Empty:
            alive = [w for w in workers if w.is_alive()]
            if len(alive) + finished < len(workers):
                raise PredictionWorkerFailed(
                    f"{len(workers) - len(alive) - finished} prediction "
                    "worker(s) exited without reporting completion")
            continue
        if result.get('error'):
            raise PredictionWorkerFailed(
                f"prediction worker (gpu {result.get('gpu_id')}) failed: "
                f"{result['error']}")
        if result.get('done'):
            finished += 1
            continue
        total_write_s += write_tile(int(result['row_off']),
                                    int(result['col_off']), result['array'])
        total_read_s += float(result.get('read_s', 0.0))
        total_infer_s += float(result.get('infer_s', 0.0))
        done += 1
        now = time.monotonic()
        if done == 1 or done == total_tiles \
                or (now - last_report) >= progress_interval_s:
            elapsed = max(now - start, 1e-9)
            rate = done / elapsed
            eta_s = (total_tiles - done) / rate if rate > 0 else float('inf')
            print_fn(
                f"Multi-GPU prediction {done}/{total_tiles} | "
                f"{done / max(total_tiles, 1):.1%} | {rate * 60:.1f} tiles/min"
                f" | ETA {_format_eta(eta_s)} | avg read "
                f"{total_read_s / max(done, 1):.3f}s infer "
                f"{total_infer_s / max(done, 1):.3f}s write "
                f"{total_write_s / max(done, 1):.3f}s",
                flush=True)
            last_report = now
    return {'read_s': total_read_s, 'infer_s': total_infer_s,
            'write_s': total_write_s, 'done': done}


def run_multi_gpu_prediction(
    model_path: str,
    input_raster: str,
    output_raster: str,
    tile_jobs,
    gpu_ids: list[int],
    config,
):
    if not gpu_ids:
        raise ValueError('run_multi_gpu_prediction requires at least one '
                         'accelerator id; use [None] for CPU')

    ctx = mp.get_context('spawn')
    result_q = ctx.Queue(maxsize=max(8, len(gpu_ids) * 8))

    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        layout = _resampling_layout((src.height, src.width), profile, config)
        all_jobs = list(_iter_tile_jobs(layout, config)) \
            if tile_jobs is None else list(tile_jobs)

    shards = []
    start = 0
    for idx in range(len(gpu_ids)):
        end = int(round((idx + 1) * len(all_jobs) / len(gpu_ids)))
        shards.append(all_jobs[start:end])
        start = end

    out_profile = IO.build_safe_prediction_profile(
        src_profile=profile,
        width=layout['out_width'],
        height=layout['out_height'],
        transform=layout['out_transform'],
        compress='DEFLATE' if getattr(config, 'compress_output', True)
        else None,
        dtype='uint8',
    )
    os.makedirs(os.path.dirname(output_raster) or '.', exist_ok=True)
    tmp_path = IO.atomic_tmp_path(output_raster)

    cfg_dict = dict(getattr(config, 'to_dict', lambda: {})())
    workers = []
    for shard, gpu_id in zip(shards, gpu_ids):
        p = ctx.Process(
            target=prediction_worker,
            args=(gpu_id, model_path, input_raster, shard, result_q, cfg_dict),
        )
        p.start()
        workers.append(p)

    total_tiles = len(all_jobs)
    progress_interval_s = float(getattr(config, 'progress_interval_s', 20.0))

    with rasterio.open(tmp_path, 'w', **out_profile) as dst:
        def write_tile(row_off, col_off, arr):
            write_h = min(arr.shape[0], layout['out_height'] - row_off)
            write_w = min(arr.shape[1], layout['out_width'] - col_off)
            t0 = time.perf_counter()
            dst.write(
                np.ascontiguousarray(arr[:write_h, :write_w], dtype=np.uint8),
                1, window=Window(col_off, row_off, write_w, write_h))
            return time.perf_counter() - t0

        try:
            _drain_results(result_q, workers, write_tile, total_tiles,
                           progress_interval_s)
        except BaseException:      # noqa: BLE001 -- terminate, never swallow
            # Not just PredictionWorkerFailed: a write_tile error, a
            # rasterio error, or a KeyboardInterrupt all used to skip
            # terminate() here, leaving workers blocked in `results.put`
            # on a pipe nobody drains -- multiprocessing's atexit join
            # then waits forever. Every failure must terminate live
            # workers before propagating.
            for p in workers:
                if p.is_alive():
                    p.terminate()
            raise

    for p in workers:
        p.join()
    IO.finalize_raster(tmp_path, output_raster)
    return out_profile
