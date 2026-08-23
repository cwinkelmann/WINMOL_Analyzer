from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import List

import numpy as np
import rasterio
from rasterio.windows import Window

from classes.Config import Config
from utils import IO
from utils.Prediction import (_prepare_inference_batch, _resize_batch,
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


def _predict_batch(raw_tiles, raw_masks, model, config):
    # Follow the session-wide strategy, exactly as the stream loop does.
    # This used to force "native" because the workers loaded the model
    # unwrapped, which put the normalize + 1250^2 -> 512^2 resample of
    # every tile on the CPU: measured on 8xH100 as 1.974 s per tile of
    # "inference" for a model that runs in 2.97 ms, i.e. 106 tiles/min
    # against the stream's 8569. The workers now load the wrapped model,
    # so the resize happens on device and this hands it native uint8.
    tile_tensor, mask_resized = _prepare_inference_batch(
        raw_tiles, raw_masks, config)
    pred = model.predict_on_batch(tile_tensor)
    crop = config.overlap_pred // 2
    threshold = float(getattr(config, 'stem_binary_threshold', 0.5))
    cores = []
    for idx in range(pred.shape[0]):
        pred_core = pred[idx, crop:(config.img_width - crop),
                         crop:(config.img_width - crop), 0]
        mask_core = mask_resized[idx, crop:(config.img_width - crop),
                                 crop:(config.img_width - crop), 0] > 0.5
        cores.append(np.ascontiguousarray(
            ((pred_core >= threshold) & mask_core).astype(np.uint8)))
    return cores


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


def prediction_worker(
    gpu_id: int,
    model_path: str,
    input_raster: str,
    jobs: List[dict],
    results,
    config_dict: dict,
):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    from utils.IO import load_model_from_path

    cfg = _config_from_dict(config_dict)
    # Wrapped, so the normalize + resize run on THIS worker's GPU instead
    # of on the CPU inside the timed inference block.
    model = load_model_from_path(model_path, cfg)
    out_size = _graph_out_size(cfg)
    batch_size = max(1, int(getattr(cfg, 'prediction_batch_size', None)
                            or getattr(cfg, 'prediction_batch_gpu', 4)))

    with rasterio.open(input_raster) as src:
        indexes = list(range(1, min(cfg.n_channels, src.count) + 1))
        for batch_jobs in _group_jobs(jobs, batch_size):
            raw_tiles, raw_masks, read_stats = \
                _read_batch_jobs(src, indexes, batch_jobs, out_size)
            infer0 = time.perf_counter()
            pred_cores = _predict_batch(raw_tiles, raw_masks, model, cfg)
            infer_s = time.perf_counter() - infer0
            for job, pred_core in zip(batch_jobs, pred_cores):
                results.put({
                    'row_off': job['dst_row'],
                    'col_off': job['dst_col'],
                    'array': pred_core,
                    'read_s': read_stats['read_s'] / max(len(batch_jobs), 1),
                    'infer_s': infer_s / max(len(batch_jobs), 1),
                })
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
                pred_cores = _predict_batch(raw_tiles, raw_masks, model, cfg)
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


def run_multi_gpu_prediction(
    model_path: str,
    input_raster: str,
    output_raster: str,
    tile_jobs,
    gpu_ids: list[int],
    config,
):
    if not gpu_ids:
        raise ValueError('multi_gpu_prediction requires at least one GPU id')

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
    done = 0
    finished = 0
    start = time.monotonic()
    last_report = start
    total_read_s = 0.0
    total_infer_s = 0.0
    total_write_s = 0.0
    progress_interval_s = float(getattr(config, 'progress_interval_s', 20.0))

    with rasterio.open(tmp_path, 'w', **out_profile) as dst:
        while finished < len(workers):
            result = result_q.get()
            if result.get('done'):
                finished += 1
                continue
            arr = result['array']
            row_off = int(result['row_off'])
            col_off = int(result['col_off'])
            write_h = min(arr.shape[0], layout['out_height'] - row_off)
            write_w = min(arr.shape[1], layout['out_width'] - col_off)
            t0 = time.perf_counter()
            dst.write(
                np.ascontiguousarray(arr[:write_h, :write_w],
                                     dtype=np.uint8), 1,
                window=Window(col_off, row_off, write_w, write_h))
            total_write_s += time.perf_counter() - t0
            total_read_s += float(result.get('read_s', 0.0))
            total_infer_s += float(result.get('infer_s', 0.0))
            done += 1
            now = time.monotonic()
            if (
                done == 1 or done == total_tiles
                or (now - last_report) >= progress_interval_s
            ):
                elapsed = max(now - start, 1e-9)
                rate = done / elapsed
                eta_s = \
                    (total_tiles - done) / rate if rate > 0 else float('inf')
                print(
                    f"Multi-GPU prediction {done}/{total_tiles} | "
                    f"{done / total_tiles:.1%} | {rate * 60:.1f} tiles/min"
                    f" | ETA {_format_eta(eta_s)} | avg read "
                    f"{total_read_s / max(done, 1):.3f}s infer "
                    f"{total_infer_s / max(done, 1):.3f}s write "
                    f"{total_write_s / max(done, 1):.3f}s",
                    flush=True, )
                last_report = now

    for p in workers:
        p.join()
    IO.finalize_raster(tmp_path, output_raster)
    return out_profile
