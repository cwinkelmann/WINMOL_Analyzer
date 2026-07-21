from __future__ import annotations

import contextlib
import io
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from rasterio.windows import Window

from classes.Config import Config
from utils.IO import (
    build_safe_prediction_profile,
    load_raster_window_with_profile,
    load_stem_map,
    write_all_layers_to_gpkg,
    write_stems_to_gpkg,
    write_tile_raster,
)
import utils.Quantification as Quant
import utils.Skeletonization as Skel
import utils.Vectorization as Vec


def _clone_config(config, **updates):
    cfg = Config()
    for key, value in getattr(config, 'to_dict', lambda: {})().items():
        try:
            setattr(cfg, key, value)
        except Exception:
            pass
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def export_tile_results(stems, profile, process_type: str, output_prefix: str):
    if process_type == 'Trees':
        return write_stems_to_gpkg(stems, profile, output_prefix)
    return write_all_layers_to_gpkg(stems, profile, output_prefix)


def _format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds == float('inf'):
        return 'n/a'
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:d}h {minutes:02d}m'
    if minutes > 0:
        return f'{minutes:d}m {secs:02d}s'
    return f'{secs:d}s'


def _vector_result(
    tile_label,
    output_path,
    fg_count,
    segment_count,
    stem_count,
    timings,
):
    return {
        'tile_label': tile_label,
        'gpkg_path': output_path,
        'fg_count': int(fg_count or 0),
        'segment_count': int(segment_count or 0),
        'stem_count': int(stem_count or 0),
        'timings': dict(timings),
    }


def _vector_summary(
    tile_label,
    fg_count,
    segment_count,
    stem_count,
    timings,
    output_path,
):
    print(
        f'VECTOR TILE {tile_label} | fg {fg_count} | segments '
        f'{segment_count} | stems {stem_count} | skel '
        f'{timings["skel_s"]:.3f}s restore {timings["restore_s"]:.3f}s '
        f'build {timings["build_s"]:.3f}s connect '
        f'{timings["connect_s"]:.3f}s quant '
        f'{timings["quant_s"]:.3f}s write {timings["write_s"]:.3f}s '
        f'| total {timings["total_s"]:.3f}s | output '
        f'{os.path.basename(output_path) if output_path else "none"}',
        flush=True,
    )


def _run_vector_pipeline(
    pred,
    profile,
    config,
    process_type: str,
    output_prefix: str,
    tile_label: str,
):
    fg_count = int(np.count_nonzero(pred))
    if fg_count <= 0:
        return None

    timings = {
        'skel_s': 0.0,
        'restore_s': 0.0,
        'build_s': 0.0,
        'connect_s': 0.0,
        'quant_s': 0.0,
        'write_s': 0.0,
        'total_s': 0.0,
    }
    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    segments = Skel.find_segments(pred, config, profile)
    timings['skel_s'] = time.perf_counter() - t0
    if not segments:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(
                tile_label,
                fg_count,
                0,
                0,
                timings,
                None,
            )
        return _vector_result(tile_label, None, fg_count, 0, 0, timings)
    segment_count = len(segments)

    t0 = time.perf_counter()
    segments = Vec.restore_geoinformation(segments, config, profile)
    timings['restore_s'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stems = Vec.build_stem_parts(segments)
    timings['build_s'] = time.perf_counter() - t0
    if not stems:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(
                tile_label,
                fg_count,
                segment_count,
                0,
                timings,
                None,
            )
        return _vector_result(
            tile_label,
            None,
            fg_count,
            segment_count,
            0,
            timings,
        )

    t0 = time.perf_counter()
    stems = Vec.connect_stems(stems, config)
    timings['connect_s'] = time.perf_counter() - t0
    if not stems:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(
                tile_label,
                fg_count,
                segment_count,
                0,
                timings,
                None,
            )
        return _vector_result(
            tile_label,
            None,
            fg_count,
            segment_count,
            0,
            timings,
        )

    Vec.rebuild_endnodes_from_stems(stems)

    t0 = time.perf_counter()
    stems = Quant.quantify_stems(stems, pred, profile, config=config)
    timings['quant_s'] = time.perf_counter() - t0
    if not stems:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(
                tile_label,
                fg_count,
                segment_count,
                0,
                timings,
                None,
            )
        return _vector_result(
            tile_label,
            None,
            fg_count,
            segment_count,
            0,
            timings,
        )
    stem_count = len(stems)

    t0 = time.perf_counter()
    output_path = export_tile_results(
        stems,
        profile,
        process_type,
        output_prefix,
    )
    timings['write_s'] = time.perf_counter() - t0
    timings['total_s'] = time.perf_counter() - total_t0
    output_exists = bool(output_path) and os.path.exists(output_path)
    print(
        f'VECTOR TILE {tile_label} | output_exists {output_exists} | path '
        f'{output_path}',
        flush=True,
    )

    if bool(getattr(config, 'vector_summary_log', True)):
        _vector_summary(
            tile_label,
            fg_count,
            segment_count,
            stem_count,
            timings,
            output_path,
        )
    return _vector_result(
        tile_label,
        output_path,
        fg_count,
        segment_count,
        stem_count,
        timings,
    )


def _run_with_debug_control(fn, config, *args, **kwargs):
    vector_mode = str(
        getattr(config, 'vector_mode', 'tiled') or 'tiled'
    ).lower()
    vector_debug = bool(getattr(config, 'vector_debug', False))
    if vector_mode != 'tiled' or vector_debug:
        return fn(*args, **kwargs)

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            with contextlib.redirect_stderr(buffer):
                return fn(*args, **kwargs)
    except Exception:
        captured = buffer.getvalue().strip()
        if captured:
            print('VECTOR TILE DEBUG DUMP START', flush=True)
            print(captured, flush=True)
            print('VECTOR TILE DEBUG DUMP END', flush=True)
        raise


def process_prediction_array_to_gpkg(
    pred_arr,
    profile,
    config_dict,
    process_type: str,
    output_prefix: str,
):
    config = Config()
    for key, value in (config_dict or {}).items():
        try:
            setattr(config, key, value)
        except Exception:
            pass
    config.cpu_workers = 1
    config.vector_tile_workers = 1

    pred = np.asarray(pred_arr)
    if pred.size == 0 or not np.any(pred >= 1):
        return None

    tile_label = os.path.basename(output_prefix)
    return _run_with_debug_control(
        _run_vector_pipeline,
        config,
        pred,
        profile,
        config,
        process_type,
        output_prefix,
        tile_label,
    )


def process_prediction_tile(
    pred_tile_path: str,
    config,
    process_type: str,
    output_prefix: str,
):
    pred, profile = load_stem_map(pred_tile_path)
    pred_arr = np.asarray(pred)
    if pred_arr.size == 0 or not np.any(pred_arr >= 1):
        return None
    tile_label = os.path.splitext(os.path.basename(pred_tile_path))[0]
    try:
        return _run_with_debug_control(
            _run_vector_pipeline,
            config,
            pred,
            profile,
            config,
            process_type,
            output_prefix,
            tile_label,
        )
    except Exception as exc:
        raise RuntimeError(
            f'Vector tile failed: {tile_label} ({pred_tile_path}) | '
            f'{type(exc).__name__}: {exc}'
        ) from exc


def make_tile_spec(src_path: str, tile_id: str, window) -> dict:
    """Plain-dict task describing one tile window of the stem-map raster.

    Only picklable builtins cross the pool boundary (macOS/Windows use
    spawn); the worker rebuilds the rasterio Window itself.
    """
    return {
        'src_path': str(src_path),
        'tile_id': str(tile_id),
        'col_off': int(window.col_off),
        'row_off': int(window.row_off),
        'width': int(window.width),
        'height': int(window.height),
    }


def _tile_handoff(pred_arr, tile_profile):
    """In-memory equivalent of write_tile_raster -> load_stem_map.

    The legacy handoff wrote every tile as a GeoTIFF and re-read it in the
    worker. That round trip was not a no-op: build_safe_prediction_profile
    defaults to dtype='float32', so the pipeline always received float32
    pixels (source uint8, truncated via astype(uint8), cast on write) and
    the REWRITTEN tile profile — not the windowed source profile. Mirror
    both exactly so skipping the re-read is results-identical.
    """
    pred = np.asarray(pred_arr).astype(np.uint8).astype(np.float32)
    profile = build_safe_prediction_profile(
        tile_profile,
        width=pred.shape[1],
        height=pred.shape[0],
        transform=tile_profile['transform'],
        compress=None,
    )
    # Align with what rasterio reports when re-reading the written tile.
    profile.pop('BIGTIFF', None)
    profile['nodata'] = None
    profile['interleave'] = 'band'
    return pred, profile


def process_prediction_tile_spec(
    spec: dict,
    config,
    process_type: str,
    output_prefix: str,
):
    """Read one tile window from the source stem map and vectorize it.

    Replaces the parent-side prepare loop (serial windowed read +
    foreground scan + tile GeoTIFF write) plus the worker-side re-read and
    re-scan: the worker now reads its own window, checks foreground ONCE,
    and hands the array straight to the pipeline. The tile GeoTIFF is
    still written for every foreground tile — the merge stage derives its
    seam-dedup keep-region from the tile raster's bounds, and on-disk
    tiles are the failure-inspection/resume contract
    (docs/CODE_REVIEW_2.md A-15) — but the write now happens inside the
    parallel worker and nothing re-reads it.
    """
    tile_label = str(spec.get('tile_id') or os.path.basename(output_prefix))
    window = Window(
        spec['col_off'], spec['row_off'], spec['width'], spec['height'])
    try:
        pred, tile_profile = load_raster_window_with_profile(
            spec['src_path'], window)
    except Exception as exc:
        raise RuntimeError(
            f'Vector tile failed: {tile_label} '
            f'(reading {spec["src_path"]}) | '
            f'{type(exc).__name__}: {exc}'
        ) from exc
    pred_arr = np.asarray(pred)
    if pred_arr.size == 0 or not (pred_arr >= 1).any():
        return None

    tile_raster_path = f'{output_prefix}_roi_stem_map.tif'
    write_tile_raster(pred_arr, tile_profile, tile_raster_path)

    pred_f32, handoff_profile = _tile_handoff(pred_arr, tile_profile)
    try:
        return _run_with_debug_control(
            _run_vector_pipeline,
            config,
            pred_f32,
            handoff_profile,
            config,
            process_type,
            output_prefix,
            tile_label,
        )
    except Exception as exc:
        raise RuntimeError(
            f'Vector tile failed: {tile_label} ({tile_raster_path}) | '
            f'{type(exc).__name__}: {exc}'
        ) from exc


def _process_prediction_tile_star(args):
    if isinstance(args[0], dict):
        return process_prediction_tile_spec(*args)
    return process_prediction_tile(*args)


def _init_progress_totals():
    return {
        'fg_count': 0,
        'segment_count': 0,
        'stem_count': 0,
        'empty_tiles': 0,
        'no_output_tiles': 0,
        'written_tiles': 0,
        'skel_s': 0.0,
        'restore_s': 0.0,
        'build_s': 0.0,
        'connect_s': 0.0,
        'quant_s': 0.0,
        'write_s': 0.0,
        'total_s': 0.0,
        'timed_tiles': 0,
    }


def _update_progress_totals(totals, result):
    if result is None:
        totals['empty_tiles'] += 1
        return

    totals['fg_count'] += int(result.get('fg_count', 0) or 0)
    totals['segment_count'] += int(result.get('segment_count', 0) or 0)
    totals['stem_count'] += int(result.get('stem_count', 0) or 0)

    if result.get('gpkg_path'):
        totals['written_tiles'] += 1
    else:
        totals['no_output_tiles'] += 1

    timings = result.get('timings') or {}
    for key in (
        'skel_s',
        'restore_s',
        'build_s',
        'connect_s',
        'quant_s',
        'write_s',
        'total_s',
    ):
        totals[key] += float(timings.get(key, 0.0) or 0.0)
    totals['timed_tiles'] += 1


def _print_vector_progress(done, total, start, totals):
    now = time.monotonic()
    elapsed = max(now - start, 1e-9)
    rate = done / elapsed
    eta_s = (total - done) / rate if rate > 0 else float('inf')
    timed_tiles = max(totals['timed_tiles'], 1)
    avg_total = totals['total_s'] / timed_tiles
    avg_quant = totals['quant_s'] / timed_tiles
    avg_connect = totals['connect_s'] / timed_tiles
    print(
        f'Vector tiles {done}/{total} | {done / total:.1%} | '
        f'{rate * 60:.1f} tiles/min | ETA {_format_eta(eta_s)} | wrote '
        f'{totals["written_tiles"]} | empty {totals["empty_tiles"]} | '
        f'no_output {totals["no_output_tiles"]} | avg total '
        f'{avg_total:.3f}s quant {avg_quant:.3f}s connect '
        f'{avg_connect:.3f}s',
        flush=True,
    )


def _print_vector_summary(
    total,
    totals,
    tile_workers,
    inner_workers,
    start,
):
    elapsed = max(time.monotonic() - start, 1e-9)
    timed_tiles = max(totals['timed_tiles'], 1)
    print('')
    print('VECTOR SUMMARY')
    print(f'Tiles queued:          {total}')
    print(f'Tile workers:          {tile_workers}')
    print(f'Inner workers:         {inner_workers}')
    print(f'Tiles with foreground: {total - totals["empty_tiles"]}')
    print(f'Tiles written:         {totals["written_tiles"]}')
    print(f'Tiles without output:  {totals["no_output_tiles"]}')
    print(f'Total segments:        {totals["segment_count"]}')
    print(f'Total stems:           {totals["stem_count"]}')
    print(f'Elapsed:               {elapsed:.3f}s')
    print(
        f'Avg timed tile:        {totals["total_s"] / timed_tiles:.3f}s '
        f'(quant {totals["quant_s"] / timed_tiles:.3f}s, connect '
        f'{totals["connect_s"] / timed_tiles:.3f}s)',
    )


def _tile_task_name(tile) -> str:
    if isinstance(tile, dict):
        return str(tile['tile_id'])
    name = os.path.splitext(os.path.basename(tile))[0]
    return name.replace('_roi_stem_map', '')


def process_prediction_tiles(
    pred_tiles: list,
    config,
    process_type: str,
    output_dir: str,
    cpu_workers: int,
):
    """Vectorize prediction tiles: paths (legacy) or window specs.

    Each item of ``pred_tiles`` is either a tile GeoTIFF path (legacy
    callers and tests) or a make_tile_spec() dict, in which case the
    WORKER reads its window from the source raster directly.

    Tile workers run in a ProcessPoolExecutor — its workers are
    non-daemonic, so (unlike mp.Pool workers) they may host the
    skeletonization refine pool — and the CPU budget COMPOSES:
    tile_workers * inner_workers <= total_workers.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_workers = max(
        1,
        int(cpu_workers or getattr(config, 'cpu_workers', 1) or 1),
    )
    configured_tile_workers = max(
        1,
        int(getattr(config, 'vector_tile_workers', 1) or 1),
    )
    tile_workers = min(
        configured_tile_workers,
        total_workers,
        len(pred_tiles),
    )
    if tile_workers > 1:
        inner_workers = max(1, total_workers // tile_workers)
    else:
        inner_workers = total_workers
    progress_interval_s = float(getattr(config, 'progress_interval_s', 60.0))

    tasks = []
    for tile in pred_tiles:
        output_prefix = os.path.join(output_dir, _tile_task_name(tile))
        tile_cfg = _clone_config(
            config,
            cpu_workers=inner_workers,
            vector_tile_workers=1,
        )
        tasks.append((tile, tile_cfg, process_type, output_prefix))

    if not tasks:
        print('Vector tiles 0/0 | no foreground tiles queued', flush=True)
        return []

    print(
        f'Running vector stage on {len(tasks)} tile(s) | tile_workers '
        f'{tile_workers} | inner_workers {inner_workers}',
        flush=True,
    )

    start = time.monotonic()
    last_report = start
    results = []
    totals = _init_progress_totals()

    if tile_workers <= 1 or len(tasks) <= 1:
        for idx, task in enumerate(tasks, start=1):
            result = _process_prediction_tile_star(task)
            results.append(result)
            _update_progress_totals(totals, result)
            now = time.monotonic()
            if idx == len(tasks) or (now - last_report) >= progress_interval_s:
                _print_vector_progress(idx, len(tasks), start, totals)
                last_report = now
        _print_vector_summary(
            len(tasks),
            totals,
            tile_workers,
            inner_workers,
            start,
        )
        return results

    with ProcessPoolExecutor(max_workers=tile_workers) as executor:
        futures = [
            executor.submit(_process_prediction_tile_star, task)
            for task in tasks
        ]
        try:
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                _update_progress_totals(totals, result)
                now = time.monotonic()
                if (
                    idx == len(tasks)
                    or (now - last_report) >= progress_interval_s
                ):
                    _print_vector_progress(idx, len(tasks), start, totals)
                    last_report = now
        except Exception:
            # Completed tiles stay on disk for inspection/resume; stop
            # handing out new ones. Running tiles finish on shutdown.
            for future in futures:
                future.cancel()
            raise

    _print_vector_summary(
        len(tasks),
        totals,
        tile_workers,
        inner_workers,
        start,
    )
    return results
