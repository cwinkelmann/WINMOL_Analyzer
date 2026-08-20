from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
import os
import time

import numpy as np

from classes.Config import Config
from utils.IO import (
    load_stem_map,
    write_all_layers_to_gpkg,
    write_stems_to_gpkg,
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


def _process_prediction_tile_star(args):
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


def process_prediction_tiles(
    pred_tile_paths: list[str],
    config,
    process_type: str,
    output_dir: str,
    cpu_workers: int,
):
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
        len(pred_tile_paths),
    )
    inner_workers = 1 if tile_workers > 1 else total_workers
    progress_interval_s = float(getattr(config, 'progress_interval_s', 60.0))

    tasks = []
    for pred_tile_path in pred_tile_paths:
        name = os.path.splitext(os.path.basename(pred_tile_path))[0]
        name = name.replace('_roi_stem_map', '')
        output_prefix = os.path.join(output_dir, name)
        tile_cfg = _clone_config(
            config,
            cpu_workers=inner_workers,
            vector_tile_workers=1,
        )
        tasks.append((pred_tile_path, tile_cfg, process_type, output_prefix))

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

    with mp.Pool(tile_workers) as pool:
        for idx, result in enumerate(
            pool.imap_unordered(_process_prediction_tile_star, tasks),
            start=1,
        ):
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
