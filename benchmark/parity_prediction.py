#!/usr/bin/env python
"""Gate (ii) for the reader pool: bit-identical output, throughput >=,
peak RSS <= main. Run on the T14 (1 GPU), once CPU-only, and on carrot.

    python benchmark/parity_prediction.py --base /path/to/main-checkout \
        --branch /path/to/branch-checkout --model <model> --ortho <tif> \
        --out /tmp/parity [--cpu]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import rasterio
import geopandas as gpd


def _poll_tree_rss(pid, peak, stop_event, interval_s=0.5):
    """Peak of the SUM of RSS across the process tree, sampled every
    ``interval_s`` while the child runs.

    `os.wait4`'s `ru_maxrss` is the peak of the LARGEST single process, not
    the total: on the T14, `main` runs prediction as one process while this
    branch splits it into a coordinator plus one worker per GPU, so
    `ru_maxrss` alone can look flat (or even improve) while total memory
    rose -- exactly the failure mode `rss_ok` exists to catch. `peak[0]` is
    a single-element list so this thread can write a result the caller
    reads after joining it.
    """
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return
    while not stop_event.is_set():
        try:
            procs = [root] + root.children(recursive=True)
            total = sum(p.memory_info().rss for p in procs
                        if p.is_running())
        except psutil.Error:
            # A child can exit between children() and memory_info(); skip
            # this sample rather than crash the poller.
            total = 0
        if total > peak[0]:
            peak[0] = total
        stop_event.wait(interval_s)


def run(checkout, model, ortho, outdir, cpu):
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "stem.tif"
    env = dict(os.environ)
    if cpu:
        env["WINMOL_ONNX_FORCE_CPU"] = "1"
    # Timestamp every stdout line as it ARRIVES: the progress line carries
    # only the cumulative average, and the #43 cliff is invisible in a
    # cumulative average until long after it happened.
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-u", "winmol_run.py", model, ortho,
         str(stem), str(outdir / "out"), "Nodes"],
        cwd=checkout, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    tree_peak = [0]      # bytes; written by the poller thread, read after join
    stop_poll = threading.Event()
    poller = threading.Thread(
        target=_poll_tree_rss, args=(proc.pid, tree_peak, stop_poll),
        daemon=True)
    poller.start()
    stamped = []                       # (seconds, done_tiles) per progress line
    log = open(outdir / "run.log", "w")
    for line in proc.stdout:
        log.write(line)
        log.flush()   # else run.log stays empty for minutes on a real run
        is_progress = "prediction" in line or "Written tile" in line
        if "tiles/min" in line and is_progress:
            try:
                done = int(line.split("|")[0].split()[-1].split("/")[0])
            except (ValueError, IndexError):
                continue
            stamped.append((time.perf_counter() - t0, done))
    log.close()
    # Ruling 1: peak RSS must be read per-child via os.wait4, not
    # RUSAGE_CHILDREN (which accumulates the max over ALL waited children,
    # so the second arm could never read lower than the first -- making
    # rss_ok vacuous). Do NOT call proc.wait() after this: the pid is
    # already reaped by wait4.
    _, status, ru = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    stop_poll.set()
    poller.join(timeout=2.0)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(
            f"{checkout}: exit {proc.returncode}, see {outdir}/run.log")
    # Ruling 2: ru_maxrss is KB on Linux, bytes on macOS -- normalise to MB.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    peak_rss_mb = ru.ru_maxrss / divisor
    peak_tree_rss_mb = tree_peak[0] / (1024 * 1024)
    # Ruling 4: GPKGs live under <outdir>/out*; compare the LAST (merged)
    # one, but let the caller notice a length mismatch instead of indexing
    # blindly into two lists of different shape.
    gpkgs = sorted(outdir.rglob("*.gpkg"))
    return {"wall_s": wall, "curve": stamped, "peak_rss_mb": peak_rss_mb,
            "peak_tree_rss_mb": peak_tree_rss_mb,
            "stem": stem, "gpkg": gpkgs}


def inst_rate(curve, lo, hi):
    """Instantaneous tiles/min over the slice of the run where done is in
    [lo, hi] of the total -- from successive progress lines, never from the
    cumulative average."""
    if not curve:
        return float("nan")
    total = curve[-1][1]
    pts = [(t, d) for t, d in curve if lo * total <= d <= hi * total]
    if len(pts) < 2:
        # A fast arm can cross the whole slice within one progress
        # interval (carrot: <10% of R13 in under one 20s tick) -- fall
        # back to the two lines nearest this edge instead of NaN-ing out.
        if len(curve) < 2:
            return float("nan")
        if lo == 0:
            pts = curve[:2]
        elif hi == 1:
            pts = curve[-2:]
        else:
            return float("nan")
    (t0, d0), (t1, d1) = pts[0], pts[-1]
    return 60.0 * (d1 - d0) / max(t1 - t0, 1e-9)


def cliff_ratio(curve):
    """last-10% rate / first-10% rate. ~1.0 is flat; the #43 collapse on
    the T14 measured ~0.3 (2288 -> 677/min on R13). NaN when either slice
    has too few points to judge (e.g. a tiny smoke-test input)."""
    first = inst_rate(curve, 0.0, 0.1)
    last = inst_rate(curve, 0.9, 1.0)
    if first != first or last != last or first == 0:   # NaN or zero guard
        return float("nan")
    return last / first


def same_raster(a, b):
    with rasterio.open(a) as ra, rasterio.open(b) as rb:
        if ra.shape != rb.shape or ra.count != rb.count:
            return False, "shape/count differ"
        diff = 0
        for _, win in ra.block_windows(1):
            block_a = ra.read(1, window=win)
            block_b = rb.read(1, window=win)
            diff += int(np.count_nonzero(block_a != block_b))
        return diff == 0, f"{diff} differing px"


def _row_multiset(gdf, geom_col, cols):
    """Sorted list of (wkb_bytes, *str(attr)) tuples, one per row.

    astype(str) per column makes NaN and mixed dtypes compare
    deterministically; sorting makes the comparison row-order-independent
    while still catching a changed or missing duplicate (unlike a
    unique-WKB-keyed merge, which real R13 output violates)."""
    columns = [list(gdf[geom_col].to_wkb())]
    columns += [gdf[c].astype(str).tolist() for c in cols]
    return sorted(zip(*columns))


def same_gpkg(a, b):
    """Multiset comparison of (geometry, attributes except stem_id).

    ``main`` renumbers stem_id and reorders rows run-to-run (measured
    2026-09-15: running unmodified main against itself on the same input,
    with byte-identical rasters, stems/nodes/vectors rows land in
    different positions and stem_id differs run-to-run). Real R13 output
    also has duplicate geometries (stems 1, nodes 122, vectors 59), which
    breaks a unique-WKB merge key. So each layer is compared as a sorted
    multiset of (geometry_wkb, *attributes-except-stem_id) tuples --
    order-independent, duplicate-safe, and stem_id-independent.
    """
    import pyogrio
    la = list(pyogrio.list_layers(a)[:, 0])
    lb = list(pyogrio.list_layers(b)[:, 0])
    if la != lb:
        return False, f"layer sets differ: {la} vs {lb}"
    for layer in la:
        ga = gpd.read_file(a, layer=layer, engine="pyogrio")
        gb = gpd.read_file(b, layer=layer, engine="pyogrio")
        if len(ga) != len(gb):
            return False, f"{layer}: {len(ga)} vs {len(gb)} rows"
        geom_col = ga.geometry.name
        cols = [c for c in ga.columns if c not in (geom_col, "stem_id")]
        rows_a = _row_multiset(ga, geom_col, cols)
        rows_b = _row_multiset(gb, geom_col, cols)
        if rows_a != rows_b:
            return False, f"{layer}: rows differ as a multiset"
    return True, "identical (geometry-keyed, stem_id excluded)"


def compare_gpkgs(base_gpkgs, br_gpkgs):
    # Ruling 4: report a length mismatch as a difference instead of
    # indexing into mismatched lists.
    if not base_gpkgs and not br_gpkgs:
        return True, "no gpkg"
    if len(base_gpkgs) != len(br_gpkgs):
        msg = (f"gpkg count differs: base={len(base_gpkgs)} "
               f"branch={len(br_gpkgs)}")
        return False, msg
    return same_gpkg(base_gpkgs[-1], br_gpkgs[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ortho", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    base = run(a.base, a.model, a.ortho, out / "base", a.cpu)
    br = run(a.branch, a.model, a.ortho, out / "branch", a.cpu)
    ok_r, why_r = same_raster(base["stem"], br["stem"])
    ok_g, why_g = compare_gpkgs(base["gpkg"], br["gpkg"])
    faster = br["wall_s"] <= base["wall_s"]
    # The gate compares TREE totals, not the single largest process: on the
    # T14, main is one process while the branch splits coordinator +
    # worker, so the largest-process number (peak_rss_mb) can pass while
    # total memory rose. Both are still reported below.
    leaner = br["peak_tree_rss_mb"] <= base["peak_tree_rss_mb"]
    # #43 gate: the pool must not cliff where main does not.
    base_cliff, br_cliff = cliff_ratio(base["curve"]), cliff_ratio(br["curve"])
    cliff_note = None
    if base_cliff != base_cliff or br_cliff != br_cliff:  # either is NaN
        # Too few progress lines to judge (e.g. a tiny smoke-test input) --
        # do not fail the gate on noise. The full-ortho run is where the
        # cliff criterion actually bites.
        flat = True
        cliff_note = (
            "cliff_ratio undefined (NaN) for at least one arm -- too few "
            "progress lines to judge; no_cliff_ok forced true. Re-run on "
            "a full ortho to exercise this gate.")
    else:
        flat = br_cliff >= base_cliff * 0.95
    report = {
        "raster": why_r,
        "gpkg": why_g,
        "base": {
            "wall_s": base["wall_s"],
            "peak_rss_mb": base["peak_rss_mb"],
            "peak_tree_rss_mb": base["peak_tree_rss_mb"],
            "first10_tpm": inst_rate(base["curve"], 0, .1),
            "last10_tpm": inst_rate(base["curve"], .9, 1),
            "cliff_ratio": base_cliff,
        },
        "branch": {
            "wall_s": br["wall_s"],
            "peak_rss_mb": br["peak_rss_mb"],
            "peak_tree_rss_mb": br["peak_tree_rss_mb"],
            "first10_tpm": inst_rate(br["curve"], 0, .1),
            "last10_tpm": inst_rate(br["curve"], .9, 1),
            "cliff_ratio": br_cliff,
        },
        "throughput_ok": faster,
        "rss_ok": leaner,
        "no_cliff_ok": flat,
    }
    if cliff_note:
        report["cliff_note"] = cliff_note
    print(json.dumps(report, indent=2))
    sys.exit(0 if (ok_r and ok_g and faster and leaner and flat) else 1)


if __name__ == "__main__":
    main()
