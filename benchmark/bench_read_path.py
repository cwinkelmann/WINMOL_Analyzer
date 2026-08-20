#!/usr/bin/env python
"""Isolate the read-path slowdown behind issue #43.

Issue #43 reports throughput collapsing from ~3650 to ~288 tiles/min on a
99k-tile ortho. Its own numbers already exonerate inference:

    avg read 0.029s prep 0.005s infer 0.011s write 0.000s   (tile 3660)
    avg read 0.613s prep 0.006s infer 0.011s write 0.001s   (tile 20824)

prep, infer and write are CONSTANT; only `read` moves, by ~21x, and the
producer queue goes from "100% full" to "0% full". So the question is not
"did inference get slower" -- it demonstrably did not -- but "why did the
GDAL read get slower as the run progressed".

Two hypotheses, and they are distinguishable:

  POSITIONAL  later tiles are intrinsically more expensive to read (file
              layout: missing/partial overviews, compression, block size,
              a mask band that only covers part of the raster).
  TEMPORAL    the same tiles are cheap early and expensive late (OS page
              cache exhaustion, disk/network contention with the growing
              output raster, GDAL block-cache thrash).

The probes below tell them apart:

  --probe layout      what the file actually is (overviews, blocks, mask)
  --probe positional  sample windows from the START, MIDDLE and END of the
                      raster, interleaved, in one pass. If the END samples
                      are slow from the very first measurement, it is
                      positional. If all three are equally fast, it is not.
  --probe temporal    replay the real job order and report read time per
                      bucket of tiles. Reproduces the reported curve.
  --probe modes       the CHANGED read (GDAL out_shape + cubic, what the
                      reimplementation does) vs the ORIGINAL native read
                      plus a downstream resize, on identical windows.

Nothing here needs a GPU, a model, TensorFlow or onnxruntime -- it is the
read path alone, so a full answer costs minutes instead of the ~5 hours a
paired end-to-end run of the reported ortho would take.

    python benchmark/bench_read_path.py --ortho <big.tif> --probe layout
    python benchmark/bench_read_path.py --ortho <big.tif> --probe positional
    python benchmark/bench_read_path.py --ortho <big.tif> --probe temporal \
        --tiles 4000
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

# Matches classes/Config.py for the reported run.
IMG_WIDTH = 512
OVERLAP_PRED = 8
TILE_SIZE_M = 15.0


def build_layout(src, tile_size_m=TILE_SIZE_M):
    """The pipeline's tile geometry, from the raster's own transform.

    Mirrors utils/Prediction._iter_tile_jobs: windows of
    (px_per_tile - 1) stepping by (px_per_tile - overlap_img), row-major.
    """
    px_size = abs(src.transform.a)
    px_per_tile = tile_size_m / px_size
    inner = IMG_WIDTH - OVERLAP_PRED
    # overlap in source pixels, mirroring the model-grid overlap
    overlap_img = px_per_tile * (OVERLAP_PRED / float(IMG_WIDTH))
    step = px_per_tile - overlap_img
    x_tiles = max(1, int(np.ceil((src.width - px_per_tile) / step)) + 1)
    y_tiles = max(1, int(np.ceil((src.height - px_per_tile) / step)) + 1)
    return {
        "px_size": px_size,
        "px_per_tile": px_per_tile,
        "src_w": max(1, int(px_per_tile) - 1),
        "step": step,
        "x_tiles": x_tiles,
        "y_tiles": y_tiles,
        "out": inner + OVERLAP_PRED,   # 512, the model grid
        "total": x_tiles * y_tiles,
    }


def jobs_for(layout):
    """Row-major windows, the same order the producers are handed."""
    for i in range(layout["y_tiles"]):
        row = int(np.floor(i * layout["step"]))
        for j in range(layout["x_tiles"]):
            col = int(np.floor(j * layout["step"]))
            yield col, row, layout["src_w"], layout["src_w"]


def read_changed(src, indexes, win, out):
    """What the reimplementation does: resample inside the GDAL read."""
    tile = src.read(indexes, window=win, out_shape=(len(indexes), out, out),
                    resampling=Resampling.cubic, boundless=True,
                    fill_value=0).transpose(1, 2, 0)
    mask = src.read_masks(1, window=win, out_shape=(out, out),
                          resampling=Resampling.nearest, boundless=True) > 0
    return tile, mask


def read_native(src, indexes, win, out):
    """What the original did: native-resolution read, resize downstream."""
    tile = src.read(indexes, window=win, boundless=True,
                    fill_value=0).transpose(1, 2, 0)
    mask = src.read_masks(1, window=win, boundless=True) > 0
    return tile, mask


def read_changed_bounded(src, indexes, win, out):
    """The changed read WITHOUT boundless -- boundless routes through a
    VRT wrapper that can bypass overviews entirely."""
    tile = src.read(indexes, window=win, out_shape=(len(indexes), out, out),
                    resampling=Resampling.cubic).transpose(1, 2, 0)
    return tile, None


def read_no_mask(src, indexes, win, out):
    """The changed read without read_masks -- read_masks is a SECOND pass
    over the same window, so a dataset with a real mask band doubles the
    I/O per tile."""
    tile = src.read(indexes, window=win, out_shape=(len(indexes), out, out),
                    resampling=Resampling.cubic, boundless=True,
                    fill_value=0).transpose(1, 2, 0)
    return tile, None


MODES = {
    "changed": read_changed,
    "native": read_native,
    "changed-no-boundless": read_changed_bounded,
    "changed-no-mask": read_no_mask,
}


def probe_layout(path):
    """What the file IS. Most read pathologies are visible from here."""
    with rasterio.open(path) as src:
        print(f"path        {path}")
        print(f"size        {src.width} x {src.height} px, "
              f"{src.count} bands, {src.dtypes[0]}")
        print(f"pixel       {abs(src.transform.a):.4f} m")
        prof = src.profile
        print(f"driver      {prof.get('driver')}")
        print(f"compress    {prof.get('compress')}")
        print(f"tiled       {prof.get('tiled')}  "
              f"block {prof.get('blockxsize')}x{prof.get('blockysize')}")
        print(f"interleave  {prof.get('interleave')}")
        ov = src.overviews(1)
        ov_txt = ov if ov else "NONE  <-- every read is full-resolution"
        print(f"overviews   {ov_txt}")
        print(f"mask flags  {[mf[0].name for mf in src.mask_flag_enums]}")
        lay = build_layout(src)
        print(f"\nlayout      src {lay['src_w']}x{lay['src_w']} -> "
              f"out {lay['out']}x{lay['out']}")
        print(f"tiles       {lay['x_tiles']} x {lay['y_tiles']} = "
              f"{lay['total']}")
        # The decisive ratio: how much source data per model tile.
        ratio = (lay["src_w"] ** 2) / float(lay["out"] ** 2)
        print(f"downsample  {ratio:.1f}x source px per model px")
        if not ov:
            print("\nNOTE: with no overviews, out_shape cannot read a "
                  "cheaper level -- GDAL reads every source pixel and "
                  "downsamples in RAM.")


def timed(fn, *a):
    t0 = time.perf_counter()
    fn(*a)
    return time.perf_counter() - t0


def probe_positional(path, mode, samples):
    """START / MIDDLE / END windows, interleaved in ONE pass.

    Interleaving is the whole point: if the END group is slow from the
    first round-trip, the cost is positional (file layout). If all three
    groups track each other and drift up together, it is temporal.
    """
    read = MODES[mode]
    with rasterio.open(path) as src:
        lay = build_layout(src)
        alljobs = list(jobs_for(lay))
        n = len(alljobs)
        groups = {
            "start": alljobs[:samples],
            "middle": alljobs[n // 2: n // 2 + samples],
            "end": alljobs[-samples:],
        }
        indexes = list(range(1, min(3, src.count) + 1))
        out = lay["out"]
        times = {k: [] for k in groups}
        print(f"mode={mode}  {samples} windows per group, interleaved\n")
        print(f"{'round':>5}  {'start':>9}  {'middle':>9}  {'end':>9}")
        for i in range(samples):
            row = []
            for key in ("start", "middle", "end"):
                col, r, w, h = groups[key][i]
                dt = timed(read, src, indexes, Window(col, r, w, h), out)
                times[key].append(dt)
                row.append(dt)
            print(f"{i:>5}  {row[0]:>8.3f}s  {row[1]:>8.3f}s  "
                  f"{row[2]:>8.3f}s")
        print()
        for key in ("start", "middle", "end"):
            v = times[key]
            print(f"{key:>7}  median {statistics.median(v):.3f}s   "
                  f"mean {statistics.mean(v):.3f}s   max {max(v):.3f}s")
        med = {k: statistics.median(v) for k, v in times.items()}
        spread = max(med.values()) / max(1e-9, min(med.values()))
        print(f"\npositional spread (max/min median): {spread:.2f}x")
        print("  ~1x  -> NOT positional; look at the temporal probe.")
        print("  >2x  -> positional: later tiles are intrinsically dearer.")


def probe_temporal(path, mode, tiles, bucket):
    """Replay the real job order; report read time per bucket.

    This is the reported curve. A rising series with a flat positional
    probe means the file is fine and the environment is degrading --
    page cache, disk contention, or GDAL block-cache thrash.
    """
    read = MODES[mode]
    with rasterio.open(path) as src:
        lay = build_layout(src)
        indexes = list(range(1, min(3, src.count) + 1))
        out = lay["out"]
        print(f"mode={mode}  replaying {tiles} of {lay['total']} tiles "
              f"in job order, buckets of {bucket}\n")
        print(f"{'tiles':>12}  {'mean read':>10}  {'tiles/min':>10}")
        buf, done, t_start = [], 0, time.perf_counter()
        for col, r, w, h in jobs_for(lay):
            if done >= tiles:
                break
            buf.append(timed(read, src, indexes, Window(col, r, w, h), out))
            done += 1
            if len(buf) >= bucket:
                m = statistics.mean(buf)
                print(f"{done - bucket + 1:>5}-{done:<6}  {m:>9.3f}s  "
                      f"{60.0 / max(m, 1e-9):>10.0f}")
                buf = []
        total = time.perf_counter() - t_start
        print(f"\n{done} tiles in {total:.1f}s "
              f"({60.0 * done / max(total, 1e-9):.0f} tiles/min overall)")


def probe_modes(path, tiles):
    """Every read strategy, on DISJOINT windows spread across the raster.

    Two traps this avoids, both of which made an earlier version of this
    probe report nonsense (one mode at 0.000s):

    * sharing windows between modes lets the first mode warm GDAL's block
      cache and every later mode read from RAM;
    * taking the FIRST n windows samples only the top-left corner, which
      on an ortho is usually empty background -- uniform JPEG blocks
      decode far faster than real imagery.

    So each mode gets its own evenly-spaced stride through the full job
    list, interleaved with the others.
    """
    modes = sorted(MODES)
    with rasterio.open(path) as src:
        lay = build_layout(src)
        alljobs = list(jobs_for(lay))
        # One shared stride, then deal windows round-robin to the modes:
        # same spatial distribution for each, no window reused.
        need = tiles * len(modes)
        step = max(1, len(alljobs) // max(1, need))
        picked = alljobs[::step][:need]
        indexes = list(range(1, min(3, src.count) + 1))
        out = lay["out"]
        print(f"{tiles} disjoint windows per mode, strided across all "
              f"{len(alljobs)} tiles\n")
        times = {m: [] for m in modes}
        for i, (c, r, w, h) in enumerate(picked):
            name = modes[i % len(modes)]
            times[name].append(
                timed(MODES[name], src, indexes, Window(c, r, w, h), out))
        print(f"{'mode':>22}  {'median':>9}  {'mean':>9}  {'tiles/min':>10}")
        for name in modes:
            v = times[name]
            med = statistics.median(v)
            print(f"{name:>22}  {med:>8.3f}s  {statistics.mean(v):>8.3f}s  "
                  f"{60.0 / max(med, 1e-9):>10.0f}")


def probe_contended(path, tiles, bucket, aligned=False, compress=True):
    """Read AND write, the way a real run does. The reproduction.

    predict_stream_to_raster writes 504x504 cores at dst offsets 4, 508,
    1012 ... into an output whose blocks are 512x512 and DEFLATE
    compressed. Every write therefore straddles four compressed blocks,
    so GDAL must read-modify-write: decompress the block, patch it,
    recompress. While the output still fits in page cache that is cheap;
    once it does not, each partial write costs real disk I/O -- which
    competes with the very reads that feed the queue.

    ``--aligned`` writes at 512-aligned offsets instead (same volume, no
    straddle) and ``--no-compress`` drops DEFLATE, so the two suspected
    ingredients can be removed one at a time.
    """
    import tempfile
    read = MODES["changed"]
    with rasterio.open(path) as src:
        lay = build_layout(src)
        core = IMG_WIDTH - OVERLAP_PRED
        prof = {
            "driver": "GTiff", "dtype": "uint8", "count": 1,
            "width": lay["x_tiles"] * core + IMG_WIDTH,
            "height": lay["y_tiles"] * core + IMG_WIDTH,
            "crs": src.crs, "transform": src.transform,
            "tiled": True, "blockxsize": 512, "blockysize": 512,
            "BIGTIFF": "YES", "interleave": "BAND",
        }
        if compress:
            prof.update(compress="DEFLATE", predictor=2, zlevel=1)
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.close()
        indexes = list(range(1, min(3, src.count) + 1))
        out = lay["out"]
        print(f"output {prof['width']}x{prof['height']} "
              f"compress={'DEFLATE' if compress else 'NONE'} "
              f"writes={'512-aligned' if aligned else '504 straddling'}")
        print(f"scratch {tmp.name}\n")
        print(f"{'tiles':>12}  {'read':>9}  {'write':>9}  {'tiles/min':>10}")
        try:
            with rasterio.open(tmp.name, "w", **prof) as dst:
                rbuf, wbuf, done = [], [], 0
                t0 = time.perf_counter()
                for i, (c, r, w, h) in enumerate(jobs_for(lay)):
                    if done >= tiles:
                        break
                    rbuf.append(
                        timed(read, src, indexes, Window(c, r, w, h), out))
                    ty, tx = divmod(i, lay["x_tiles"])
                    if aligned:
                        dr, dc = ty * 512, tx * 512
                    else:
                        dr = OVERLAP_PRED // 2 + ty * core
                        dc = OVERLAP_PRED // 2 + tx * core
                    block = np.zeros((core, core), dtype=np.uint8)
                    tw0 = time.perf_counter()
                    dst.write(block, 1,
                              window=Window(dc, dr, core, core))
                    wbuf.append(time.perf_counter() - tw0)
                    done += 1
                    if len(wbuf) >= bucket:
                        rm, wm = statistics.mean(rbuf), statistics.mean(wbuf)
                        print(f"{done - bucket + 1:>5}-{done:<6}  "
                              f"{rm:>8.3f}s  {wm:>8.3f}s  "
                              f"{60.0 / max(rm + wm, 1e-9):>10.0f}")
                        rbuf, wbuf = [], []
                el = time.perf_counter() - t0
                print(f"\n{done} tiles in {el:.1f}s "
                      f"({60.0 * done / max(el, 1e-9):.0f} tiles/min)")
        finally:
            sz = os.path.getsize(tmp.name) / (1 << 30)
            print(f"scratch grew to {sz:.2f} GiB; removing")
            os.unlink(tmp.name)


def probe_pipeline(path, tiles, bucket, producers=3, write=True,
                   compress=True, aligned=False, infer_s=0.012, batch=4):
    """The real shape of predict_stream_to_raster, minus the model.

    Single-threaded reads do NOT reproduce issue #43 -- they stay flat
    well past the tile where a real run cliffs. What the real run adds is
    concurrency: N producer threads on CONTIGUOUS shards (so they read
    three widely separated bands at once), a bounded queue, and a
    consumer writing 504x504 cores into a DEFLATE-compressed BigTIFF
    whose blocks are 512x512 -- every write straddling four blocks.

    Toggle the ingredients to find which one bends the curve:
        --producers 1        collapse the concurrency
        --no-write           reads + queue only, no writer
        --no-compress        writer without DEFLATE
        --aligned            writer without the 504/512 straddle
    """
    import queue as _q
    import threading
    import tempfile

    read = MODES["changed"]
    with rasterio.open(path) as probe:
        lay = build_layout(probe)
        crs, transform, count = probe.crs, probe.transform, probe.count
    core = IMG_WIDTH - OVERLAP_PRED
    indexes = list(range(1, min(3, count) + 1))
    out = lay["out"]
    alljobs = list(jobs_for(lay))[:tiles]
    # Contiguous shards, exactly like _split_jobs_for_producers.
    shards, n = [], len(alljobs)
    for w in range(producers):
        s, e = int(round(w * n / producers)), int(round((w + 1) * n
                                                        / producers))
        if e > s:
            shards.append(list(enumerate(alljobs))[s:e])
    q = _q.Queue(maxsize=max(2, 8))
    stop = threading.Event()

    def produce(shard):
        with rasterio.open(path) as src:
            buf = []
            for idx, (c, r, w_, h_) in shard:
                if stop.is_set():
                    break
                t0 = time.perf_counter()
                read(src, indexes, Window(c, r, w_, h_), out)
                buf.append((idx, time.perf_counter() - t0))
                if len(buf) >= batch:
                    q.put(buf)
                    buf = []
            if buf:
                q.put(buf)
        q.put(None)

    prof = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "width": lay["x_tiles"] * core + IMG_WIDTH,
        "height": lay["y_tiles"] * core + IMG_WIDTH,
        "crs": crs, "transform": transform,
        "tiled": True, "blockxsize": 512, "blockysize": 512,
        "BIGTIFF": "YES", "interleave": "BAND",
    }
    if compress:
        prof.update(compress="DEFLATE", predictor=2, zlevel=1)
    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp.close()
    # Realistic payload: a mostly-zero binary mask still has entropy, and
    # all-zero blocks would compress to nothing and fake the write away.
    rng = np.random.default_rng(0)
    payload = (rng.random((core, core)) < 0.02).astype(np.uint8)

    print(f"producers={producers} write={write} "
          f"compress={'DEFLATE' if compress else 'NONE'} "
          f"writes={'aligned' if aligned else '504-straddle'}")
    print(f"output {prof['width']}x{prof['height']}\n")
    print(f"{'tiles':>12}  {'read':>9}  {'write':>9}  {'q':>4}  "
          f"{'tiles/min':>10}")

    threads = [threading.Thread(target=produce, args=(s,), daemon=True)
               for s in shards]
    dst = rasterio.open(tmp.name, "w", **prof) if write else None
    try:
        for t in threads:
            t.start()
        done, finished, rbuf, wbuf = 0, 0, [], []
        t_start = time.perf_counter()
        while finished < len(threads):
            item = q.get()
            if item is None:
                finished += 1
                continue
            qfill = q.qsize() / float(max(1, q.maxsize))
            for idx, dt in item:
                rbuf.append(dt)
                time.sleep(infer_s / max(1, batch))
                if dst is not None:
                    ty, tx = divmod(idx, lay["x_tiles"])
                    if aligned:
                        dr, dc = ty * 512, tx * 512
                    else:
                        dr = OVERLAP_PRED // 2 + ty * core
                        dc = OVERLAP_PRED // 2 + tx * core
                    tw = time.perf_counter()
                    dst.write(payload, 1, window=Window(dc, dr, core, core))
                    wbuf.append(time.perf_counter() - tw)
                else:
                    wbuf.append(0.0)
                done += 1
                if len(rbuf) >= bucket:
                    el = time.perf_counter() - t_start
                    print(f"{done - bucket + 1:>5}-{done:<6}  "
                          f"{statistics.mean(rbuf):>8.3f}s  "
                          f"{statistics.mean(wbuf):>8.3f}s  "
                          f"{qfill:>3.0%}  "
                          f"{60.0 * done / max(el, 1e-9):>10.0f}")
                    rbuf, wbuf = [], []
        el = time.perf_counter() - t_start
        print(f"\n{done} tiles in {el:.1f}s "
              f"({60.0 * done / max(el, 1e-9):.0f} tiles/min)")
    finally:
        stop.set()
        if dst is not None:
            dst.close()
        sz = os.path.getsize(tmp.name) / (1 << 30)
        print(f"scratch grew to {sz:.2f} GiB; removing")
        os.unlink(tmp.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ortho", required=True)
    ap.add_argument("--probe", default="layout",
                    choices=["layout", "positional", "temporal", "modes",
                             "contended", "pipeline"])
    ap.add_argument("--producers", type=int, default=3)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--aligned", action="store_true",
                    help="contended: write at 512-aligned offsets")
    ap.add_argument("--no-compress", action="store_true",
                    help="contended: write the output uncompressed")
    ap.add_argument("--mode", default="changed", choices=sorted(MODES))
    ap.add_argument("--samples", type=int, default=20,
                    help="windows per group (positional)")
    ap.add_argument("--tiles", type=int, default=2000,
                    help="tiles to replay (temporal) / compare (modes)")
    ap.add_argument("--bucket", type=int, default=200)
    ap.add_argument("--gdal-cache-mb", type=int,
                    help="set GDAL_CACHEMAX for this run")
    args = ap.parse_args()

    if not os.path.exists(args.ortho):
        sys.exit(f"no such ortho: {args.ortho}")
    if args.gdal_cache_mb:
        os.environ["GDAL_CACHEMAX"] = str(args.gdal_cache_mb)
        print(f"GDAL_CACHEMAX={args.gdal_cache_mb} MB\n")

    if args.probe == "layout":
        probe_layout(args.ortho)
    elif args.probe == "positional":
        probe_positional(args.ortho, args.mode, args.samples)
    elif args.probe == "temporal":
        probe_temporal(args.ortho, args.mode, args.tiles, args.bucket)
    elif args.probe == "pipeline":
        probe_pipeline(args.ortho, args.tiles, args.bucket,
                       producers=args.producers,
                       write=not args.no_write,
                       compress=not args.no_compress,
                       aligned=args.aligned)
    elif args.probe == "contended":
        probe_contended(args.ortho, args.tiles, args.bucket,
                        aligned=args.aligned,
                        compress=not args.no_compress)
    else:
        probe_modes(args.ortho, args.tiles)


if __name__ == "__main__":
    main()
