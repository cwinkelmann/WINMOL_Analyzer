# Benchmarks

What this release costs in wall time and memory, on which inputs, and
how each number was obtained. Every figure below is a complete
end-to-end run — no extrapolation from crops or partial phases — and
every pair of runs being compared was checked for output equality, not
just speed.

**Short version.** Against the released v0.6.1.1: **2.6× faster end to
end**, prediction **1.8×**, vector phase **6–10×**, output **identical**
(merged GPKG row for row, stem map pixel for pixel), peak memory down
from over 16 GiB to 6.5 GiB in the worst case. Against v0.5.0,
prediction is roughly **7×** — but read [Provenance](#provenance) before
quoting that one.

## The machine

| | |
|---|---|
| CPU | Intel i7-1255U, 10 cores / 12 threads |
| GPU | NVIDIA RTX 4080 SUPER, 16 GB |
| RAM | 46.75 GB, ZFS root (ARC typically holds 17–23 GB) |
| Storage | local NVMe |
| Driver / CUDA | 580.173.02 / 12.5.1 |

A laptop, not a workstation: it thermally throttles under 11 busy
threads, so repeat runs of the same arm vary by a few percent. Every
comparison below is between runs on this same machine, mostly on the
same day, and the differences are far larger than that noise.

## The inputs

| ortho | pixels | file | compression | prediction tiles | vector windows |
|---|---|---|---|---|---|
| Tegel R13 `result_Res1.3_COG.tif` | 392558 × 335327 | 19.7 GB | JPEG, overviews | 99231 | 1512 (796 with foreground) |
| Tegel R12 `result_Res1.2_webp.tif` | 338566 × 333907 | 12.3 GB | WEBP 4-band, overviews | 75072 | 1156 (416 with foreground) |

Model: `Spruce_Deadwood` throughout. v0.5.0 runs the Keras `.hdf5`; every
later version runs the ONNX fp16 export of the same weights.

## Against the released v0.6.1.1

Both arms run the released container image
`ghcr.io/cwinkelmann/winmol-analyzer-gpu:v0.6.1.1`; the "this release"
arm has the branch mounted over `/app`. The OS, Python, onnxruntime and
GDAL are therefore identical and only the code under test differs.

### Tegel R13

| stage | v0.6.1.1 | this release | factor |
|---|---|---|---|
| prediction | 2187 s | 1216 s | 1.80× |
| vector phase (incl. tile split) | ~1790 s | 295 s | 6.07× |
| **total** | **4008 s** (67 min) | **1555 s** (26 min) | **2.58×** |
| avg per vector tile | 21.0 s | 2.6 s | |
| peak container RSS | 8.2 GiB | 7.1 GiB | |

### Tegel R12

| stage | v0.6.1.1 | this release | factor |
|---|---|---|---|
| prediction | 1964 s | 1095 s | 1.79× |
| vector phase (incl. tile split) | ~1380 s | 140 s | 9.9× |
| **total** | **3340 s** (56 min) | **1249 s** (21 min) | **2.67×** |
| avg per vector tile | 27.9 s | 1.8 s | |
| peak container RSS | > 16 GiB (see below) | 6.5 GiB | |

Prediction throughput: R13 2722 → 4895 tiles/min, R12 2294 → 4114
tiles/min.

The vector phase was 45 % of an R13 run and is now 19 %. Prediction is
the majority of a run again, which is where the next optimisation
belongs.

### The v0.6.1.1 R12 arm needed a bigger container

Under a 16 GB cap the kernel OOM-killed one of its fork-pool workers
(6.2 GB anonymous RSS, cgroup constraint) at 96 % of the vector phase.
`imap_unordered` then waited on a result that would never arrive: the
container sat at 0.1 % CPU for 45 minutes until it was stopped. The arm
was re-run with `--memory=32g` to get a completed baseline, and that is
the 3340 s above.

This release finishes the same file under a 16 GB cap, peaking at
6.5 GiB. For anyone running large orthos on a 16 GB machine that is the
practical headline: the difference between a run that finishes and one
that hangs.

## Output is identical, not merely similar

Compared against the v0.6.1.1 arm's own output from the same machine:

| | R13 | R12 |
|---|---|---|
| stems / vectors | 12,718 / 122,896 (both arms) | 3,190 / 33,000 (both arms) |
| merged GPKG | identical row for row, all three layers | identical row for row |
| stem-map raster | 25,208,809,984 px compared, **0 differing** | 19,071,698,752 px compared, **0 differing** |

"Row for row" means geometry WKB **and** every attribute, in file order —
not just the same set of features. Method: read each layer with
`geopandas` (pyogrio engine) and compare WKB hex plus all non-geometry
columns positionally; compare rasters block-wise through `rasterio`
windows.

This was also checked at tile granularity throughout development: a pool
benchmark over 453 real halo tiles from a full R13 run, re-run after
every optimisation step, produced **419/419 identical per-tile GPKGs**
each time.

## Outside the container

The QGIS plugin does not use the image — it spawns its managed venv's
Python directly. Driving that exact command (Python 3.11,
onnxruntime-gpu 1.26, rasterio 1.4.4 / GDAL 3.10.3 wheels) gives the same
picture, and the output still matches the container's byte for byte,
across two different onnxruntime builds and two different GDAL builds:

| | R13 | R12 |
|---|---|---|
| prediction | 1232 s | 1150 s |
| vector phase | 295 s | 142 s |
| **total** | **1571 s** | **1308 s** |

## Where the vector-phase time went

Average per tile at 11 workers, from the run log's own `VECTOR SUMMARY`.
The "mid-branch" column is the state after the first round of
optimisation, kept because it shows which change did what; v0.6.1.1
printed only the `quant` and `connect` averages, so no full stage
breakdown exists for it.

| stage | mid-branch | this release |
|---|---|---|
| skeletonisation | 2.93 s | 0.80 s |
| quantification | 1.59 s | 1.15 s |
| connect stems | 0.79 s | 0.33 s |
| write | 0.40 s | 0.35 s |
| restore + build | 0.01 s | 0.01 s |
| **total per tile** | **5.71 s** | **2.64 s** |
| serial tile split (once per run) | 89 s | 0 s (done in the workers) |

Skeletonisation dominated because it scanned the
`max_tree_height`-padded tile — 7102² for a 4916 px tile, 50 M pixels —
about fifteen times per tile, for a foreground under 1 %. With eleven
workers those scans contended for memory bandwidth: the per-tile average
in a full run was 2.5× the same tile measured in isolation, which is why
adding workers had stopped helping.

Isolated pool benchmark, 453 real halo tiles, output identical at every
step:

| state | 8 workers | 11 workers |
|---|---|---|
| branch point | 890 s | — |
| + foreground-cropped skeleton | 309 s | 265 s |
| + coordinate-list skeleton, connect caching | — | 147 s |
| + cropped polygonisation, endpoint tests computed once | — | 120 s |

## Against v0.5.0

v0.5.0 is the last TensorFlow release; everything after it runs ONNX
Runtime. Most of this gap therefore predates this release — it came with
the runtime change, not with the vector-phase work.

| | v0.5.0 | this release | factor |
|---|---|---|---|
| R13 prediction | 8344 s (139 min) | 1216 s (20 min) | 6.86× |
| R12 prediction | 7864 s (131 min) | 1095 s (18 min) | 7.18× |
| R13 throughput | 714 tiles/min | 4895 tiles/min | |
| R12 throughput | 573 tiles/min | 4114 tiles/min | |
| R12 total run | 9707 s (162 min) | 1249 s (21 min) | 7.77× |
| R12 vector phase | 1842 s | 140 s | 13.2× |

**v0.5.0 never completed R13 on this machine.** Its vector phase was
killed by the kernel (SIGKILL, out of memory) at 445 of 796 tiles, so
there is no v0.5.0 end-to-end R13 total to compare against. The
prediction figure above is from the phase that did complete.

Counts differ slightly between eras because the model format differs
(Keras `.hdf5` vs ONNX fp16): v0.5.0 wrote 3190 stems / 32,877 vectors on
R12, this release writes 3190 / 33,000 — same stems, 0.4 % more vector
nodes. Cross-era output is therefore *comparable*, not identical, unlike
the v0.6.1.1 comparison above, which is exact.

## Provenance

Be precise about this when quoting numbers.

| figure | when measured | how |
|---|---|---|
| this release, both orthos, container | at this release's HEAD | fresh, full runs |
| this release, both orthos, plugin venv | at this release's HEAD | fresh, full runs |
| v0.6.1.1 R12 | same day as its comparison arm | fresh, full run, 32 GB container |
| v0.6.1.1 R13 | earlier in development | full run, carried forward |
| v0.5.0, both orthos | 2026-08-09 | full runs; R13 died in the vector phase |

The v0.6.1.1 R13 baseline is the one figure not re-measured on the same
day as its comparison arm. The R12 baseline, which was, gives 1.79× for
prediction against R13's 1.80×, so the two agree.

### The v0.5.0 environment no longer runs here

A fresh v0.5.0 re-measurement was attempted and abandoned. Two
independent problems, neither in v0.5.0's own code:

1. Its TensorFlow 2.21 fails to `dlopen` `libcusolver.so.11` and falls
   back to the CPU **silently** — 3 tiles/min instead of ~600, a
   400-hour ETA. Adding only the wheel's `nvidia/cusolver/lib` to
   `LD_LIBRARY_PATH` restores GPU detection. (Adding *all* the wheel lib
   dirs instead changes which cuDNN and cuBLAS TensorFlow loads, and it
   then hangs on the first inference.)
2. With the GPU restored it runs one burst of inference (GPU at 94 %)
   and then deadlocks in `futex_wait` before printing a single progress
   line. Reproduced twice.

The venv has not been modified since before the August run (every
package dated 2026-08-07, the run 2026-08-09), so whatever changed is
outside it — the driver, or the environment the August run was launched
from. Re-establishing a working v0.5.0 GPU environment is a piece of
work in its own right; until someone does it, the v0.5.0 column rests on
the August logs.

## Reproducing

```
docker run --rm --gpus all --memory=16g --memory-swap=16g \
  -e WINMOL_PROCESS_TYPE=Nodes -e GDAL_CACHEMAX=2048 \
  -e WINMOL_CONFIG_OVERRIDES_JSON='{"prediction_batch_override": 1}' \
  -v <ortho.tif>:/data/input/<ortho.tif>:ro \
  -v <outdir>:/data/output -v <modeldir>:/data/models \
  ghcr.io/cwinkelmann/winmol-analyzer-gpu:<tag> Spruce_Deadwood
```

The run prints `Elapsed time` per phase and a `VECTOR SUMMARY` block with
the per-stage averages. Worker counts come from the execution plan;
nothing above pins them beyond the batch size.

For memory behaviour on smaller machines see
[memory-tiers.md](memory-tiers.md); for the historical throughput cliff
on large orthos see [performance-v05-to-now.md](performance-v05-to-now.md).
