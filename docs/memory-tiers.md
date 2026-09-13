# Running WINMOL on limited RAM — measured, by tier

Everything below is measured (GPU image, RTX 4080 host, 2026-09-13), not
modelled. Two inputs: a 25-tile crop for the knob sweep, and the full
19 GB R13 orthomosaic (99,231 prediction tiles, 796 vector tiles) for
the realistic peak. Every configuration produced bit-identical output.

## What actually costs RAM

Peak container RSS on the 25-tile crop, single ortho, single GPU:

| vector workers | GDAL cache | batch | peak RSS |
|---|---|---|---|
| 1 | 256 MB | 1 | 2.49 GiB |
| 2 | 256 MB | 1 | 2.49 GiB |
| 2 | 512 MB | 1 | 2.75 GiB |
| 4 | 512 MB | 1 | 3.60 GiB |
| 4 | 2048 MB | 1 | 4.68 GiB |
| 8 | 2048 MB | 1 | 4.77 GiB |
| 8 | 2048 MB | 4 | 5.56 GiB |

Read down the table and the levers separate cleanly:

- **The floor is ~2.5 GiB**: the CUDA session (~1.1 GB RSS), the Python
  process with its geo/science imports, the writer thread, and one tile
  in flight. Nothing below this is reachable on the GPU image.
- **Vector workers are nearly free.** 1→2 workers: +0 MB. 4→8: +90 MB.
  Each spawned worker is ~100 MB of imports plus a working set that the
  foreground crop (utils/Skeletonization.py) keeps small. The execution
  plan budgets 256 MB per worker (`VECTOR_WORKER_PRIVATE_BYTES`, a 10×
  margin over the measured marginal cost); the previous 1.75 GB figure
  predated the crop and throttled the pool to 10 on a 46 GB host.
- **GDAL block cache is the biggest knob**: 512 → 2048 MB cost +1.1 GB.
- **Batch size costs host RAM too**, not just VRAM: batch 1 → 4 at 8
  workers cost +0.8 GB. Batch 1 is also the fastest (measured across
  1/4/8/12/16), so there is no reason to raise it on a small machine.

**Full-scale correction:** the same 8-worker / 2048 MB config peaked at
**8.36 GiB on the full orthomosaic** vs 4.77 on the crop — larger halo
tiles, a larger merge, more in flight. Treat full-scale peak as roughly
**1.75× the crop figure** and size from that.

## Recommendations (single ortho, single GPU)

Reserve ~2 GB for the OS. Peaks below are the full-scale estimate
(crop × 1.75), and the 8 GB row was verified directly: the 2-worker /
256 MB configuration completed under a hard 4 GB container cap.

| RAM | `max_vector_tile_workers` | `GDAL_CACHEMAX` | batch | est. full-scale peak |
|---|---|---|---|---|
| **8 GB** | 2 | 512 MB | 1 | ~4.5 GiB |
| **16 GB** | 8 | 2048 MB | 1 | ~8.5 GiB |
| **24 GB** | 8 | 3072 MB | 1 | ~10 GiB — room for `WINMOL_JOBS=2` on multi-file runs |
| **32 GB** | 12 | 4096 MB | 1 | ~12 GiB — `WINMOL_JOBS=2`, or 3 with a smaller cache |

Since workers are almost free, on 16 GB and up the pool size should be
set by **CPU count**, not RAM; the RAM-derived cap in the plan can only
under-provision it. On 8 GB, the small pool costs about 25% on the vector
phase (53 s vs 42 s on the crop) — acceptable, and far better than an
OOM.

`WINMOL_JOBS` above 1 helps only with **multiple** input files (a single
ortho runs one job regardless); every row is then *per job*.

All three are set as environment variables on `docker run`;
`max_vector_tile_workers` and the batch go in `WINMOL_CONFIG_OVERRIDES_JSON`.

## Overcommit: measured, not a wall

A CUDA onnxruntime session maps ~10 GB of *virtual* address space that
the driver keeps for the process lifetime (VmSize 10.3 GB after one
inference, 9.3 GB after `close()` + `cudaDeviceReset()`). Under the
Linux default `vm.overcommit_memory=0` this is **not** a limit: every
configuration above ran at 101–126% of `CommitLimit`, and the full run
reached 144%, all without incident. The heuristic only refuses single
allocations that are absurd relative to free memory.

It becomes a hard wall only under `vm.overcommit_memory=2` (strict),
where `CommitLimit ≈ RAM/2 + swap` and a 10 GB session alone exceeds it
on a 16 GB host. If a site runs strict overcommit, prediction must run
in a child process that exits before vectorisation
(utils/PredictionProcess.py — present, not wired by default).

## If a run is killed at the vector phase

1. The tile split before the pool writes one halo tile per grid cell;
   make sure they are DEFLATE-compressed (utils/IO.write_tile_raster).
   Uncompressed, a full ortho pushed ~78 GB through the page cache and
   was killed there, in the loop, before any worker existed.
2. Lower `GDAL_CACHEMAX` — the largest single lever.
3. Keep batch at 1.
4. Only then lower `max_vector_tile_workers`; it recovers almost nothing.
5. On a ZFS host, `MemAvailable` excludes the ARC (reclaimable but
   uncounted); a monitor reading it may kill a healthy run.
