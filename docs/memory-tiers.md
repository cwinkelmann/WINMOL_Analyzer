# Running WINMOL on limited RAM — recommendations by tier

The pipeline has two peaks, and they are in **different** phases:

- **Prediction** holds one CUDA onnxruntime session (~1.1 GB RSS; it also
  maps ~10 GB of *virtual*, which does not count against RSS but does
  count against the kernel's overcommit limit — see below) plus a GDAL
  block cache the entrypoint sizes at 20 % of RAM.
- **Vectorisation** forks/spawns a pool of tile workers. Since the pool
  now uses the `spawn` start method (utils/VectorTilePipeline.py), each
  worker is an independent process: **~0.1 GB fixed** (Python + geo/
  science imports, measured) plus its per-tile working set. Its memory is
  *additive*, not shared copy-on-write.

Peak RAM is therefore roughly:

```
prediction:  1.1 (session) + GDAL_CACHEMAX + ~0.5 overhead
vector:      parent + N_workers × (0.1 + per_tile_working_set)
```

Two knobs move these, both settable as env vars on `docker run`:

| knob | controls | entrypoint default |
|---|---|---|
| `WINMOL_JOBS` | orthomosaics processed in parallel | `RAM // 6`, capped by GPU count |
| `WINMOL_CONFIG_OVERRIDES_JSON` `max_vector_tile_workers` | vector pool size | auto (RAM- and CPU-bounded) |
| `GDAL_CACHEMAX` | GDAL block cache, MB | `20 % of RAM ÷ jobs` |

## The overcommit constraint (why a GPU run can be killed with RAM free)

A CUDA session maps ~10 GB of **virtual** address space that the driver
never releases for the life of the process. With `vm.overcommit_memory=0`
(the Linux default) the kernel refuses new allocations once
`Committed_AS` exceeds `CommitLimit` (≈ RAM×0.5 + swap). Forking the
vector pool from a CUDA-holding parent multiplied that virtual footprint
by the worker count and blew the limit **with real RAM still free**. The
`spawn` fix removes this: spawned workers do not inherit the parent's
mappings. These recommendations assume the spawn fix is in.

## Recommendations (single ortho, single GPU)

Reserve ~2 GB for the OS. Numbers are for the **GPU image**; the CPU image
drops the ~1.1 GB session but the model then runs in RAM, so treat CPU as
one tier lower.

| RAM | WINMOL_JOBS | max_vector_tile_workers | GDAL_CACHEMAX | notes |
|---|---|---|---|---|
| **8 GB** | 1 | 2 | 1024 MB | tight; prefer overviews on the input to cut read amplification |
| **16 GB** | 1 | 4 | 2048 MB | comfortable for one ortho |
| **24 GB** | 1 (2 if CPU-bound) | 6 | 3072 MB | headroom for a second job on CPU-heavy vector work |
| **32 GB** | 2 | 8 | 4096 MB | two orthos in parallel, or one with a wide vector pool |

`WINMOL_JOBS` above 1 only helps with **multiple** input files — a single
ortho runs one job regardless. On multi-job runs every number in the row
is *per job*, so the pool total is `JOBS × max_vector_tile_workers`; keep
that product within `(RAM − 2) ÷ 0.5`.

## If a run is killed at the vector phase

1. Confirm the spawn fix is present (`grep spawn utils/VectorTilePipeline.py`).
2. Lower `max_vector_tile_workers` first — it is the additive term.
3. Then lower `GDAL_CACHEMAX`.
4. On a ZFS host, `MemAvailable` under-reports by the ARC size (ARC is
   reclaimable but excluded); a monitor reading it may kill a healthy run.
