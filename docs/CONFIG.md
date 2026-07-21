# Configuration reference

Defaults live in `classes/Config.py`. Override any of them without editing code:

```bash
export WINMOL_CONFIG_OVERRIDES_JSON='{"max_cpu_workers":128}'
# in a container:
docker run -e WINMOL_CONFIG_OVERRIDES_JSON='{"max_cpu_workers":128}' ...
```

Unknown keys are reported and skipped (`winmol_run.py:91`).

## Read the run log carefully — it prints two different things

```
Execution plan:              <- what the planner DECIDED
  gpu_workers      = 2
  cpu_workers      = 32
  vector_tile_workers = 4
  vector_inner_workers = 1

Configurations:              <- the Config object AFTER the plan overwrote it
  cpu_workers    1           <- NOT the same number as above!
  gpu_workers    2
```

`_apply_plan_to_config` (`winmol_run.py:154`) writes plan results back into the
config, and in `vector_mode: tiled` it sets `config.cpu_workers =
plan.vector_inner_workers`. So the `cpu_workers 1` in the second block means
*one worker inside each vector tile*, not "one CPU". Compare against
`vector_tile_workers` to see the real parallelism: **4 tiles × 1 inner = 4
processes**.

Tile and inner workers COMPOSE: the planner picks `vector_tile_workers`
(capped by `max_vector_tile_workers` and the estimated vector tile count,
printed as `est_vector_tiles`) and gives each tile worker
`cpu_workers // vector_tile_workers` inner workers, so
`vector_tile_workers * vector_inner_workers <= cpu_workers` always holds.

## What is actually settable

**Critical:** `cpu_workers` and `gpu_workers` are **outputs, not inputs**. The
planner computes them and overwrites whatever you set — it never reads them
(`classes/ExecutionPlan.py`). Setting them has no effect. Tune the ceilings
below instead.

| Key | Default | Meaning |
|---|---|---|
| `vector_processing` | auto | Vector-phase strategy: `tiled`, `untiled`, or `auto`. See the section below. |
| `max_cpu_workers` | 32 | Hard ceiling on CPU workers, applied as `min(this, cores-1)`. **The main lever on a many-core box.** |
| `max_gpu_workers` | 8 | Ceiling on GPU worker processes (one per GPU). |
| `single_gpu_cpu_workers` | 24 | CPU workers requested when 1 GPU is present; clamped by `max_cpu_workers`. |
| `multi_gpu_cpu_workers` | 48 | Same for >1 GPU. With the default ceiling of 32 this **never takes effect**. |
| `max_vector_tile_workers` | 4 | Ceiling on tiles vectorised in parallel. The vector phase is ~73 % of a run, so this caps the slowest stage. Remaining CPU budget becomes inner workers per tile (they compose). |
| `prediction_batch_gpu` / `_multi_gpu` / `_max_gpu` | 4 / 12 / 16 | Tiles per inference batch. |
| `prediction_producer_workers_*` | 1 / 6 / 6 | Threads reading + preparing tiles to feed the GPU. |
| `tile_inner_px` | 4096 | Vector tile size. **Changes results** (measured) — see the sweep. Bigger = fewer seams, more RAM per worker; smaller = proportionally more halo recomputation. |
| `tile_overlap_m` | 12.0 | Halo between vector tiles; also the merge de-duplication buffer. |
| `gpu_memory_fraction` | 0.9 | Fraction of GPU memory a worker may use. |
| `prediction_batch_autotune` | False | Off because it re-runs every prediction, costing minutes on CoreML/Metal for ~1 % throughput. |
| `stem_binary_threshold` | 0.5 | Mask binarisation cut-off. **Changing this changes results** — the golden fixtures assume 0.5. |
| `min_length` | 2.0 | Shortest stem kept (m). Also a results-changing knob. |
| `measuring_point_spacing_m` | 0.5 | Diameter sampling interval along a stem. |
| `diameter_method` | contour | `contour` or `edt`. |

## `vector_processing` — tiled vs. un-tiled vector phase

The tiled vector path does not reproduce the legacy whole-raster results:
seam-split stems can survive the edge-buffer dedup and the order-sensitive
`connect_stems` cascades (fixture: 60 vs 58 stems; full ortho: 755 vs 742).
The un-tiled chain is the reference-correct output and, when the raster fits
in RAM, ~2x faster (measured 64.9 s vs 120.3 s on a 14624x10088 stem map).

- `untiled` — always run the legacy whole-raster chain
  (`Skel.find_segments` → … → `Quant.quantify_stems`) and write the same
  GeoPackage contract as the tiled merge (`stems`/`nodes`/`vectors` layers,
  identical schema).
- `tiled` — always tile + merge (constant memory; the pre-existing behavior).
- `auto` (default) — pick `untiled` only when it is clearly safe:
  estimated stem-map working set (prediction-grid pixels x ~48 bytes/pixel,
  covering the uint8 map + skeletonize/labels/EDT copies with a 2x margin)
  is at most **50 % of detected RAM**. Unknown RAM or a degenerate raster
  falls back to `tiled`. Heuristic in
  `classes/ExecutionPlan.py::_untiled_fits_in_ram`.

On the golden fixture both paths produce byte-identical stem geometry
(58 stems, same order-independent WKB hash), because the default
`tile_inner_px=4096` puts the whole fixture in one tile.

## The GPU count is NOT configurable

`gpu_workers` is capped by the **number of prediction tiles**, hardcoded in
`ExecutionPlan.py`:

```python
gpu_workers = min(max_gpu_workers, gpus_available)
if   tiles < 1000: gpu_workers = min(gpu_workers, 2)
elif tiles < 2500: gpu_workers = min(gpu_workers, 4)
else:              gpu_workers = min(gpu_workers, 8)
```

On an 8-GPU node a job with fewer than 1000 tiles uses **two GPUs**, and no
config setting changes that. The reasoning is sound — each worker is a process
that loads the model and initialises CUDA, and that startup dominates a small
job — but the thresholds were not chosen with H100-class hardware in mind, where
startup is cheap and the job is often "small" by tile count while still worth
spreading. Raising them is a code change.

## Measured: a config sweep, and why nothing helped

Six configurations, 3 runs each, Spruce_Deadwood on Barnekow, RTX 4080 SUPER
(16 cores), via `benchmark/`:

| config | median | vs base | stems | output |
|---|---|---|---|---|
| **baseline** | **33 s** | — | 458 | — |
| `max_vector_tile_workers:1` | 43 s | +30 % | 458 | same |
| `max_cpu_workers:128` | 34 s | +3 % | 458 | same |
| + `max_vector_tile_workers:16` | 34 s | +3 % | 458 | same |
| `tile_inner_px:2048` | 46 s | +39 % | **459** | **changed** |
| `tile_inner_px:8192` | 40 s | +21 % | **456** | **changed** |

**No output-preserving config beat the defaults.** They are well tuned for a
normal workstation; do not tune this blindly.

### Why the overrides did nothing — read the resolved plan

Every run resolved to `cpu_workers = 11` and `vector_tile_workers = 2`,
*including* the one setting `max_cpu_workers: 128`:

```
max_cpu_workers = min(128, hw_cpu-1 = 15) = 15  →  ×0.75  →  cpu_workers = 11
vector_tile_workers = min(max_vector_tile_workers, cpu_workers//4 = 2, tiles) = 2
```

`max_vector_tile_workers: 16` was inert because **`cpu_workers//4` binds first**,
and that is pinned by the core count. The lesson generalises: after setting an
override, check the `Execution plan:` block in the log. An override that is
clamped away looks exactly like one that had no effect.

### What this does NOT tell you about a large node

On 224 cores the arithmetic lands in a different regime —
`cpu_workers ≈ 32` → `tile_workers = min(max_vector_tile_workers=4, 8) = 4` —
so there `max_vector_tile_workers` **is** the binding cap and raising it may
help. That is the one case worth testing, and it cannot be tested on a small
machine. Measure it on the target hardware before adopting anything.

### Two results that do generalise

- **Inner parallelism loses.** One tile with 11 inner workers was 30 % slower
  than two tiles with one each. Consistent with everything else measured in
  this project: a process pool in quantification was 10× slower than serial,
  and prediction batch 8 was a 36 % regression versus batch 4.
- **`tile_inner_px` changes results** — 459 and 456 stems versus 458. It is not
  a performance knob; it alters tiling and therefore seam handling.

## Results-changing vs performance-only

Performance-only — safe to tune, output must not move:
`max_cpu_workers`, `max_gpu_workers`, `*_cpu_workers`,
`max_vector_tile_workers`, `prediction_batch_*`, `prediction_producer_*`,
`producer_queue_batches`, `progress_interval_s*`, `compress_output`.

**Changes results** — the golden fixtures pin these, so a change invalidates
comparisons against earlier runs: `stem_binary_threshold`, `min_length`,
`max_distance`, `tolerance_angle`, `measuring_point_spacing_m`,
`diameter_method`, `max_tree_height`, `tile_overlap_m`, **`tile_inner_px`**,
`img_width`/`img_height`.

`tile_inner_px` is in this group on measured evidence, not on principle: the
sweep above produced 459 and 456 stems against a baseline of 458 purely by
changing it. It looks like a performance knob and is not one.

After tuning anything in the first group, confirm the stem count is unchanged.
A faster run that finds a different number of stems is not a faster run.
