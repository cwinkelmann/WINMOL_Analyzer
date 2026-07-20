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

## What is actually settable

**Critical:** `cpu_workers` and `gpu_workers` are **outputs, not inputs**. The
planner computes them and overwrites whatever you set — it never reads them
(`classes/ExecutionPlan.py`). Setting them has no effect. Tune the ceilings
below instead.

| Key | Default | Meaning |
|---|---|---|
| `max_cpu_workers` | 32 | Hard ceiling on CPU workers, applied as `min(this, cores-1)`. **The main lever on a many-core box.** |
| `max_gpu_workers` | 8 | Ceiling on GPU worker processes (one per GPU). |
| `single_gpu_cpu_workers` | 24 | CPU workers requested when 1 GPU is present; clamped by `max_cpu_workers`. |
| `multi_gpu_cpu_workers` | 48 | Same for >1 GPU. With the default ceiling of 32 this **never takes effect**. |
| `max_vector_tile_workers` | 4 | Ceiling on tiles vectorised in parallel. The vector phase is ~73 % of a run, so this caps the slowest stage. |
| `prediction_batch_gpu` / `_multi_gpu` / `_max_gpu` | 4 / 12 / 16 | Tiles per inference batch. |
| `prediction_producer_workers_*` | 1 / 6 / 6 | Threads reading + preparing tiles to feed the GPU. |
| `tile_inner_px` | 4096 | Vector tile size. Bigger = fewer seams, more RAM per worker. |
| `tile_overlap_m` | 12.0 | Halo between vector tiles; also the merge de-duplication buffer. |
| `gpu_memory_fraction` | 0.9 | Fraction of GPU memory a worker may use. |
| `prediction_batch_autotune` | False | Off because it re-runs every prediction, costing minutes on CoreML/Metal for ~1 % throughput. |
| `stem_binary_threshold` | 0.5 | Mask binarisation cut-off. **Changing this changes results** — the golden fixtures assume 0.5. |
| `min_length` | 2.0 | Shortest stem kept (m). Also a results-changing knob. |
| `measuring_point_spacing_m` | 0.5 | Diameter sampling interval along a stem. |
| `diameter_method` | contour | `contour` or `edt`. |

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

## Suggested defaults by machine

These are **starting points to measure from, not tuned values.** Nothing below
has been benchmarked; the defaults were set for laptops and small workstations,
and are clearly conservative on a large node.

**Workstation / laptop (8–16 cores, 1 GPU)** — leave the defaults alone.

**Large node (100+ cores, multi-GPU)**

```json
{
  "max_cpu_workers": 96,
  "multi_gpu_cpu_workers": 96,
  "max_vector_tile_workers": 24,
  "prediction_batch_multi_gpu": 16
}
```

Rationale, in order of expected value:

1. **`max_vector_tile_workers` is the biggest available lever.** The vector
   phase dominates a run and is currently capped at 4 processes regardless of
   core count. Note `tile_workers = min(max_vector_tile_workers, cpu_workers//4,
   tiles)`, so raising `max_cpu_workers` too is required for it to bite — and
   it is also bounded by the number of tiles, so a small ortho will not use 24
   either way.
2. **`max_cpu_workers` at 32 wastes a 224-core machine** and silently neuters
   `multi_gpu_cpu_workers: 48`.
3. **Memory scales with workers.** Each vector tile worker holds a
   `tile_inner_px`-sized array plus geometry; 24 workers × 4096 px tiles is
   substantial. Watch RSS before pushing further.

**Do not raise blindly.** Measured on this project, more workers has twice made
things *worse*: a process pool in quantification was **10× slower** than serial
because pickling dominated (fixed by switching to threads), and prediction
batch 8 was a **36 % regression** versus batch 4 on CoreML. Change one value,
measure with `benchmark/bench_orig_vs_changed.py`, keep it only if it helps.

## Results-changing vs performance-only

Performance-only — safe to tune, output must not move:
`max_cpu_workers`, `max_gpu_workers`, `*_cpu_workers`,
`max_vector_tile_workers`, `prediction_batch_*`, `prediction_producer_*`,
`producer_queue_batches`, `progress_interval_s*`, `compress_output`.

**Changes results** — the golden fixtures pin these, so a change invalidates
comparisons against earlier runs: `stem_binary_threshold`, `min_length`,
`max_distance`, `tolerance_angle`, `measuring_point_spacing_m`,
`diameter_method`, `max_tree_height`, `tile_overlap_m`, `img_width`/`img_height`.

After tuning anything in the first group, confirm the stem count is unchanged.
A faster run that finds a different number of stems is not a faster run.
