# Prediction reader pool — design

Issue: cwinkelmann/WINMOL_Analyzer#60. Branch: `perf/prediction-reader-pool`.

## Problem

On the multi-GPU prediction path, each GPU worker process is its own
single-threaded reader, and read and inference run back to back on one thread:

```python
# utils/PredictWorkers.py, prediction_worker (one per GPU)
with rasterio.open(input_raster) as src:
    while True:
        for batch_jobs in _group_jobs(jobs, batch_size):
            raw_tiles = _read_batch_jobs(src, ...)   # ~27 ms/tile, serial
            pred_cores = _predict_batch(raw_tiles)   # ~4 ms/tile
```

While a worker decodes, its GPU idles; while the GPU infers, its reader idles.
Measured on carrot (8× H100, 224 cores, Tegel R13, file page-cached):

```
8 workers × 1 / (0.027 + 0.004) = 258 tiles/s = ~15,500 tiles/min   (model)
observed                                          13,657 tiles/min
```

The GPUs are busy ~11 % of the time. The read is CPU decode-bound, not
disk-bound. The execution plan's `cpu_workers = 32` is never consumed by this
path. TensorRT (#58) cannot help: it shortens the 4 ms half of a 31 ms serial
loop (measured −0.4 % on R13, +4.1 % on R12).

The single-GPU streaming path in `utils/Prediction.py` has thread producers,
but its own comment (line 244) records they reached only 1.21× "because of the
GIL". The spike below shows that finding is stale.

## Spike result (2026-09-15, carrot, R13, 300 random 1217² native windows)

Same calls as `_read_batch_jobs`: `src.read(indexes, window, boundless=True,
fill_value=0)` + `read_masks(1, window, boundless=True)`. Single-thread
baseline 29.2 ms/tile, matching production's 27 ms.

| readers | threads, shared handle | threads, own handle | processes |
|---|---|---|---|
| 1 | 34 tiles/s (1.0×) | 39 | — |
| 8 | 177 (5.2×) | 178 (5.2×) | 254 (7.4×) |
| 16 | **257 (7.5×)** | 259 (7.6×) | 396 (11.5×) |

- **Threads scale.** `rasterio` releases the GIL around `GDALRasterIO`, and
  with resize done in-graph there is no GIL-holding work left per tile. The
  1.21× figure predates in-graph resize.
- **One H100 consumes ~250 tiles/s at 4 ms.** Sixteen reader threads saturate
  it. Processes scale further (11.5×) and are the fallback if inference ever
  outruns 16 threads — TensorRT at 3 ms would need ~333 tiles/s.
- **Shared vs own handle: no difference in throughput.** But the
  `boundless=False` + shared-handle + 2-thread run terminated abnormally
  without output. Each reader therefore gets its **own** handle: it costs
  nothing and removes a class of bug.
- `boundless=True` costs ~7 % (27.2 vs 29.2 ms at one thread). Not worth
  changing; it handles edge tiles.

## Decision

**Approach A with threads.** Each GPU worker process owns a pool of reader
threads over its own shard, feeding an in-process bounded `queue.Queue`. One
design for N GPUs; N = 1 is the laptop. The `Prediction.py` thread-producer
path is replaced by this, not kept alongside.

Rejected: a shared cross-GPU reader pool (approach B) — better balance only
when shards are skewed, at the cost of abandoning up-front sharding and
pushing ~1.5 GB/s through one queue. Processes as readers — more code
(pickling, cross-process failure propagation, `mp.Queue` per GPU) for headroom
the current inference cost cannot use.

## Constraints (decided)

1. **Every machine the plugin runs on.** Carrot, the T14 (1 GPU, 12 threads,
   46 GB) and CPU-only QGIS installs all go through the same code.
2. **Proof of no regression, not absence of change** — the (ii) gate:
   - stem map raster and every GPKG layer **bit-identical** to current `main`
     on the same input, row for row (geometry WKB + all attributes, in file
     order), block-wise for the raster;
   - throughput **≥** and peak RSS **≤** current `main`, measured on the T14
     (1 GPU) and on a CPU-only run. Faster is allowed; slower or hungrier
     fails.
3. Nothing "activates". Small machines get a small reader count from the same
   sizing rule.

## Architecture

Three roles, all in `utils/PredictWorkers.py`.

**Reader** — a thread inside a GPU worker process. Owns one `rasterio`
dataset handle (opened in the thread). Loop: take a job-batch from the shard,
`_read_batch_jobs` (unchanged), `q.put(decoded batch)`. Blocks on `put` when
the queue is full. No model, no GPU.

**GPU worker** — the existing per-GPU process. Its `rasterio.open` and inline
read are removed. Loop: `batch = q.get(timeout) → _predict_batch → result_q`.
On `None` from every reader it drains and exits. On an exception object from
any reader it re-raises.

**Coordinator** — the existing `run_multi_gpu_prediction` / service. Shards
jobs across GPUs as today. Per GPU it spawns the worker; the worker spawns its
own readers over its shard. The result queue and assembly are untouched.

The CPU-only path is the same worker with a CPU session and `R = 1`.

## Data flow

```
coordinator ─shard─▶ [GPU worker process]
                        readers (R threads, own handle each)
                          └─▶ queue.Queue(maxsize = producer_queue_batches)
                                └─▶ predict loop ─▶ result_q ─▶ assembly
```

- Queue depth = existing plan field `producer_queue_batches` (4). Readers
  block on `put`: that is the backpressure.
- Memory bound per GPU = `producer_queue_batches × prediction_batch × tile
  bytes`. A native 1217² tile is 4.44 MB RGB uint8 + 1.48 MB mask ≈ 5.9 MB.
  Carrot: 4 × 12 × 5.9 MB ≈ 284 MB per GPU, ~2.3 GB total. Laptop: 284 MB,
  less where the plan picks a smaller `prediction_batch`.
- Tile order within a shard is **not preserved** across readers. Assembly is
  keyed by tile coordinates; the plan includes a test that proves it is
  order-independent rather than assuming it.

## Sizing

New plan field `prediction_readers_per_gpu`, computed in the existing
`ExecutionPlan`:

```
R = clamp( floor((hw_cpu − 1) / n_gpu), 1, 16 )
```

- `hw_cpu` is `hardware.cpu_count`, as the plan already reads it; the −1 is
  the one core the plan leaves for the OS and coordinator (`hw_cpu − 1` is
  its existing ceiling for `cpu_workers`).
- Not derived from `cpu_workers`: that field is capped at 32 by
  `max_cpu_workers`, so on carrot it would give 32 / 8 = 4 readers per GPU —
  ~136 tiles/s against a card that consumes 250.
- No vector-phase reserve: prediction and vectorisation are sequential
  phases, so during prediction the readers may use every core.
- Upper bound 16: where thread scaling plateaus and one H100 is saturated.
- CPU-only: `R = 1`. Inference is the bottleneck there; more readers only
  spend RAM.
- Carrot: `223 / 8` → 16 (cap). T14: `11 / 1` → 11.
- `WINMOL_PREDICTION_READERS` overrides, for measurement only.

## Failure semantics

A reader that dies must fail the run, never stall it.

- Normal exit: each reader puts `None`. The worker counts sentinels and exits
  after the R-th.
- Error: the reader puts the exception object and exits. The worker treats an
  exception on the queue as fatal: it re-raises, which propagates to the
  coordinator through the existing worker-failure path, which terminates the
  pool and fails the run.
- Silent death: `q.get(timeout=30)`. On timeout the worker checks `reader.is_alive()` for all readers; if any is
  dead without having put a sentinel, that is fatal.
- No partial output with a warning.

## Testing

Unit (pure, no GPU, run in CI on all three OSes):
- sizing rule over a table `(hw_cpu, n_gpu, cpu_only) → R`,
  including the floor of 1 and cap of 16;
- a reader raising → worker raises within the timeout, run fails;
- a reader dying silently → detected via `is_alive()` within the timeout;
- assembly with tiles delivered in shuffled order equals assembly in
  row-major order (bit-identical arrays).

Integration (T14 and carrot, recorded in the PR as numbers, not assertions):
- bit-identical output vs `main` on a real ortho, per constraint 2 —
  existing parity harness pattern (`geopandas`/pyogrio positional WKB +
  attributes; `rasterio` block-wise);
- throughput and peak RSS vs `main` on the T14 (1 GPU) and CPU-only;
- carrot R13 throughput, expected to move from ~13.7k toward the GPU ceiling
  (~120k tiles/min at 4 ms, minus the vector phase's share of the cores).

## Out of scope

Cross-GPU balancing; `shared_memory`; changing the sharding; the vector
phase's own under-provisioning (`vector_tile_workers = 16` on 224 cores —
real, separate issue); reader processes (documented fallback only).
