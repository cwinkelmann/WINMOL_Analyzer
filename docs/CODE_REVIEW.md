# WINMOL Analyzer — Code Review

> **See also:** [`CODE_REVIEW_2.md`](./CODE_REVIEW_2.md) — the second-pass deep
> review (2026-07-11) with 19 additional verified bugs (including a HIGH
> merge-crash), refuted claims, and quantified optimization proposals.

Full-codebase review of the core pipeline (prediction → vectorization →
quantification → tiled merge), 2026-07. Findings are grouped by subsystem and
ranked **High / Medium / Low**. Each has a concrete failure scenario and a
`file:line` anchor. Two of the High/dead-code findings (V-1, V-3) were verified
directly against the source; the rest come from close reading and are marked
*PLAUSIBLE* where empirical confirmation is still advisable.

A companion doc, [`mask-to-stem-parts.md`](./mask-to-stem-parts.md), explains the
vectorization pipeline in depth.

## Contents
- [Vectorization & skeletonization (V-*)](#vectorization--skeletonization)
- [Prediction & orchestration (P-*)](#prediction--orchestration)
- [Tiling & vector merge (M-*)](#tiling--vector-merge)
- [Cross-cutting themes](#cross-cutting-themes)

---

## Vectorization & skeletonization
`utils/Skeletonization.py`, `utils/Vectorization.py`, `utils/Quantification.py`,
`utils/Geometry.py`, `classes/*`

### High

**V-1 — `build_stem_parts` if/else branches are identical (verified).**
`Vectorization.py:381-390`. Both the `if start[1] >= stop[1]` and the `else`
branch perform the same endpoint swap + `path.reverse()`. The intended
orientation *normalization* (make `start` the left-most vertex) never occurs —
orientation is simply flipped for every part. Any downstream logic assuming a
consistent left-to-right ordering is operating on flipped data.
*Fix:* make only one branch swap (or delete the conditional and swap by comparing
columns).

**V-2 — Serial skeleton refinement mutates a shared NumPy view (PLAUSIBLE).**
`Skeletonization.py:392-395,448`. `refine_skeleton_segments` slices
`sub_skel = skel[low:up]` — a **view**, not a copy — and
`refine_skeleton_segment` writes `skel[(x,y)] = False`. Parts whose ±5 px
bounding boxes overlap therefore erase one another's pixels, and the result
differs between the serial path (mutates the master skeleton) and the `mp.Pool`
path (each worker gets a pickled, isolated copy). Non-deterministic output
depending on worker count. *Fix:* `sub_skel = skel[low:up].copy()`.

### Medium

**V-3 — `min_length` divided by 4 twice (verified).**
`Skeletonization.py:60` computes `min_length = floor((config.min_length/4)/px)`,
then `:79` passes `floor(min_length/4)` to `find_skeleton_segments`. The initial
segment-length gate is therefore ~`min_length/16` in metres instead of the
intended value — near-noise fragments survive the first filter (they are only
culled later at the `min_length` check in refinement, `:510`). *Fix:* pass
`min_length` (not `/4`) at `:79`, or collapse to one intended scaling.

**V-4 — Padding pixel-size mismatch for non-square pixels.**
`find_segments` pads using `int(max_tree_height/px_x)+1` (`Skeletonization.py:61`,
`transform[0]` only) but `restore_geoinformation` un-pads with
`int(max_tree_height/max(px_x,px_y))+1` (`Vectorization.py:469`). When
`px_x != px_y` these differ, producing a constant world-coordinate offset on
every stem. *Fix:* use the same padding expression in both places.

**V-5 — `clean_diameter` skips the second-to-last node.**
`Quantification.py:196`. `range(1, len-2)` covers `1 … len-3`; index `len-2` is
never smoothed, so an outlier diameter there survives into the volume integral.
*Fix:* `range(1, len-1)`.

**V-6 — Contour diameter path is O(stems × polygons) with repeated pickling.**
`Quantification.py:152-156,306`. The full polygon `GeoDataFrame` is passed as an
`apply_async` argument per stem (re-pickled each time) and `calc_d` intersects
against the entire GeoSeries with no spatial index. Slow on dense scenes.
*Fix:* build an STRtree over the polygons once; query per node.

**V-7 — Split angles are hard-coded, not tied to Config.**
`Skeletonization.py:456,471,486` use 10°/30° independent of
`config.tolerance_angle=7`. Tuning the config has no effect on refinement splits.

### Low

- **V-8 — `rebuild_endnodes_from_stems` is a no-op.** `Vectorization.py:405-417`
  returns a node list that the caller discards; mutates nothing.
- **V-9 — EDT worker-split branch is a copy-paste no-op.**
  `Quantification.py:111-126`: the `workers<=1` and `else` branches run identical
  serial code.
- **V-10 — `Geometry.ang()` sign correction is unreachable.**
  `Geometry.py:26-28`: `arccos ∈ [0,180]`, so `% 380` is a no-op and the
  `>180 → -360` branch never runs. Callers' `abs(ang(...))` are therefore
  redundant; any expectation of a *signed* angle is false.
- **V-11 — `stem_binary_threshold` config is ignored.** `_as_binary_mask`
  hard-codes `>= 0.5` (`Skeletonization.py:30`, `Quantification.py:30`) instead of
  reading `config.stem_binary_threshold`.
- **V-12 — `stem_id` is always -1 in exports.** `Stem.get_nodes/get_vectors`
  read `getattr(self,'stem_id',-1)` (`Stem.py:64,79`) but `Stem` never defines
  `stem_id`; `get_vectors` also sets `geom` to the whole path while labelling it
  per-node.
- **V-13 — Meter-unit magic thresholds assume a projected CRS.** node-proximity
  `0.01` (`Quantification.py:310`), dedup buffers `0.3`
  (`Vectorization.py:87,428,443`) break for a degree-based CRS.
- **V-14 — Deprecated import.** `import scipy.ndimage.measurements`
  (`Skeletonization.py:11`) was removed in SciPy ≥ 1.14.
- **V-15 — Misleading parameter names.** `restore_geoinformation` and
  `connect_stems` name their argument `stems`, but at those stages the objects
  are `Part`s.

---

## Prediction & orchestration
`winmol_run.py`, `classes/ExecutionPlan.py`, `classes/HardwareInfo.py`,
`utils/Prediction.py`, `utils/PredictWorkers.py`

### High

**P-1 — A multi-GPU worker crash hangs the parent forever (PLAUSIBLE).**
`PredictWorkers.py:221`. `prediction_worker` has no `try/except`; any exception
(OOM, model-load failure, read error) kills the process **without** emitting the
`{'done': True}` sentinel. The parent's `while finished < len(workers):
result_q.get()` (`:520-523`) then blocks forever — `join()`/`finalize_raster`
never run. *Failure:* one GPU OOMs on a large scene → the whole job hangs with no
error and no output. *Fix:* wrap the worker body; always emit a done/error
sentinel in `finally`.

**P-2 — Multi-GPU path has no OOM / adaptive-batch fallback (PLAUSIBLE).**
`PredictWorkers.py:149`. `_predict_batch` calls `predict_on_batch` directly,
unlike the streaming path's `_predict_batch_adaptive` (`Prediction.py:289`) which
halves the batch on `ResourceExhaustedError`. Combined with P-1, an OOM is
unrecoverable and silent.

**P-3 — GPU memory tiering is dead; batch size is effectively hard-coded
(PLAUSIBLE).** `ExecutionPlan.py:233-238,286-293`. The mem-tiered `default_batch`
(2/4/8 single-GPU; 3/4/6/8 multi-GPU) is overridden because `Config` always
defines the override attributes: single-GPU uses
`prediction_batch_gpu` = **4**, multi-GPU uses `prediction_batch_multi_gpu`
= **12** (`Config.py:21,23`). A 12/16 GB card still gets batch 4; every multi-GPU
worker gets batch 12 regardless of VRAM → OOM on modest cards → P-1/P-2 hang.
*Fix:* let the tiered default win unless the override is explicitly set (use
`None` sentinels).

### Medium

**M-P1 — `CUDA_VISIBLE_DEVICES` restriction ignored by workers.**
`winmol_run.py:154` sets `gpu_ids=list(range(plan.gpu_workers))` (absolute
0..N-1); each worker then sets `CUDA_VISIBLE_DEVICES=str(gpu_id)`
(`PredictWorkers.py:229`), overriding an inherited restriction. *Failure:* user
sets `CUDA_VISIBLE_DEVICES=4,5,6,7` to share a node — `detect()` counts 4, but
workers bind physical GPUs 0-3, colliding with other users.

**M-P2 — Apple Silicon / non-NVIDIA GPUs never used.** GPU detection is
`nvidia-smi`-only (`HardwareInfo.py:52-93`); on Metal `gpu_count=0` → always
`cpu_stream`, despite the repo's `tensorflow-metal` setup. Every "GPU" plan is
unreachable on macOS.

**M-P3 — Autotune predicts sample tiles twice.** `Prediction.py:580,597`. When
`done==0`, `_autotune_batch_size` runs full inferences on the first `chunk_size`
tiles and **discards** the results, then the same tiles are re-predicted and
written. Wasted GPU time (up to the candidate sweep × repeats); output correct.

**M-P4 — Temp raster leaks on failure.** `Prediction.py:667`,
`winmol_run.py:560`. Output is written to `IO.atomic_tmp_path` and only
`finalize_raster`-d at the very end; any exception before finalize leaves the tmp
file behind uncleaned.

### Low

- **L-P1 — Large dead-code surface.** `trees_processing` (`winmol_run.py:175`);
  the entire service path `start/stop_multi_gpu_prediction_service`,
  `predict_jobs_multi_gpu`, `prediction_service_worker`
  (`PredictWorkers.py:264-458`); legacy `predict`,
  `predict_with_resampling_per_tile`, `predict_stream_single_gpu`,
  `predict_stream_cpu`, `predict_with_resampling_stream_to_raster`
  (`Prediction.py:676-816`). Confusion / maintenance risk.
- **L-P2 — `img_width` used for the Y axis.** `ExecutionPlan.py:101`,
  `Prediction.py:64,83`. Harmless only because `img_width == img_height == 512`;
  latent bug if they ever differ.
- **L-P3 — `total_ram_gb` detected but never used** in planning
  (`winmol_run.py:97`); producer count / batch are not RAM-capped.
- **L-P4 — Undocumented magic numbers throughout the planner**
  (`ExecutionPlan.py`): GB thresholds 20/12/40/70, tile thresholds
  800/1000/1200/2500, ratios 0.75/`//3`/`//4`/`//5`. Untested.
- **L-P5 — `_force_tensorflow_cpu_only()` failure is swallowed.**
  `winmol_run.py:50-51`. If the TF context is already initialized,
  `set_visible_devices([],'GPU')` raises, is caught + printed, and prediction
  **silently runs on GPU** despite `prediction_backend='cpu'`.

**Verified non-issues:** `run_multi_gpu_prediction` positional args map correctly
(`output_raster=stem_path`); the vector worker split is honoured — the passed
`plan.cpu_workers` is the total budget and `process_prediction_tiles` reads
`config.vector_tile_workers`, so there is no over-subscription.

---

## Tiling & vector merge
`utils/Tiling.py`, `utils/VectorTilePipeline.py`, `utils/IO.py`

### High

**M-1 — Stems within `edge_buffer_m` of the mosaic outer perimeter are dropped.**
`IO.py:932-943,1024-1026`. `_raster_filter_geom` shrinks the tile box uniformly
on all four sides by `edge_buffer_m` (= `tile_overlap_m`, 12 m). For edge tiles
the halo is clamped to the mosaic boundary (`Tiling.py:50-53`), so on the outer
side the shrink eats *real* data. The only tile responsible for that strip keeps
just `intersects(inner)`, excluding stems within 12 m of the mosaic border, and
they are never added to the edge-reconstruction pool (which draws from the
already-filtered set). *Failure:* any stem within 12 m of the orthomosaic frame
vanishes from output. *Fix:* filter against the true inner window (as
`process_tile_gpkg` does), not the raster box shrunk in CRS space; or don't
shrink sides that coincide with the mosaic edge.

**M-2 — Interior-seam stems may be double-counted / fragmented (PLAUSIBLE).**
`IO.py:1447-1456`, docstring `:1391-1394`. Adjacent tiles each fully vectorize the
shared halo, producing two overlapping copies of every seam-straddling stem.
Dedup relies entirely on `Vec.connect_stems` reconciling them in the pooled edge
pass, and the code deliberately avoids stronger reconciliation. `connect_stems`
joins endpoint gaps but is not guaranteed to collapse two overlapping full/partial
duplicates → inflated stem counts and volume totals at seams. *Needs an empirical
check* of `connect_stems` on overlapping duplicates.

### Medium

**M-3 — CRS mismatch between filter geometry and stems.** `IO.py:1066,1083-1087`
(and `:1409-1415`). `filter_geom` is taken in the raster's `src.crs`, but stems
are reprojected to `target_crs` before `intersects`. If a tile's stem CRS differs
from its raster CRS, the spatial filter runs in mismatched coordinates → mass drop
or wrong keep. *Fix:* reproject `filter_geom`/edge band into `target_crs` first.

**M-4 — Silent "no-dedup" fallbacks cause double counting.** `_raster_filter_geom`
returns `(None,None)` on any exception/missing raster (`IO.py:942-943`); with
`filter_geom=None`, `_globalize_stems` keeps **all** stems (`:1034-1039`) → every
seam stem doubled. For a **geographic CRS**, `buffer(-abs(12))` shrinks by 12
*degrees* → empty → falls back to the full box (`:939-940`) → no filtering. Both
degrade silently.

**M-5 — `intersects` boundary rule is touch-inclusive.** `IO.py:1025`. A stem
merely touching the inner boundary is kept by both neighbouring tiles,
reinforcing seam double counting. A "centroid within inner" / half-open rule
gives clean one-tile-owns-it ownership.

**M-6 — Binary tiles written as float32.** `write_tile_raster` (`IO.py:145-156`)
inherits `build_safe_prediction_profile`'s `dtype='float32'` (`:70`) yet writes
`astype(np.uint8)`, so 0/1 tiles cost 4× disk/IO. *Fix:* pass `dtype='uint8'`.

### Low

- **M-L1 — Dead / wasted merge work.** `merged_nodes`/`merged_vectors` are
  accumulated (`IO.py:1514-1522`) but never passed to `_write_merged` (nodes and
  vectors are regenerated from stems). `merge_selected_tile_results` /
  `process_tile_gpkg` (`:1103,1135-1211`) appear unused; the former
  unconditionally `os.remove`s every input gpkg (`:1186-1190`) even for tiles
  that failed to parse.
- **M-L2 — Inconsistent reporting.** The loop computes `total_stems`
  (`IO.py:1520`) but the summary prints `final_stem_count` from reconstruction
  (`:1561-1562`); the two can disagree, obscuring dedup effects.
- **M-L3 — misc.** `_detect_tiles` raises "Multiple GPKG candidates" if a
  `_new`/`_new_{pid}` fallback file lands in `work_dir` (`:970-974`);
  `load_model_from_path` collapses all load failures into one generic
  `RuntimeError`, discarding tracebacks (`:206-220`); `_fiona_write_layer` append
  mode infers schema from the current gdf only (`:686,720`).

---

## Cross-cutting themes

1. **Dead code is extensive.** Whole service-based multi-GPU path, several legacy
   `predict_*` functions, `trees_processing`, `rebuild_endnodes_from_stems`, and
   copy-paste no-op branches. Removing them would shrink the surface area
   materially and prevent confusion about which path actually runs.
2. **Config knobs that don't do anything.** `stem_binary_threshold` (V-11),
   `tolerance_angle` on refinement (V-7), and the GPU batch tiering (P-3) are all
   silently overridden or ignored — tuning them has no effect, which is a trap
   for operators.
3. **Silent degradation over loud failure.** Several paths swallow errors and
   fall back to a worse-but-quiet behaviour: CPU-forcing (L-P5), dedup fallbacks
   (M-4), multi-GPU worker crashes (P-1). Prefer failing loudly.
4. **CRS assumptions.** Multiple metre-unit magic distances and one metre-vs-degree
   buffer bug (V-13, M-3, M-4) assume a projected CRS; a geographic CRS silently
   misbehaves.
5. **No automated tests.** Per `CLAUDE.md`, flake8 is the only CI check. The
   heuristics above (V-1..V-5, M-1..M-2) are exactly the kind of logic that a
   small fixture-based test suite (one synthetic mask → expected stem count /
   total volume) would have caught.

### Suggested priorities
1. Fix V-1 (identical branch) and V-3 (double `/4`) — verified, cheap, affect
   every run.
2. Fix M-1 (perimeter stems dropped) — silent data loss.
3. Address P-1/P-3 before relying on the multi-GPU path.
4. Add a minimal end-to-end fixture test to lock the above in.
