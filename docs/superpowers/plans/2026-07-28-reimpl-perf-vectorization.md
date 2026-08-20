# reimpl/perf-vectorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Program spec: `docs/superpowers/specs/2026-07-28-reimpl-off-main-design.md`.

**Goal:** Feature 4: the essential code of old #3 (`origin/perf/resample-and-edge-fix`)
and #9 (`origin/perf/vector-index-hoist`) — read-side cubic resampling,
boundary-stems edge fix, and vector-phase index hoisting (11.7× e2e) — adapted
to this branch. ~400 LOC gross target. Branch `reimpl/perf-vectorization` off
`reimpl/deterministic-stems` (cc6315b). NO PR.

**Do NOT port:** the reverted writer-thread pair (49949dc/b604647), bench
harnesses, docs commits, superseded 1650793.

**Order constraint:** Tasks 1–2 are results-changing (cubic-read drift +
intentional boundary-stems recovery) and land BEFORE Task 3, which must be
bit-identical on top of them.

## Task 0: branch + plan commit
- [ ] `git checkout -b reimpl/perf-vectorization reimpl/deterministic-stems`
- [ ] commit this plan.

## Task 1: resample in the GDAL read (from 47882a4 + 0477afe, ~40 LOC)

**Files:** Modify `utils/Prediction.py` (TileBatchProducer :217-263, producer
ctor call :540-549, predict_stream_to_raster passes out_size), `utils/IO.py:244`
(bilinear→cubic one-liner in load_orthomosaic_with_resampling); Test extends
`tests/test_prediction_tf_free.py`.

- TileBatchProducer gains `out_size=(h,w)` — reads imagery with
  `out_shape=(bands, oh, ow)` + `Resampling.cubic`, masks with
  `Resampling.nearest`, so tiles arrive on the model grid and the existing
  `_resize_batch` identity fast path (:138-140, currently dead) short-circuits
  — the read-side resample REPLACES the skimage path, not composes with it.
  Reference: `git show 47882a4` / `git show 0477afe`.
- Cubic not bilinear is load-bearing: bilinear thins masks ~6% at the 0.5
  threshold (~50 stems/scene lost). GDAL cubic (Catmull-Rom) vs skimage
  order=3 drift is expected & absorbed by the threshold (documented risk;
  parity re-check happens in the user's e2e round).
- Tests: producer-level — a synthetic GeoTIFF read through TileBatchProducer
  with out_size yields tiles of exactly (512,512) and the batch path hits the
  identity fast path (assert `_resize_batch` returns same object / or assert
  shapes+dtype and that no skimage resize occurs via monkeypatch counter).
- TDD; flake8; suite green; commit
  `perf(predict): resample tiles in the GDAL read — cubic, model-grid tiles`.

## Task 2: boundary-stems edge fix (from 153759d, ~35 LOC)

**Files:** Modify `utils/IO.py` (`_raster_filter_geom` :935-946 per-side
logic; merge :1469-1512 reads ortho bounds from the UNUSED `stem_map_path`
param :1474 and threads `ortho_bounds` to `_process_tile` :1069),
`winmol_run.py:236-241` (pass `stem_map_path=self.stem_path`); Test NEW
`tests/test_edge_filter_geom.py` (~50 LOC, pure geometry, no fixtures).

- `_raster_filter_geom` currently shrinks ALL four tile sides by
  `edge_buffer_m` for seam dedup; sides on the ortho's true outer boundary
  have no neighbour, so real stems were dropped (worst at corners). Port the
  per-side logic verbatim (tolerance `edge_buffer*1e-3`): shrink only
  interior-seam sides. Reference: `git show 153759d`.
- Tests: pure-geometry table — interior tile shrinks 4 sides; corner tile
  shrinks only its 2 interior sides; edge tile 3; tile==ortho shrinks none.
- TDD; flake8; suite green; commit
  `fix(merge): keep boundary stems — shrink only interior seam sides`.

## Task 3: vector-phase index hoisting (from 685cb16 + f34c120 + 6e52f76, ~90 LOC)

**Files:** Modify `utils/Quantification.py` (ContourIndex built once in
get_diameters :139-141; calc_d :303-306 queries it, keep the sindex fallback
branch of 685cb16's final form; mp.Pool→ThreadPool :76,:149),
`utils/Vectorization.py` (remove_duplicates :440-455 O(n²)→STRtree alive-list,
order-preserving; STRtree already imported :13), `utils/Skeletonization.py`
(get_neighbors :533-548 numpy→scalar); Tests: port
`origin/perf/vector-index-hoist:tests/test_quantification_workers.py`
(52 LOC, fixture-free) + NEW dedup-equivalence test (~30 LOC: random
geometries, new remove_duplicates output == inline O(n²) reference).

- MUST be bit-identical in results (old stack verified byte-identical output;
  the equivalence test pins the dedup half; ContourIndex returns the same
  intersection set as the full scan by construction).
- ThreadPool not Pool: pickling stems cost 45 s vs 4.6 s serial; GEOS releases
  the GIL (8.4× measured).
- TDD; flake8; suite green; commit
  `perf(vector): hoist spatial indexes — ContourIndex, STRtree dedup, scalar neighbors, ThreadPool`.

## Task 4: verification + push
- [ ] Full suite green; flake8 all touched files; LOC report
  (`git diff --stat cc6315b..HEAD`, target ≈400 gross) + cumulative non-GUI.
- [ ] Push branch. NO PR. Compact ledger checkpoint (no user stop — user
  requested autonomous continuation; full parity check deferred to the user's
  e2e round).
