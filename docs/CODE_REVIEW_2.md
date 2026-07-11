# WINMOL Analyzer — Code Review, Second Pass (deep review)

Follow-up to [`CODE_REVIEW.md`](./CODE_REVIEW.md) (2026-07-09). Method: 11
lens-specific analysis agents read every core module in full (seeded with all
36 first-pass findings to avoid rediscovery), producing 120 candidate findings
→ 90 after dedup. Verification was split between an adversarial agent pass
(interrupted by rate limits after 9 confirmations) and direct line-by-line
inline verification of the ~25 highest-impact claims against source. Every
finding below marked **verified** was confirmed against the actual code;
§D lists candidates whose verification is still pending.

Categories: **bug** (wrong behavior) · **misleading** (code says one thing,
does another) · **simplify** / **optimize** (same output, less code / faster).

---

## A. Confirmed bugs (verified, ranked)

### A-1 · HIGH — Tiled merge crashes with `IndexError` whenever a cross-seam
### stem connection succeeds — and the crash then destroys all tile outputs

> **STATUS: FIXED 2026-07-11.** Confirmed in production first (twostage UNet
> full-ortho run: 1057 dense stems → edge joins fired → IndexError → work dir
> deleted). Fix: `Vectorization._merge_diameter_lists` merges the parents'
> per-node diameters into the merged candidate (in-tile behavior unchanged —
> lists are empty there; golden suite stayed green), plus a guard at
> `IO.py` merge quantification, plus A-15's rmtree removed (work dir is now
> kept on failure). Pinned by `test_merge_diameter_lists_*` in test_units.py.

**The chain (all verified):**
1. `calc_connectivity_votes` builds the merged stem as
   `candidate = _clone_stem(stems0); candidate.path = new_path`
   (`Vectorization.py:299-301`). `_clone_stem` copies the **base's**
   `segment_diameter_list` verbatim (`Vectorization.py:40-51`), while the
   merged path now contains base + bridge + slave coordinates.
2. In-tile this is harmless — `quantify_stems` re-measures diameters from the
   raster (`get_diameters`) before `quantify_stem` runs.
3. But the merge phase calls `Quant.quantify_stem(stem)` **directly**, with no
   raster available (`IO.py:1481-1482`). `quantify_stem` indexes
   `segment_diameter_list[i+1]` for every consecutive path pair
   (`Quantification.py:167-172`). Merged path coords > stale diameter count →
   guaranteed `IndexError`.
4. The exception propagates out of `run_vector_phase`, whose handler does
   `shutil.rmtree(work_dir)` (`winmol_run.py:250-253`) — **deleting every
   completed per-tile GeoPackage** before re-raising. Total loss of the vector
   phase.

Every observed successful run had `"0 stem segments appended to other stems"`
at merge time — the crash path simply hasn't fired yet. The first mosaic where
two seam fragments genuinely connect dies at the last step and deletes its own
partial results. *(Independently flagged by 5 of 11 analysis lenses.)*

**Fix:** after merge-time `connect_stems`, rebuild per-node diameters (either
skip `quantify_stem` for merged stems, carry per-node diameters through the
merge by resampling both parents' node lists onto the new path, or persist the
stem-map raster into the merge phase and re-run `get_diameters`). Separately:
never `rmtree` on failure — keep the work dir for resume/debugging.

### A-2 · MEDIUM — `/255` normalization for every integer dtype; floats pass unscaled

`_to_float32_image` (`Prediction.py:49-54`): any integer dtype is divided by
255 — a **uint16** orthomosaic (0–65535) enters the U-Net as 0–257; any
non-float32 float raster is passed through **unscaled**. Identical logic in the
standalone loader (`IO.py:227`). The models were trained on [0,1]; predictions
on 16-bit imagery are garbage with no warning.
**Fix:** dtype-aware scaling (`uint8→/255`, `uint16→/65535`, float→clip/assert
range) plus a loud warning on unexpected ranges.

### A-3 · MEDIUM — Tile fencepost: source windows are `px_per_tile−1` wide but
### the output transform assumes `px_per_tile`

`_iter_tile_jobs` reads `src_width = px_per_tile_x - 1`
(`Prediction.py:92-93`), yet `out_transform` scales by
`px_per_tile_x / img_width` (`Prediction.py:70-75`). Each 512-px output tile
therefore represents `px−1` source pixels while claiming `px`: all content is
stretched by `px/(px−1)` (≈ 0.27 % at 375 px/tile) inside every tile, with
mismatched seams. Every downstream length, diameter and volume inherits the
error. The same `−1` exists in the legacy path (`Prediction.py:799-800`).
**Fix:** read the full `px_per_tile` window (the `−1` looks like an inherited
slicing-idiom mistake).

### A-4 · MEDIUM — Run-to-run nondeterminism from string-salted `Part.__hash__`

`Part.__hash__` hashes a tuple containing **strings**
(`classes/Part.py:19-23`); Python salts string hashes per process
(`PYTHONHASHSEED`). `find_skeleton_segments` returns `set(parts)`
(`Skeletonization.py:349`) and `build_stem_parts` does `set(segments)`
(`Vectorization.py:391`), so **stem ordering differs every run**. The greedy
`connect_stems` loop (base pick + first-wins merges) is order-sensitive →
identical inputs can produce different stem counts/volumes across runs.
**Fix:** return sorted lists (e.g. by `(start, stop)`), or drop the strings
from the hash and never iterate sets into ordered pipelines.

### A-5 · MEDIUM — Nodata mask tests only the red channel (standalone/notebook path)

`mask = np.where(img_pd[:, :, 0:3] == (0, 0, 0), False, True)[:, :, 0]`
(`Prediction.py:789`) keeps only the **red**-channel comparison: any pixel with
R==0 is erased from the stem map regardless of G/B (dark shadows, green-heavy
pixels). Intended semantics is "all three bands zero".
**Fix:** `mask = ~np.all(img_pd[:, :, 0:3] == 0, axis=2)`.

### A-6 · MEDIUM — Config overrides applied with no type checking

`apply_env_config_overrides` does raw `setattr` (`winmol_run.py:86-91`).
`WINMOL_CONFIG_OVERRIDES_JSON='{"tile_size": "20"}'` (string, not number)
survives until tile math and dies with an opaque `TypeError`; wrongly-typed
bools/None silently poison planner comparisons.
**Fix:** coerce to `type(getattr(Config, key))` and fail fast with the key name.

### A-7 · MEDIUM — `CUDA_VISIBLE_DEVICES` UUID/MIG forms silently downgrade to CPU

`_apply_cuda_visible_devices` parses tokens with `int()` and **skips**
non-integer tokens (`classes/HardwareInfo.py:112-124`); NVIDIA's documented
`GPU-<uuid>` / `MIG-…` forms parse to zero indices → `[], []` → `gpu_count=0`
→ CPU-only plan, silently.
**Fix:** treat unparseable tokens as "restriction present but unresolvable" —
keep GPUs visible or warn loudly, don't zero them.

### A-8 · MEDIUM — `geopandas==0.14.0` pinned, pandas/fiona unpinned

`requirements/base.txt` pins geopandas 0.14.0; the active env has
**pandas 3.0.3** (unsupported by geopandas 0.14, which predates it by ~2
years). This is not hypothetical: `gpd.read_file` in this env crashes with
`AttributeError: module 'fiona' has no attribute 'path'` (observed in this
repo on 2026-07-10), and only pyogrio-engine calls work.
**Fix:** pin compatible ranges (`pandas<3`, `fiona<1.10` for gpd 0.14 — or
upgrade geopandas to 1.x and require pyogrio).

### A-9 · MEDIUM — Per-stem quantification failures silently delete stems

`error_callback` is print-only (`Quantification.py:104-105`); on any exception
in `calc_v_d_contour`/`calc_v_d_edt` (serial or pool: `:111-158`) the
`return_callback` never runs, so the stem is **absent from `measured_stems`**
and vanishes from the output GeoPackage with only a console line.
**Fix:** re-append the stem with NaN diameters (visible in output) or fail the
tile; count and report dropped stems in the summary.

### A-10 · MEDIUM-LOW — `calc_d` returns 0 when no chord passes the node; zeros
### survive into volumes

`d = 0` default (`Quantification.py:303-317`): if the 2 m probe misses the
mask polygons — routine for `connect_stems` **bridge nodes over occluded
gaps** — the node gets diameter 0.0. `clean_diameter` only fixes interior
outliers when `n > 4` and both fences trigger; runs of zeros widen the IQR
fence and survive → systematic volume underestimate, silently.
**Fix:** return NaN for "no measurement", interpolate NaNs explicitly, and
never treat 0 as a measurement. (Related cap: a diameter can never exceed the
probe length `2·diameter_vector_half_length_m` = 2 m — clipped without
warning; `Quantification.py:244-256`.)

### A-11 · LOW-MED — Only `start↔stop` end pairings are ever tried

The candidate filter accepts only `base.start↔cand.stop` or
`base.stop↔cand.start` (`Vectorization.py:158-165`); no orientation flip is
attempted anywhere. Whether two fragments of one tree can join therefore
depends on their (arbitrary — see A-4, and first-pass V-1) trace orientation:
head-to-head / tail-to-tail fragments are unjoinable.
**Fix:** also test same-end pairings with a reversed candidate path.

### A-12 · LOW-MED — 0.3 m corridor dedup can delete genuinely distinct stems

`base.path.buffer(0.3).contains(candidate.path)`
(`Vectorization.py:87-100`, again in `remove_duplicates:440-455`): a distinct
parallel stem lying within 0.3 m of another's centerline — stacked/parallel
logs are common in windthrow — is silently removed as a "duplicate". The 0.3 m
is hard-coded (also first-pass V-13).
**Fix:** require endpoint proximity or high overlap ratio, not mere corridor
containment; make the radius a config knob tied to expected stem diameter.

### A-13 · LOW — `clean_diameter` details

(a) Stems with exactly 3 or 4 diameters compute quantiles then ignore them —
the `if n > 4` guard skips all correction (`Quantification.py:190-220`).
(b) The outlier interpolation divides by the **straight-line chord** `i−1→i+1`
while the weights use **path segments**; on a bent path (up to 30° by design)
the replacement overshoots by `(s1+s2)/chord` (a few %) —
(`Quantification.py:200-209`). *(First pass called the weights correct; the
denominator subtlety is new.)*

### A-14 · LOW — Legacy tiler swaps row/col tile sizes

Rows are sliced with `px_per_tile_x` (an **x**/column quantity) and columns
with `px_per_tile_y` (`Prediction.py:793-800`). Latent while pixels are
square; wrong for anisotropic rasters (same family as first-pass L-P2, new
site).

### A-15 · LOW — Failure amplification: work dir deleted on any vector-phase error

`except Exception: shutil.rmtree(work_dir)` (`winmol_run.py:250-253`). Per-tile
errors *are* caught inside the pipeline (`VectorTilePipeline.py:310-321`), but
anything escaping (e.g. A-1, a merge bug, ctrl-C timing) deletes hours of
completed tile results. No resume capability exists.

### A-16 · LOW — `winmol_batch` invokes `winmol_run.py` relative to CWD

`command = [sys.executable, "-u", "winmol_run.py", ...]`
(`winmol_batch.py:76-84`) while `config.json` is resolved relative to
`__file__`: batch runs only work from the repo root.
**Fix:** `os.path.join(os.path.dirname(__file__), "winmol_run.py")`.

### A-17 · LOW — `FeatureFactory` calls properties as methods

`stem.length()` / `stem.volume()` (`qgisutil/FeatureFactory.py:55-56`) but both
are `@property` (`classes/Stem.py:34,47`) → `TypeError: 'float' object is not
callable`. Currently unreferenced anywhere in the repo — shipped broken; will
crash the moment the QGIS plugin wires it in.

### A-18 · LOW — Standalone `run_pipeline` drops `config` at quantification

`Quant.quantify_stems(stems, pred, profile)` — no `config`
(`standalone/WINMOL_Analyzer.py:77`) → `diameter_method`, probe length and
worker settings are silently defaulted in the standalone path (and in the
notebooks, which copied the call). Also `output_dir + file_name` string
concatenation (`:81-82`) breaks without a trailing separator.

### A-19 · LOW (waste, not wrongness) — Foreground skip tests the halo window

The vector-phase skip reads the **halo** window and checks `(arr >= 1).any()`
(`winmol_run.py:214-222`): a tile whose halo contains foreground but whose
inner window is empty is fully vectorized, then 100 % discarded by the merge
filter.

---

## B. Claims investigated and refuted (recorded to prevent re-flagging)

| Claim | Why it's wrong |
|---|---|
| EDT world→pixel uses `round()` instead of `floor()` (`Quantification.py:269`) | Nodes are placed at pixel **corners** by `restore_geoinformation`; `round()` recovers the exact index for corner-placed points (floor would be off-by-one under negative float jitter). Consistent by design. |
| Frustum volume 4× too large (diameters used as radii) | `calc_l_v` halves correctly: `(d1/2)**2 + (d1/2)*(d2/2) + (d2/2)**2` (`Quantification.py:320-324`). |
| `calc_vote` angle factor "cancels out", ranking = span²+gap² (`Vectorization.py:362`) | The factor `(1+Σangles)/tolerance` differs per candidate, so it does **not** cancel across the ranking. (It *is* needlessly written as two identical-factor terms — see C-list.) |
| `calc_vote` args swapped at the second call site | Intentional: the swap maps the body's `stems0.stop.distance(stem.start)` onto the correct gap for that branch (verified both sites). Misleading naming only. |
| Legacy path stitches raw floats into uint8, binarizing at 1.0 | `_predict_batch_core` binarizes at `stem_binary_threshold` **before** the uint8 buffer assignment (`Prediction.py:159-178`); thresholds are consistent across all three run paths. |
| One failing vector tile kills the whole pool | Per-tile work is wrapped in `try/except` returning an error result (`VectorTilePipeline.py:310-321`). The real issue is the caller's rmtree (A-15). |

---

## C. Simplification & optimization proposals

### Verified by adversarial judges (before the verification pass was interrupted)

1. **[optimize/med] Drop the 801-px physical padding** —
   `find_segments` pads by `max_tree_height/px + 1` per side
   (`Skeletonization.py:61-67`). The pad is constant `False` and is used only
   as a coordinate offset that `restore_geoinformation` subtracts again; no
   raster operation needs the pixels. Skeletonize + node-finding therefore run
   on 2–6× the necessary area (2048² tile → 3650²). Replace with pure
   coordinate arithmetic (or a 1–2 px pad for kernel safety): identical
   output, phase-level 2–6× saving.
2. **[optimize/med] Rewrite `get_neighbors`** (`Skeletonization.py:533`) —
   allocates an 8×2 ndarray + fancy-indexing **per skeleton pixel**; measured
   13.6 µs/call vs 0.99 µs for a plain 8-tuple loop (13.8×). It sits in both
   tracing hot loops; pure-Python version is byte-identical and ~10-30× faster
   tracing.
3. **[optimize/med] Kill the full-window `temp` arrays in the refine walk**
   (`Skeletonization.py:445,463,467,484,499`) — `np.full(skel.shape, False)`
   per accepted measuring point plus whole-window `np.where` restores; a list
   of visited pixels makes both O(step) instead of O(window). Line 467's
   allocation is provably dead (branch exits the loop).
4. **[simplify/low] Delete dead endpoint machinery** — `find_skeleton_segments`'
   `end_nodes`/`padding`/`config` params are unused; `get_nodes`' endpoint scan
   feeds only a (stale, post-re-skeletonize) print (`Skeletonization.py:298-310`).
   The line-112 re-skeletonize itself **is** needed (thins L-triangles left by
   branchpoint removal) — resolves the `TODO is this code correct?`.
5. **[simplify/low] Closed loops are traced then always discarded** —
   ring parts have `start == stop`, so refine's `while w != z:` never runs and
   the part is always dropped at the length gate
   (`Skeletonization.py:446,510`). Either skip `_trace_loop` entirely or split
   rings properly.
6. **[simplify/low] Shapely 1.x fallback is dead** on the pinned Shapely 2.0.2
   (`Vectorization.py:65-78` — the geometry-identity branch).
7. **[simplify/low] `remove_duplicates`' `stems0` branch is dead**; the return
   annotation is wrong (returns a tuple) (`Vectorization.py:421-438`).
8. **[simplify/low] Duplicate angle recompute** in the second connect branch
   (`Vectorization.py:319-321` recomputes `ang_mp_l_st`).
9. **[simplify/low] `_worker_count` + `multiprocessing` import are dead** in
   `Vectorization.py`.

### High-value proposals pending verification (flagged, not yet judged)

- **[optimize/med] Move `matplotlib` out of `utils/IO.py` module level** — it
  serves only the dead `save_image`; every spawned worker pays the import
  (~0.5 s, ~70 MB).
- **[optimize/med] Stop spawning a fresh `mp.Pool` per tile per stage**
  (`Quantification`, `Skeletonization` refine) — ~1.6 s spawn overhead each
  under macOS spawn semantics, often exceeding the work.
- **[optimize/med] `features.shapes(…, mask=None)` polygonizes the background
  too** (`Quantification.py:128-137`) — pass the foreground mask; skip
  polygonizing ~90 % of the raster. Same for full-raster EDT when only a few
  hundred node pixels are sampled.
- **[optimize/med] `ExecutionPlan` hard-caps the vector phase at 4 tile
  workers** (`max_vector_tile_workers=4`) regardless of a 24-core host.
- **[optimize/low] STRtree rebuilt per `global_change` cycle;
  O(n²) `remove_duplicates`** — index once, invalidate incrementally.
- **[simplify] Delete dead IO helpers** (`iter_prediction_tiles`,
  `read_window`, `write_window`, `create_output_raster_like`,
  `load_orthomosaic_with_resampling`, `save_image`) and the legacy
  `predict_*` family — several hundred LOC.

---

## D. Remaining candidates, pending verification

The multi-agent verification pass was interrupted by session rate limits
(9 of 90 verified; ~25 more verified inline above). Notable still-unverified
candidates (see `scratchpad wf_candidates.json` for the full 121):

- `IO.py:757` — `winmol_gpkg_*` temp dirs leak into the output tree on write
  failure and later abort merges with "Multiple GPKG candidates".
- `IO.py:966` — merge ownership derived by negative-buffering halo bounds
  instead of using the exact `TileJob.inner` windows (root of first-pass M-1).
- `IO.py:1454` — `stem_id`/`tile_id` provenance built by `_globalize_stems`
  never reaches the merged output.
- `IO.py:1308` — `is None` checks miss NaN from GPKG NULLs → `Point(nan, nan)`.
- `IO.py:833` — `with_suffix('.gpkg')` mangles dotted basenames
  (`site_2.5cm` → `site_2.gpkg`).
- `VectorTilePipeline.py:254` — per-tile logs swallowed on success, making
  `vector_summary_log` largely dead.
- `Prediction.py:101` — outer `overlap_pred//2` border of the stem map is
  never predicted (always 0).
- `Prediction.py:511` — `total_tiles` ignores an external `tile_jobs` subset
  (progress/ETA wrong).
- `Prediction.py:609` — stream write loop may bypass the adaptive-OOM helper.
- `Skeletonization.py:319` — node-definition mismatch: branchpoint removal
  uses crossing-number, tracing uses degree≠2.
- `Skeletonization.py:390` — `l_bound−5` can go negative → wraparound slice.
- `Skeletonization.py:450` — refine re-walks the window taking the first
  neighbor arbitrarily instead of following the traced path.
- `Skeletonization.py:456` — final-leftover angle test on 1–3 px chords
  (~45° quantization).
- `HardwareInfo.py:123` — Metal fallback trusts package metadata; TF may still
  see no GPU (plan says `stream`, inference silently runs CPU).
- `classes/ExecutionPlan.py:309`, `classes/Config.py:16,66`,
  `classes/Stem.py:20,63`, `classes/Timer.py:24` — assorted misleading/dead
  knobs (`stem_map_binary` is never read anywhere).

**To finish verification:** re-run the review workflow with
`resumeFromRunId: wf_69854b19-beb` after the rate-limit window resets — all
finder results and completed verdicts replay from cache; only the ~80 failed
verifier agents and the completeness critic run.

---

## E. Suggested remediation order

1. **A-1** (merge IndexError + rmtree amplification) — the only known
   crash-and-destroy-output path; fires on the first mosaic where seam stems
   actually connect.
2. **A-2** (dtype-blind normalization) — silent garbage on 16-bit imagery.
3. **A-4** (nondeterminism) — undermines reproducibility of every scientific
   result; trivial fix (sorted lists).
4. **A-3** (fencepost stretch) — small but systematic bias on *all* geometry.
5. **A-8** (dependency pins) — already breaking in practice.
6. **A-9/A-10** (silent stem drops / zero diameters) — silent volume bias.
7. C-1/C-2/C-3 (padding, `get_neighbors`, refine temps) — the vector phase's
   dominant costs; all output-identical.
8. First-pass M-1 + A-19 together — redo merge ownership on exact inner
   windows (fixes perimeter loss and halo-only waste in one change).
