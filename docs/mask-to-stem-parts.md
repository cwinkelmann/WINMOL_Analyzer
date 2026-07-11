# From U-Net mask to tree stem parts

This document explains, stage by stage, how the WINMOL Analyzer turns a
**binary stem-map raster** (the U-Net segmentation output) into a set of
**individual, quantified tree stems** — each an ordered chain of measuring
points carrying a diameter, length and volume.

It is the "vector phase" of the pipeline. The prediction phase that produces the
mask is covered separately; here the input is a raster where each pixel is
`1` (stem) or `0` (background).

The most legible, un-tiled version of the sequence lives in
`standalone/WINMOL_Analyzer.py:62-77`. In the tiled runtime the same six calls
run per tile inside `utils/VectorTilePipeline.py::_run_vector_pipeline`, then
the tiles are stitched by `IO.merge_and_filter_tiled_results`.

```
find_segments ─▶ restore_geoinformation ─▶ build_stem_parts
      │                                          │
   (pixels)                                  (world coords)
      ▼                                          ▼
  List[Part]                                 List[Stem]
                                                 │
   connect_stems ─▶ rebuild_endnodes_from_stems ─▶ quantify_stems
        │                                              │
   (merged Stems)                              (Stems + diameters,
                                                lengths, volumes)
```

---

## The domain model (`classes/`)

Four small classes form a **pixel-space → world-space → measured** progression.

| Class | Space | Key fields | Role |
|-------|-------|-----------|------|
| `Part` (`Part.py:7`) | pixels | `start`,`stop` `(row,col)`; `path: List[(row,col)]`; `l_bound`/`u_bound` bbox | Raw skeleton trace primitive |
| `Stem` (`Stem.py:11`) | world (CRS) | `start`,`stop`: `Point`; `path`: `LineString`; `segment_{diameter,length,volume}_list` | The reconstructed tree |
| `Node` (`Node.py`) | world | `diameter`, `geom: Point`, `node_id`, `stem_id` | One measuring point (export view) |
| `Vector` (`Vector.py`) | world | like `Node` but `geom` is the whole path | Per-stem export view |

`Part` is the geometry the skeleton produces. It becomes a `Stem` at
`build_stem_parts`. `Node` and `Vector` are **export-only views**, generated from
a finished, quantified `Stem` via `Stem.get_nodes()` / `get_vectors()`
(`Stem.py:54-82`).

> **Note on `Part` identity:** `Part.__eq__`/`__hash__` (`Part.py:15-23`) key on
> `start`/`stop`/bounds only — *not* on `path`. Two different traces that share
> endpoints and bounding box are treated as equal. This matters at
> `build_stem_parts`, which calls `set(segments)` to dedup.

---

## Stage 1 — `find_segments`: mask → skeleton → pixel `Part`s

**File:** `utils/Skeletonization.py:52-91`
**In:** `pred` (H×W mask), `config`, rasterio `profile` · **Out:** `List[Part]` in pixel coords

This is the heaviest stage. It reduces the "fat" stem blobs of the mask to
1-pixel-wide center-lines, splits that skeleton at junctions into individual
traces, and resamples each trace into a clean, gently-curving polyline.

1. **Pixel size & thresholds** (`:59-61`). `px_size` from the affine transform;
   `min_length = floor((config.min_length/4)/px_size)` (pixels);
   `padding = int(config.max_tree_height/px_size)+1`.
2. **Pad** the mask with `False` by `padding` on every side (`:62-67`) so stems
   near the raster edge aren't clipped when skeletonized.
3. **Binarize** via `_as_binary_mask` — a hard `>= 0.5` threshold (`:24-30`).
4. **Skeletonize** with `skimage.morphology.skeletonize` (`:71`) → 1-px skeleton.
5. **Find & clean nodes** (`get_nodes`, `:95-118`):
   - `remove_dense_skeleton_nodes` erodes 2×2 blobs (`:122-135`);
   - iteratively detect skeleton nodes with the **crossing-number** operator
     `A(p)` — a pixel is an *endpoint* when the 8-neighbour transition count is 1
     and a *branch-point* when ≥3 (`find_skeleton_nodes`, `:138-183`) — and zero
     out the 3×3 around each branch-point (`remove_branchpoints_from_skel`,
     `:186-202`) until no branch-points remain, then re-skeletonize
     (`:112`, marked with a `TODO is this code correct?`).
6. **Trace segments** (`find_skeleton_segments`, `:298-354`): mark nodes as
   skeleton pixels whose neighbour-degree ≠ 2, then walk each chain between two
   nodes (`_trace_chain`, `:236-265`; closed loops via `_trace_loop`, `:268-295`).
   Each chain shorter than the length gate is dropped; the survivors become
   `Part`s oriented by row (`_build_part_from_path`, `:219-233`).
7. **Refine** (`refine_skeleton_segments` → `refine_skeleton_segment`,
   `:359-530`): walk each sub-skeleton pixel by pixel and
   - **resample** a node whenever the running distance exceeds
     `measuring_point_spacing` (`:469`), giving evenly spaced vertices;
   - **split** into a new `Part` whenever the local turn angle exceeds a
     hard-coded **10°** near the ends or **30°** in the interior (`:456,471,486`)
     — this breaks a bent skeleton at genuine corners rather than smoothing them
     into one implausible stem.
   Refined parts below `min_length` are rejected (`:510`).

**Config knobs:** `min_length=2.0 m`, `max_tree_height=32 m`,
`measuring_point_spacing_m=0.5 m` (`classes/Config.py`).

---

## Stage 2 — `restore_geoinformation`: pixels → world coordinates

**File:** `utils/Vectorization.py:459-488` · **In/Out:** the same `List[Part]`, mutated in place

Each pixel `(row,col)` is mapped to map coordinates using the raster transform,
undoing the Stage-1 padding:

```
x = left + (col - padding) * px_x
y = top  - (row - padding) * px_y      # note the y-flip: raster rows go down
```

`px_x=|transform[0]|`, `px_y=|transform[4]|`,
`padding = int(max_tree_height / max(px_x, px_y)) + 1` (`:469`). After this the
`Part.start/stop/path` hold real-world coordinates, though the CRS itself is not
attached yet.

---

## Stage 3 — `build_stem_parts`: `Part` → `Stem`

**File:** `utils/Vectorization.py:373-402` · **In:** georeferenced `List[Part]` · **Out:** `List[Stem]`

Normalizes each part's orientation, dedups on `Part` identity via
`set(segments)`, then wraps each survivor in a `Stem`
(`Point(start)`, `Point(stop)`, `LineString(path)`, empty measurement lists).
The `Stem.crs` is left `None` here and set later.

> ⚠️ **Bug (verified):** the `if start[1] >= stop[1]` and the `else` branches
> (`:381-390`) are **identical** — both swap endpoints and reverse the path
> unconditionally. The intended orientation *normalization* (make `start` the
> left-most vertex) never happens; orientation is merely flipped. See
> `docs/CODE_REVIEW.md` (V-1).

---

## Stage 4 — `connect_stems`: join broken / occluded stems

**File:** `utils/Vectorization.py:108-225` · **In/Out:** `List[Stem]`

Fallen stems are frequently broken into fragments by canopy occlusion, other
stems crossing them, or gaps in the mask. This heuristic re-joins fragments that
plausibly belong to the same tree. It is the analytic heart of the pipeline.

- An outer `while global_change` loop rebuilds two `STRtree` spatial indices —
  one over all stem **start** points, one over all **stop** points (`:134-135`).
- For each base stem, buffer both of its ends by `max_distance` (default **8 m**,
  `:145-148`) and gather candidate stems whose *opposite* endpoint falls inside
  a buffer (`:150-165`).
- Score each candidate with `calc_connectivity_votes` (`:228-356`): it builds the
  bridging `missing_part` across the gap and requires **three collinearity
  angles** (base→bridge, bridge, bridge→candidate) all below
  `tolerance_angle * dist_f`, plus a total span below `max_tree_height`. The
  factor `dist_f = 1 - 1/sqrt(3 + max_distance - gap)` (`:269-272`) **tightens**
  the angle tolerance as the gap widens — a wide gap must be bridged nearly
  straight to be believed.
- Adopt the lowest-vote merge (`calc_vote`, `:360-367` — summed angles × chord²
  plus gap² × angle factor), `linemerge` the two paths with the bridge, absorb
  the slave stem, and drop stems fully inside `base.path.buffer(0.3)`
  (`_remove_duplicates_against_base`, `:81-100`). Repeat until no stem changes.
- A final pass drops stems with `length <= min_length` and removes duplicates
  (`:206-212`).

**Config knobs:** `max_distance=8 m`, `tolerance_angle=7°`, `max_tree_height=32 m`.

---

## Stage 5 — `rebuild_endnodes_from_stems`

**File:** `utils/Vectorization.py:405-417`

Collects each stem's `start`/`stop` coordinates into a list and returns it.
**In the standalone driver the return value is discarded and nothing is mutated**
(`standalone/WINMOL_Analyzer.py:74`), so this stage is effectively a timed no-op
today. See `docs/CODE_REVIEW.md` (V-8).

---

## Stage 6 — `quantify_stems`: measure diameter, length, volume

**File:** `utils/Quantification.py:52-86` · **In:** `List[Stem]`, `pred`, `profile`, `config`

For every node (measuring point) along a stem this estimates a trunk diameter,
then integrates volume between consecutive nodes.

1. **Diameter** (`get_diameters`, `:89-161`) — two methods, selected by
   `config.diameter_method`:
   - **`contour`** (default): polygonize the mask with
     `rasterio.features.shapes` into stem polygons (`:129-139`); at each node drop
     a short measurement line of length `2 * diameter_vector_half_length_m`
     (default 1.0 → a **2 m** probe) along the local **normal** to the stem
     (`_local_normal`, `:224-233`); intersect it with the polygons and take the
     chord passing *through* the node as the diameter (`calc_d`, `:303-317`).
   - **`edt`**: a Euclidean distance transform of the mask in metres
     (`distance_transform_edt(sampling=(py,px))`, `:259-262`); the value at the
     node pixel is the trunk radius, so `diameter = 2 * radius`.
2. **Clean** (`clean_diameter`, `:185-221`): IQR (1.5×) outlier fences over the
   per-stem diameter list; flagged interior nodes are replaced by
   distance-weighted linear interpolation of their neighbours, ends by their
   nearest neighbour.
3. **Length & volume** (`quantify_stem` → `calc_l_v`, `:164-176, 320-325`): for
   each consecutive node pair, `length` is the Euclidean distance and `volume`
   is the **truncated-cone (frustum)** integral
   `⅓·π·(r₁² + r₁r₂ + r₂²)·L`. Summing the per-segment lists gives the stem's
   total length and volume (`Stem.length`/`Stem.volume`, `Stem.py:34-51`).

**Config knobs:** `diameter_method="contour"`, `diameter_vector_half_length_m=1.0`,
`measuring_point_spacing_m=0.5`, `stem_binary_threshold=0.5`.

---

## Tiled runtime & the merge seam

On large orthomosaics the stem raster is cut into tiles with a halo overlap
(`utils/Tiling.py`); empty tiles are skipped, and each tile runs the six stages
above independently on its **full halo raster** (`VectorTilePipeline.py`). Because
neighbouring tiles both reconstruct the stems in their shared halo,
`IO.merge_and_filter_tiled_results` (`IO.py:1466`) filters each tile to roughly
its inner window and re-runs `connect_stems` on the pooled edge stems to reconcile
seams. Two correctness risks in that stitching (outer-perimeter stems dropped;
interior-seam double counting) are documented in `docs/CODE_REVIEW.md` (M-1, M-2).

---

## Parameter reference (`classes/Config.py`)

| Attribute | Default | Used in |
|-----------|---------|---------|
| `min_length` | 2.0 m | segment length gate (Stage 1, 4) |
| `max_tree_height` | 32 m | padding, connect span limit |
| `max_distance` | 8 m | connect endpoint search radius |
| `tolerance_angle` | 7° | connect collinearity gate |
| `measuring_point_spacing_m` | 0.5 m | node resampling / measuring |
| `diameter_method` | `contour` | diameter estimation |
| `diameter_vector_half_length_m` | 1.0 m | contour probe half-length |
| `stem_binary_threshold` | 0.5 | mask binarization |
