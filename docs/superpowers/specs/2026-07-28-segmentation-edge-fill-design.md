# Nodata-aware edge fill before segmentation

**Date:** 2026-07-28
**Branch:** `feat/segmentation-edge-fill` (off the current rc tip, not `main` —
`main` is 166 commits behind and lacks the cubic-resample and per-side
edge-buffer work this fix sits on top of). Experimental; **not** part of any
release candidate until validated in isolation.

## Problem

On the barnekow orthomosaic the U-Net produces **spurious "stem" activations
along the top edge** — false detections where the real canopy meets the
transparent border of the flight footprint.

The report in `docs/BUGS.md:48-50` also mentions that "in a previous version
quite a huge area was left out when segmenting, which removed a lot of trees."
That is a **separate, already-fixed** symptom (see Scope) and is **out of scope**
here.

## Why it happens (mechanism)

The barnekow ortho is RGBA (`9610 x 8662`, `uint8`), **`nodata: None`** — its
validity lives in a 4th **alpha** band, and the transparent regions carry
`RGB = 0` (black). The top rows are ~99% transparent, tapering along a diagonal;
only ~63% of the bounding box is real data.

The prediction path feeds that black region straight into the CNN:

- `utils/Prediction.py` `TileBatchProducer.run` reads each 512-px model tile with
  `src.read(..., boundless=True, fill_value=0)` (lines ~580-608). Every
  out-of-bounds pixel becomes `0` (black), and the interior alpha-transparent
  pixels are already `0` in the source. Either way the tile handed to the model
  contains a **hard black cliff** next to real canopy.
- The validity mask is computed (lines ~610-624:
  `gdal_mask & pixel_mask`, with a `pixel != 0` fallback when `gdal_mask` is
  all-valid) but is applied **only after inference** (`_binarize_prediction_core`
  = `prediction & mask`). So the U-Net still *sees* the cliff, and its receptive
  field bleeds spurious activations onto the **valid** pixels just inside the
  boundary. Those pixels are not masked away, so the artifacts survive into the
  stem map.

Consequence for the "add a border" instinct: adding **more constant black**
would not help — the black cliff *is* the problem. The cliff has to be replaced
with plausible content **before** inference.

## Scope

- **In scope:** the top-edge / boundary artifacts caused by the hard invalid-pixel
  seam reaching the CNN before masking.
- **Out of scope (already fixed):** the "huge area left out." Root cause was
  `utils/IO.py::_raster_filter_geom` doing `box(*b).buffer(-eb)`, shrinking every
  tile inward — including the ortho's true outer boundary — so real boundary
  stems were dropped in the tile-merge dedup. Fixed by per-side buffering
  (`utils/IO.py:972-980`, only interior seams shrink), paired with the
  bilinear->cubic resample fix (`Prediction.py:592`). This is the
  `WINMOL_fig6_barnekow_boundary_diff.png` result: new+cubic recovers 50
  boundary stems legacy silently dropped. No prediction-side tile dropping
  exists.

## Approach

Replace invalid pixels with the **nearest valid pixel** before inference, so the
model sees canopy smoothly continued past the boundary instead of a black cliff.
The output is masked exactly as today, so filled regions still produce no stems.

Alternatives considered and rejected:

- **Reflect-pad the raster rectangle edge only.** Minimal, but barnekow's top is
  an *interior* diagonal transparent region (black RGB inside the rectangle),
  which an edge-only reflect never touches. Insufficient for the actual scene.
- **Mask erosion / guard-band** (erode the valid mask inward after inference to
  drop near-boundary predictions). One line, guaranteed to kill artifacts — but
  it also discards the **real** near-boundary stems the cubic fix recovered,
  directly regressing that gain. Rejected as the primary fix; could return later
  as an off-by-default fallback knob if ever needed.

Nearest-valid replicate (edge-clamp) is chosen over reflect/inpaint because it
handles arbitrary polygon boundaries uniformly, is a single cheap library call,
and — being masked out of the output anyway — needs only to remove the cliff,
not to reconstruct true imagery.

## Component

One pure helper (proposed home: `utils/Prediction.py`, imported by the
multi-GPU producer too):

```python
def _fill_invalid_with_nearest(tile, valid_mask):
    """Replace invalid pixels with their nearest valid neighbour.

    tile:       (H, W, C) uint8 image as read from the source.
    valid_mask: (H, W) bool — True where the pixel is real data.
    Returns the tile with invalid pixels replaced; valid pixels untouched.
    """
    if valid_mask.all():          # common interior tile — no seam, no work
        return tile
    if not valid_mask.any():       # all invalid — masked out anyway
        return tile
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(
        ~valid_mask, return_distances=False, return_indices=True)
    return tile[tuple(idx)]
```

- `distance_transform_edt(~valid_mask, return_indices=True)` yields, for every
  pixel, the indices of the nearest valid pixel (nearest zero of `~valid_mask`).
  Indexing `tile` with those indices replicates the nearest valid value into
  invalid positions and is the identity on valid pixels.
- `scipy` is already a dependency (via scikit-image / the geo stack).
- Cost: one EDT on a 512² boolean — sub-millisecond, negligible against ~8 s of
  inference per tile.
- Pure function of `(tile, valid_mask)` — no I/O, trivially unit-testable.

## Data flow / integration

In `TileBatchProducer.run` (`utils/Prediction.py`), immediately after
`valid_mask` is computed (~line 624) and before the tile is queued (~line 627):

```python
if getattr(config, "fill_invalid_before_prediction", True):
    tile = _fill_invalid_with_nearest(tile, valid_mask)
batch_items.append((job, tile, valid_mask))
```

`valid_mask` is emitted **unchanged**, so the post-inference mask
(`_binarize_prediction_core = pred & mask`) is untouched and filled regions still
produce background.

The multi-GPU producer in `utils/PredictWorkers.py` shares the same read
semantics; it calls the same helper at the same point so the two paths cannot
diverge.

## Config

Add to `classes/Config.py`:

```python
fill_invalid_before_prediction = True   # nodata-aware edge fill (see spec)
```

Gating it behind a flag lets us A/B on/off via `WINMOL_CONFIG_OVERRIDES_JSON`
for the validation below, and gives a kill-switch if any scene ever regresses.
Default `True` on this branch.

## Error handling

The helper has fast paths for the all-valid and all-invalid cases and otherwise
performs a single, total array operation. It does not raise on ordinary inputs.
The producer already isolates and reports exceptions from its worker thread, but
the fill is written not to depend on that.

## Testing / validation

1. **Unit (hermetic, no model, runs in CI).** Synthetic tile with a
   black + invalid quadrant: assert invalid pixels take nearest-valid values,
   valid pixels are unchanged, and the all-valid / all-invalid fast paths return
   the input unchanged.
2. **Golden-inference regression.** The existing inference golden test must stay
   green. The fill is a no-op on all-valid fixture tiles, so the golden stem map
   should not move; if the fixture carries any nodata, the existing >= 0.999
   agreement / 0.5% foreground-drift tolerance absorbs it. Confirm which by
   inspecting the fixture.
3. **Behavioral evidence (acceptance).** On a cropped window at barnekow's top
   diagonal boundary, run the model **fill-off vs fill-on** and produce a
   before/after panel. Expect near-boundary spurious foreground to collapse to
   ~zero while interior foreground stays at parity. Figure saved for the write-up.

## Acceptance criteria

- Spurious stems in a near-boundary band drop to ~zero on barnekow.
- Interior stem count stays at parity with rc11 (no loss of real boundary stems).
- No golden-inference regression.

## Risks / open questions

- Replicated content is not true imagery; it is masked out of the output, and it
  only needs to remove the cliff. Risk is low but the behavioral panel is what
  confirms it.
- Receptive-field bleed can extend tens of pixels inside the boundary; nearest
  fill addresses the *input* seam, which is the cause, so this is expected to be
  sufficient. The panel quantifies any residual.
- Whether `fill_invalid_before_prediction` should default `True` or `False` once
  merged is deferred to after the validation; on this branch it is `True`.
