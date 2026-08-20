# The tile-resampling problem — the kernel question, and which is right

> **RETRACTED NUMBERS (2026-08-11). Read `docs/resize-mechanics.md` first.**
> The absolute stem counts in this file come from the "era" A/B harness
> (12,714 "v0.5" / 12,722 in-graph / 15,556 fullres / 16,548 overview, and
> "455 on every variant" on Barnekow). They did not survive
> re-measurement and **must not be cited as baselines**. Root cause,
> closed 2026-08-11: the era harness ran the **Spruce_Deadwood** model
> (2024-12-19) while labelling it Spruce — that pipeline reproduces
> 12,731 on R13, within 0.13 % of the era's 12,714.
>
> What replaced them: genuine v0.5.0 on R13 = **15,950 stems**, this
> branch's `graph` default = **15,940** (−0.06 %), volume to 0.002 %.
> The kernel effect is **scale-dependent and real** but smaller than
> stated here: +5.6 % stems at 1.42×, **+20 % stems / +26 % volume at
> 2.29×** — not "30 %", and not "0 % on Barnekow".
>
> **What survives:** the *structure* of the argument — that the effect
> decomposes into a kernel term (GDAL anti-aliases with scale, TF's
> 4-tap Keys kernel does not) and a smaller source term (overview =
> two-stage resample), and that validating a resampling change on a 1.4×
> ortho and shipping it for 2.3× data is what let it through. That
> reasoning is why `graph` is the default and `overview` is flag-gated.

**Status: the parity question is CLOSED (`graph` ≡ v0.5 to 0.06 %). The
open question is narrower — whether the anti-aliased fast path is *more*
accurate than v0.5 semantics, which needs the training-time resize recipe
or field ground truth, not another benchmark.**

## The problem in one table

Every row below uses the **same model** on the **same orthomosaic**
(Tegel R13, 99231 tiles). Only the way each tile is resampled to the
model's 512×512 grid differs.

| tile resampling | stems | throughput |
|---|---|---|
| v0.5.0 — `tf.image.resize` bicubic, full-res source | **12714** | 553/min |
| in-graph ONNX Resize (≡ tf bicubic to 4.2e-07), full-res | **12722** | ~1800/min |
| GDAL cubic, **full-res** source (`fullres`) | **15556** | 3662/min |
| GDAL cubic, **overview** source (`overview`, current default) | **16548** | 4229/min |

The current default finds substantially more stems than v0.5.0 on this
ortho (era figure: 30 % — **retracted**; re-measured at R13 scale: +20 %
stems / +26 % volume).
Nobody noticed, because the change was validated on Barnekow, where all
GDAL variants agree exactly (455 / 455 / 455) and the effect is invisible.

## Why the difference exists

Two independent causes, and the measurements separate them:

- **Kernel: ~22%** (15556 → 12722). GDAL's cubic *anti-aliases* when
  downsampling — it widens the kernel with the scale factor.
  `tf.image.resize(..., antialias=False)` does not: it uses a fixed 4-tap
  Keys kernel regardless of scale. At R13's 2.3× downsample that is a
  large difference; GDAL's output is smoother, tf's aliases more.
- **Source: ~6%** (16548 → 15556). Reading from an overview resamples
  from already-decimated pixels — a two-stage resample, where the first
  stage is whatever `gdaladdo` used. Reading full resolution is one stage.

**Scale-dependence is what hid it.** The harder the downsample, the more
the kernel matters:

| ortho | downsample | disagreement |
|---|---|---|
| Barnekow | 727 → 504 px (1.4×) | **0%** (455 vs 455) |
| Tegel R13 | 1172 → 504 px (2.3×) | **30%** |

Validating a resampling change on a 1.4× ortho and shipping it for 2.3×
data is the mistake that let this through.

## What the difference looks like

At the raster level, sampling 40 full-resolution 4096² windows:

```
stem px, overview + GDAL cubic : 560,615  (0.1453%)
stem px, native + tf-equivalent: 415,599  (0.1077%)
foreground ratio               : 1.349
IoU of the two masks           : 0.6627
only in A (overview+GDAL)      : 29.22% of union
only in B (native+tf)          : 4.52% of union
```

So it is **not** a boundary effect on shared stems. It is asymmetric: A
finds substantially more stem pixels, and 29% of the union is A-only.

At the vector level, **2907 of A's stems (17.6%) have no B stem within
2 m** — genuinely separate detections, not the same stems drawn
differently.

See `docs/img/stemmap_diff.png` (masks) and
`docs/img/stem_visual_check.png` (disputed detections over the imagery).

## Which one is correct?

**This cannot be settled from these runs.** There is no ground truth in
either; they are two different answers. Two readings, both plausible:

- **The extra stems are real.** GDAL's anti-aliased input is cleaner, and
  the model finds fallen stems that aliasing hid from v0.5.0. Then the
  current default is *better*, and "matches v0.5.0" is the wrong target.
- **The extra stems are false positives.** The model was trained on
  non-antialiased tiles, so a smoother input is a domain shift, and the
  extra detections are artifacts on shadows, tracks or bare ground.

`docs/img/stem_visual_check.png` puts both stem sets on the orthomosaic
at 60 m across, centred on the disputed detections, so the question can be
answered by looking. **Field-surveyed stems for any part of Tegel would
settle it properly.**

Note the assumption underneath all of this: that v0.5.0's *inference*
preprocessing is what the model was *trained* with. That has not been
verified against the training pipeline, and if training used something
else again, neither option here is the reference.

## Options, with costs

| option | stems | R13 end-to-end | matches v0.5.0 |
|---|---|---|---|
| `overview` (current default) | 16548 | **62.8 min** | no |
| `fullres` | 15556 | ~70 min | no |
| `onnx_gpu` (in-graph tf-equivalent) | ~12722 | ~95 min | **yes** |

`onnx_gpu` is the only one that reproduces v0.5.0, at roughly 1.5× the
wall-clock. That is a real cost but not a prohibitive one, and it is the
safe default if the extra detections turn out to be artifacts.

## One fix that is independent of the decision

The native read path passed `boundless=True` on **every** tile — the same
trap already fixed on the `out_shape` path. Restricting it to windows that
genuinely cross the raster edge is **3.2× cheaper (47.3 → 15.0 ms) for
pixel-identical output**, and it is what makes `onnx_gpu` viable at all
(its `read` went 0.144 → 0.037 s). Take this regardless of which
resampling wins.

## What NOT to conclude

- **Do not read the throughput cliff fix as depending on this.** The cliff
  (issue #43) is a separate problem with a separate fix — see
  `docs/clogging-explained.md`. It can be solved by giving GDAL's block
  cache room (`GDAL_CACHEMAX` ~20%), which changes no pixels at all. The
  overview read fixed the cliff *and* changed the data; only the first
  half was wanted.
- **Do not treat stem counts from before and after this change as
  comparable.** Any trend line spanning it will show a 30% step that is
  methodological, not ecological.
