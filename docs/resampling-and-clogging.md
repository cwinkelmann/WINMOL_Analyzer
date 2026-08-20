# Tile resampling and the throughput collapse on large orthomosaics

Why prediction throughput decays and then collapses on large orthos
(issue #43), what each of the three resampling implementations did to it,
and what the options are.

All figures below were measured on one machine — 12 cores, 46.75 GB RAM,
RTX 4080 SUPER (16 GB), local NVMe — on three orthos:

| ortho | pixels | size | compression | overviews | pred tiles |
|---|---|---|---|---|---|
| Barnekow | — | 318 MB | — | — | 182 |
| Tegel R13 `result_Res1.3_COG.tif` | 392558 × 335327 | 19.7 GB | JPEG | yes | 99231 |
| Tegel R12 `result_Res1.2_webp.tif` | 338566 × 333907 | 12.3 GB | WEBP, 4-band | yes | 75072 |

The root-cause identification of GDAL's **global** block cache is Stefan's;
this document adds measurements around it.

---

## 1. The clogging is not new — 0.5 has it too, hidden

The reported symptom is throughput that decays continuously and then
falls off a cliff. It is easy to read this as a regression, because 0.5
does not visibly collapse. It does clog; the collapse is masked.

Measured on Tegel R12 with v0.5.0:

```
tile  6530   read 0.091s   queue 100% full
tile 19505   read 0.184s   queue  62% full
tile 75072   read 0.268s   queue   0% full     553 tiles/min
```

`read` triples over the run. The reason it never becomes a cliff is that
0.5's consumer is slow enough to hide it: `prep 0.079s` + `infer 0.018s`
per tile caps the pipeline near 550 tiles/min, and three producers can
stay ahead of that even while their reads get three times more expensive.

v0.5.0 R12, full run: **prediction 136.5 min of 167.4 min (81.6%)**.

So the ordering matters: **the reimplementation did not introduce the
clogging, it removed the slowness that was concealing it.** Once the
consumer got fast, the producers became the binding constraint and the
pre-existing read cost surfaced as a collapse.

### What actually degrades

Not inference. Across every run and every version, `infer` is flat
(0.011–0.018 s/tile). `prep` and `write` are flat. Only `read` moves, and
the producer queue drains from 100% to 0% as it does.

The mechanism is GDAL's block cache: one **global** cache, default
**5% of RAM**, shared by every open dataset. Three producer threads each
hold their own handle onto the same file and compete for it. GDAL
documents multi-access reading as defeating its internal optimisation.
Once the working set exceeds the cache, reads stop being served from
memory and the producers fall behind — permanently, because the queue
never refills.

This is why the collapse has a **position** (a data-volume threshold,
consistently near tile ~9500 on R13) rather than a time or a leak. Two
hypotheses were tested and rejected: RSS plateaus at 4471 MB exactly at
the cliff (no leak), and reducing to one producer cliffs *harder*
(2897 → 1000 tiles/min), not softer, so it is not GIL contention.

---

## 2. Where the resampling runs, and why it kept moving

The U-Net has a fixed input GSD, so every tile must be rescaled to the
model grid. Three implementations, three different answers.

### v0.5.0 — GPU, TensorFlow

`utils/Prediction.py:135`:

```python
tile_tensor = tf.image.resize(tile_tensor, size=[img_height, img_width],
                              method='bicubic', antialias=False)
```

Producers read tiles at **native resolution**; the consumer converts to
float32, ships them to the GPU and resizes there. Deliberate: it moves
computation off the CPU and lets autotune keep a steady stream.

Cost: the bus carries **full-resolution float32** tiles. Measured
`prep 0.079 s/tile`, which is dominated by that transfer, not by the
resize itself.

### The first reimplementation — skimage on the consumer

Dropping TensorFlow removed `tf.image.resize`, and it was replaced with
`skimage.transform.resize` in `_resize_batch`, still on the consumer
thread.

**This was the wrong solution, for two independent reasons.**

*It serialises.* The resize runs on the single consumer thread and holds
the GIL, so it cannot overlap with anything. Measured end-to-end on
Barnekow: **427 s versus ~60 s** for the GDAL path — 7×. The problem is
not that skimage is slow in absolute terms; it is that the work sits on
the one thread that must also drive inference, where 0.5 had it on the
GPU and the reimplementation later had it spread across producers.

*It changes the numbers.* skimage's `order=3` is not `tf.image.resize`'s
bicubic. On Barnekow it yields **451 stems** where the GDAL paths yield
455 and the TF-equivalent yields 457. Small, but it is an unreviewed
change to model input made as a side effect of removing a dependency.

A third variant was tested — the same skimage call moved into the
producer threads (`native_producer`, verified **bit-identical**, max
|Δ| 0.0) — which fixes the serialisation without fixing the numerics.

### The current reimplementation — GDAL inside the read

The resize moved into the producers' `src.read(out_shape=..., cubic)`, so
GDAL does it in C, in parallel, during the read. `prep` collapses from
0.079 s to 0.003 s.

This is the change that fixes the clogging, but **only because of a
side effect that is easy to miss**: when `out_shape` is smaller than the
window, GDAL can serve the read from an **overview** — roughly a quarter
of the bytes. Less data through a fixed-size global cache is what stops
the cache filling.

---

## 3. The `boundless` trap

The producer passed `boundless=True` on every read. It is needed only for
windows that run past the raster edge, where it keeps the returned shape
correct — but it was applied to every interior tile too.

`boundless` routes the read through a VRT wrapper that **cannot use
overviews**, so it reads at full resolution and downsamples in RAM. Same
result shape, ~4× the bytes.

Measured per interior tile on R13:

| read strategy | time |
|---|---|
| `boundless` — full-res via VRT (production) | 40.3 ms |
| full-res, no VRT (`OVERVIEW_LEVEL=NONE`) | 13.9 ms |
| plain path — GDAL serves an overview | **7.5 ms** |

So the pipeline was pushing 4× the necessary bytes through a cache fixed
at 5% of RAM. End-to-end on R13:

```
boundless   3843 → 3658 → 3189 → 2583 → 2251 → 2001 tiles/min   collapses ~9500
overview    4006 → 3998 → 3994 → 3982 ...                       FLAT to 15936
```

Full runs with the overview path, no collapse anywhere:

| | prediction | vector | total |
|---|---|---|---|
| R13 (99231 tiles) | 23.5 min @ **4229 tiles/min** | 65.0 min | 88.5 min |
| R12 (75072 tiles) | 18.7 min @ **4018 tiles/min** | 26.6 min | 45.3 min |

Against v0.5.0 on R12: **7.3× on prediction, 3.7× on the whole run.**

### Does the overview read change results?

The two reads differ in ~31% of pixels (mean |Δ| ≈ 2 DN), which looked
alarming. On stems it makes no difference at all:

```
read mode    stems  total len m  mean len m  total vol m3
boundless      455       4537.7        9.97        269.47
overview       455       4537.7        9.97        269.47
fullres        455       4537.7        9.97        269.47
native         451       4417.5        9.79        269.55
```

The pixel differences are sub-threshold and vanish at the 0.5
binarisation. Pixel-level comparison was the wrong proxy; stems are the
right one.

---

## 4. ONNX Resize — putting it back on the GPU without TensorFlow

Stefan's preference is to return resampling to the GPU (his prototype
uses CuPy). The same thing is achievable **inside the model graph**, with
no new dependency, because onnxruntime is already the plugin's runtime.

`utils/onnx_preprocess.py` prepends to the U-Net:

```
uint8 NHWC → Transpose → Cast(f32) → Div(255) → Resize(cubic) → U-Net
```

with `cubic_coeff_a=-0.5`, `coordinate_transformation_mode=half_pixel`,
`exclude_outside=1`. Those three settings are not cosmetic:

| configuration | max abs diff vs `tf.image.resize` bicubic |
|---|---|
| ONNX cubic, a=-0.5, half_pixel, **exclude_outside=1** | **4.2e-07** |
| same but exclude_outside=0 (the default) | 0.011 |
| PyTorch `F.interpolate(mode='bicubic')` | 0.039 |

**It reproduces v0.5.0's resampling to float32 rounding.** PyTorch cannot:
its bicubic hardcodes Keys `a=-0.75` and does not expose the coefficient.

End-to-end the wrapped graph's prediction differs from feeding
TF-resized input by **1 pixel in 524288** (fp16 rounding at the
threshold). Feeding uint8 also cuts PCIe traffic 4× versus 0.5, which
sent float32 — addressing the bus-load point directly.

### But it does not fix the clogging

Measured on R13:

```
tile  2156   2243 tiles/min   read 0.063s   queue 100% full
tile 14440   1803 tiles/min   read 0.090s   queue   0% full
tile 18456   1674 tiles/min   read 0.100s   queue   0% full
```

It clogs. The reason is structural: in-graph resizing requires
**native-resolution tiles**, which is the same byte volume as
`boundless`. It moves *where* resampling happens; it does not reduce
*how much data flows through GDAL's cache*.

This is the load-bearing finding of the document: **resampling location
and read volume are independent problems.** In-graph ONNX fixes
numerics, removes all CPU normalization and resizing, and restores GPU
resampling — and still clogs, because the I/O is unchanged.

---

## 5. Where that leaves the options

| option | matches 0.5 numerics | fixes clogging | new dependency |
|---|---|---|---|
| skimage on consumer | no | no | none |
| skimage in producer | no (bit-identical to skimage) | no | none |
| **GDAL + overview** | no | **yes** | none |
| **in-graph ONNX Resize** | **yes (4.2e-07)** | no | none |
| in-graph ONNX + `GDAL_CACHEMAX` ↑ | yes | dampens only | none |
| CuPy on GPU | needs checking vs TF | no (same I/O) | CuPy |
| GDAL reads at vector-tile size | n/a | likely | none, more RAM |

Two conclusions follow.

**They are not alternatives.** The accurate option and the fast option
solve different problems, and can be combined: in-graph ONNX Resize for
numerics and CPU cost, plus something that reduces read volume for the
clogging.

**Reducing read volume is the open problem.** The overview read solves it
but is its own resampling. Reading larger blocks (vector-tile sized)
would amortise the cache pressure at the cost of memory. Raising
`GDAL_CACHEMAX` from 5% only delays the collapse, as Stefan notes — this
has not yet been measured here and is the cheapest next experiment.

## Related fixes found along the way

Independent of resampling, both pre-existing (present in `origin/main`
and `v0.5.0`, from `b5e98eb`):

- **float64 promotion.** `(arr / 255.0).astype(np.float32)` on a uint8
  tile promotes to float64 and back — a 6.3 MB temporary per tile for a
  3.1 MB result. `np.divide(arr, np.float32(255.0), dtype=np.float32)` is
  **4.9× faster and bit-identical** across all 256 uint8 values. Moot if
  normalization moves into the graph.
- **The vector phase runs on 2 of 12 cores.**
  `_vector_worker_split` computes `min(max_vector_tile_workers,
  cpu_workers // 4, tiles)`; the `// 4` reserves cores for inner workers
  that the same branch then pins to 1. The vector phase is 73% of an R13
  run (65 of 88.5 min) and 88% on Barnekow, so this is now the dominant
  cost.

## Method and caveats

- Every figure is from this machine; absolute times will differ elsewhere.
- v0.5.0 ran the `.hdf5` Keras model, the others the fp16 ONNX conversion.
  Stem counts are therefore **not** comparable across that boundary
  (v0.5.0's 411 on Barnekow vs 455–457) — only within it.
- The Barnekow stem comparison is 3 vector tiles, so ±4 stems is ~1%.
  Conclusions drawn from it should be reconfirmed at Tegel scale.
- The R13 in-graph run was still in progress at 18.6% when these numbers
  were taken; it had already clogged.
- `GDAL_CACHEMAX` has **not** been tested. It is the most direct check of
  the root cause and should be run next.
