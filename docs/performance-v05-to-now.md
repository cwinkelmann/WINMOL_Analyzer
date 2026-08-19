# Performance: v0.5.0 to now, and how the throughput cliff was fixed

Two questions this answers:

1. **How much faster is the current pipeline than v0.5.0, stage by stage —
   and are the results still comparable?**
2. **What was the throughput cliff on large orthomosaics (issue #43), and
   what actually fixed it?**


> **RETRACTION (2026-08-11) — read `docs/resize-mechanics.md` first.**
> Every absolute stem count in this document that is attributed to
> **v0.5.0** (12,714 / 12,722, "30 % because of resampling", "455 on
> every Barnekow variant") comes from the era A/B harness and is
> **retracted**: that harness ran the **Spruce_Deadwood** model while
> labelling it Spruce. Genuine v0.5.0 on R13 is **15,950 stems**, and
> this branch's default `graph` path reproduces it to **−0.06 %**
> (volume to 0.002 %). The *throughput* measurements below — the cliff,
> the stage-by-stage timings, GDAL_CACHEMAX, worker counts — are
> unaffected; only the cross-era accuracy comparisons are.

All measurements are from one machine — 12 cores, 46.75 GB RAM, RTX 4080
SUPER (16 GB), local NVMe — on three orthomosaics:

| ortho | pixels | size | compression | overviews | pred tiles |
|---|---|---|---|---|---|
| Barnekow | — | 318 MB | — | **none** | 182 |
| Tegel R13 `result_Res1.3_COG.tif` | 392558 × 335327 | 19.7 GB | JPEG | yes | 99231 |
| Tegel R12 `result_Res1.2_webp.tif` | 338566 × 333907 | 12.3 GB | WEBP, 4-band | yes | 75072 |

> A note on reading the logs: the `tiles/min` the pipeline prints is a
> **cumulative average** (total tiles ÷ total elapsed). It dilutes late
> degradation and makes a decaying run look steadier than it is. Every
> figure below marked *instantaneous* was recomputed between consecutive
> progress lines. Several wrong conclusions during this investigation came
> from reading the cumulative number as if it were current.

---

## 1. The cliff

### Symptom

On a large orthomosaic, prediction throughput decayed continuously and then
collapsed. Reported from the field on v0.6.1-rc11, reproduced here:

```
tile  7632   3658 tiles/min   read 0.048s   queue 100% full
tile  9576   3189 tiles/min   read 0.051s   queue   0% full   <- flips here
tile 12120   2001 tiles/min   read 0.063s   queue   0% full
```

Only `read` moves. `prep`, `infer` and `write` stay flat throughout, and
the producer queue drains from 100% full to 0% and never refills. The
collapse lands at a consistent **tile position** (~9500 on R13), not after
a consistent time.

### Cause

Two things compounding.

**Every tile was read at full resolution.** The producer asked GDAL for a
1170×1170 window resampled to 512×512, passing `boundless=True`.
`boundless` exists to keep the array shape correct for windows that run
past the raster edge — but it was applied to *every* tile, and it routes
the read through a VRT wrapper that **cannot use the file's overviews**.
So GDAL read every source pixel and shrank it in RAM instead of fetching a
decimated level. Measured per interior tile on R13:

| how the tile is read | time | bytes |
|---|---|---|
| `boundless` — full-res through a VRT (the old default) | 40.3 ms | ~4× |
| full-res, no VRT (`OVERVIEW_LEVEL=NONE`) | 13.9 ms | ~4× |
| plain path — GDAL serves an **overview** | **7.5 ms** | **1×** |

**Those bytes flow through one shared, fixed-size cache.** GDAL keeps
decoded blocks in a **global** block cache, default **5% of RAM**, shared
by every open dataset and contended by all three producer threads. (GDAL's
own documentation notes that multi-access reading defeats its internal
optimisation.) Four times the bytes fills a fixed cache four times sooner;
once the working set no longer fits, all producers hit disk at once.

That gives a threshold rather than a gradient — which is why the collapse
has a position, and why it never recovers.

Sweeping the cache with everything else held constant confirms it
directly:

```
GDAL_CACHEMAX  256 MB  ->  collapses immediately, never exceeds ~139 tiles/min
              1024 MB  ->  1812 -> 181 tiles/min    (10x decline)
              2400 MB  ->  3050 ->  55 tiles/min    (55x decline)  <- the 5% default
              9600 MB  ->  2675 -> 2675 tiles/min   NO decline, to tile 16204
```

### What it was NOT

Four hypotheses were tested and refuted. They are recorded because each is
plausible and each cost time:

| hypothesis | test | verdict |
|---|---|---|
| Inference got slower | `infer` is 0.011–0.018 s in every run and version | **no** |
| A memory leak | RSS plateaus at 4471 MB exactly at the cliff | **no** |
| The network mount (CIFS) | reproduces identically on local NVMe | **no** |
| GIL contention from the consumer | `producer_workers=1` cliffs *harder* (2897 → 1000) | **no** |

And one fix that seemed obvious but does not work:

**More producer threads do not rescue it.** Measured on R13, instantaneous:

```
3 producers   2643 -> 80 tiles/min   (cliff at ~9256)
8 producers   2516 -> 87 tiles/min   (cliff at ~9496)
```

Both collapse, at the same tile position, by the same order of magnitude.
Adding readers does not reduce the bytes, so the threshold lands where it
always did.

### The fix

**Use the overview.** `boundless` is now applied only where it is actually
needed — windows that genuinely cross the raster edge — and interior tiles
take the plain path, which lets GDAL serve a decimated level.

```
before (boundless)   3843 -> 3658 -> 3189 -> 2583 -> 2251 -> 2001   collapses ~9500
after  (overview)    4006 -> 3998 -> 3994 -> 3982 ...               FLAT to 15936
```

Flat across **all 99231 tiles** of R13 and all 75072 of R12 (WEBP, 4-band —
so the fix is not specific to JPEG COGs).

**Critically, the output does not change.** The two reads differ in ~31% of
*pixels* (mean |Δ| ≈ 2 DN), which looked alarming, but those differences
are sub-threshold and vanish at the 0.5 binarisation:

```
read mode    stems  total len m  mean len m  total vol m3
boundless      455       4537.7        9.97        269.47
overview       455       4537.7        9.97        269.47
fullres        455       4537.7        9.97        269.47
```

Identical to every digit. Pixel-level comparison was the wrong proxy;
stems are the right one.

### The prerequisite: overviews must exist

The fix works because GDAL can fetch a decimated level. **A file without
overviews has no such level**, so it falls back to full-resolution reads
and the cliff returns. Barnekow (a plain GeoTIFF) has none; both Tegel COGs
do.

The pipeline now warns at startup when an input over 2 GB lacks overviews.
Build them once per file:

```bash
gdaladdo -ro -r average your_ortho.tif 2 4 8 16 32 64 128
```

`-ro` writes a `.ovr` sidecar and leaves the original untouched.

### Two related fixes found alongside

Both pre-existing (identical in `origin/main` and `v0.5.0`, from `b5e98eb`):

- **float64 promotion.** `(arr / 255.0).astype(np.float32)` on a uint8 tile
  promotes to float64 and back — a 6.3 MB temporary per tile to produce a
  3.1 MB result. `np.divide(arr, np.float32(255.0), dtype=np.float32)` is
  **4.9× faster and bit-identical** across all 256 uint8 values.
- **The vector phase ran on 2 of 12 cores.** `_vector_worker_split`
  computed `min(max_vector_tile_workers, cpu_workers // 4, tiles)`; the
  `// 4` reserved cores for inner workers that the same branch then pinned
  to 1. Now sized from cores *and* free RAM (measured 1.38 GB private per
  worker, not the 2.7 GB RSS suggests — the rest is shared copy-on-write).

---

## 2. v0.5.0 vs now

### Full runs, native (no container)

| | prediction | vector | total |
|---|---|---|---|
| **Tegel R13**, v0.5.0 | 164.1 min | 64.7 min | **229.1 min** |
| Tegel R13, current, 2 vector workers | 23.5 min | 65.0 min | 88.5 min |
| Tegel R13, current, 11 vector workers | 24.3 min | 38.5 min | **62.8 min** |
| **Tegel R12**, v0.5.0 | 136.5 min | 30.8 min | **167.4 min** |
| Tegel R12, current | 18.7 min | 26.6 min | **45.3 min** |
| **Barnekow**, v0.5.0 | 31.4 s | 33.4 s | **68 s** |
| Barnekow, current | 3.1 s | 25.7 s | **29 s** |

R13: **3.6×** end to end. R12: **3.7×**.

Note the vector phase is only ~1.2× faster between versions — it is the
same code — so nearly all of the gain is in prediction, and the vector
phase is now the majority of a large run (61% of R13 even at 11 workers).

### Per-tile stages, containerised, same ortho (R12)

| | v0.5.0 | current | factor |
|---|---|---|---|
| read | 0.086 s | 0.011 s | **8×** |
| prep | 0.115 s | 0.004 s | **29×** |
| infer | 0.017 s | 0.016 s | 1.0× |
| write | ~0 | ~0 | — |
| throughput | ~440 tiles/min | ~2890 tiles/min | **6.6×** |
| host CPU used | ~100% (1 core) | ~537% | |

**`infer` is unchanged.** The model runtime — TensorFlow/Keras `.hdf5`
versus ONNX Runtime `.onnx` — is not where the time went. The gains are
entirely in getting data to the model:

- **`prep` (29×)** — v0.5.0 shipped full-resolution float32 tiles to the
  GPU for `tf.image.resize`. The current version resamples inside the GDAL
  read, in C, across the producer threads. This is the single largest term.
- **`read` (8×)** — the overview fix above.

The CPU figure is the clearest illustration: v0.5.0 pins at **one core of
twelve**, because its consumer thread is serialised on `prep` while holding
the GIL, so its three producers cannot get ahead. Its queue reads "100%
full" only because the consumer is the bottleneck.

### v0.5.0 clogs too — it is just hidden

v0.5.0 does **not** escape the cause; its slow consumer masks it. On R12:

```
tile  6530   read 0.091s   queue 100% full
tile 19505   read 0.184s   queue  62% full
tile 75072   read 0.268s   queue   0% full
```

`read` triples over the run. Instantaneous throughput goes 665 → 264
tiles/min, a 2.5× decline, most of it in the last stretch. It never becomes
a *cliff* because v0.5.0's ceiling (`prep` 0.115 s + `infer` 0.017 s ≈ 450
tiles/min) sits below the degraded read rate for most of the run.

So the reimplementation did not introduce the clogging — it removed the
slowness that was concealing it.

### Are the results comparable?

**Between read strategies: identical.** `boundless`, `overview` and
`fullres` all produce 455 stems / 4537.7 m / 269.47 m³ on Barnekow, to
every digit. Worker counts likewise: R13 gave **16548 stems and 160623
nodes** with 2 vector workers, with 11, and in the container.

**Between v0.5.0 and now: NOT comparable, and the cause is the resampling
operator, not the model.**

This is the most important caveat in this document. The *conclusion* held
up; the numbers under it did not. On Tegel R13 (**era table — retracted
2026-08-11, kept for the record of what was compared**):

| configuration | model | resampling | stems |
|---|---|---|---|
| v0.5.0 | `.hdf5` | `tf.image.resize` bicubic | **12714** |
| in-graph ONNX | `.onnx` fp16 | ONNX Resize ≡ tf bicubic (4.2e-07) | **12722** |
| overview read (default) | `.onnx` fp16 | GDAL cubic | **16548** |
| 11 vector workers | `.onnx` fp16 | GDAL cubic | **16548** |
| containerised | `.onnx` fp16 | GDAL cubic | **16548** |

Read that carefully. The two runs that share a **resampling method** agree
to **0.06%** despite using *different model artifacts*. The runs that share
a **model artifact** differ because of resampling. So the model conversion
(`.hdf5` → fp16 `.onnx`) is nearly irrelevant to stem count, and **the
resampling operator dominates it**.

Both halves of that claim were re-tested against genuine v0.5.0 on
2026-08-11 and **survived — with corrected magnitudes**. Model conversion
is indeed near-irrelevant (`graph` vs genuine v0.5.0: −0.06 % stems,
+0.002 % volume on R13; −0.13 % / −0.24 % on R12). The resampling operator
does dominate, but the kernel gap is **+20 % stems / +26 % volume at
2.29×**, not 30 %, and it is **scale-dependent** (+5.6 % at 1.42×) rather
than absent on Barnekow. The 12,714 row was the Spruce_Deadwood model
under a Spruce label, not a resampling result at all.

An earlier version of this document claimed the opposite. It was wrong: it
compared v0.5.0 against the current default and attributed the whole gap to
the model, without a run that isolated the two variables. The in-graph ONNX
run is that control, and it points the other way.

The same effect is visible but milder on Barnekow — 411 (v0.5.0) vs 454–457
(GDAL cubic), about +10% — which is consistent with resampling mattering
more the harder the downsample: R13 goes 1172 px → 512 (2.3×), Barnekow
727 px → 504 (1.4×).

**What this means in practice.** Within one resampling method, everything
is reproducible: read strategy, worker count and containerisation all give
byte-identical stem counts. Across methods they are not, and the current
default finds **~30% more stems on R13 than v0.5.0 did**. Whether that is
better or worse cannot be settled from these runs — there is no ground
truth here, only two different answers — so it needs validation against
reference data before the numbers from the two versions are treated as
interchangeable.

If exact v0.5.0 numerics are required, ONNX Resize with
`cubic_coeff_a=-0.5`, `coordinate_transformation_mode=half_pixel`,
`exclude_outside=1` reproduces `tf.image.resize` bicubic to **4.2e-07**
(`utils/onnx_preprocess.py`), and the 12722 above confirms it end to end.
It requires native-resolution reads, so it reintroduces the clogging
(R13: 118.4 min vs 88.5) — available as an option, not the default.

---

## 3. Reproducing these numbers

```bash
# throughput curves across a whole ortho, per configuration
python benchmark/bench_read_path.py --ortho <big.tif> --probe layout
python benchmark/bench_read_path.py --ortho <big.tif> --probe temporal --tiles 24000
python benchmark/bench_read_path.py --ortho <big.tif> --probe modes
```

Read strategies can be selected for benchmarking with
`WINMOL_BENCH_READ=overview|boundless|fullres|native|onnx_gpu`; `overview`
is the default and the only one measured flat across a full large ortho.
