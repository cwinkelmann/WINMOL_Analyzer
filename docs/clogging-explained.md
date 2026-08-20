# Where the clogging comes from, and would more workers fix it?

Short version:

- The producers read **~4x more bytes than they need to**, and those bytes
  flow through **one GDAL cache shared by all of them**, fixed at 5% of RAM.
- While the working set fits the cache, reads are cheap. When it stops
  fitting, every read goes to disk, the queue drains, and throughput
  collapses. That is why the collapse happens at a **tile position**
  (~9500 on R13), not after a certain time.
- **More workers would probably hide it, not fix it** — but the run that
  proves that has not finished yet, so treat that as unconfirmed.

---

## 1. What is actually slow

Only one number moves. Across every version and every run, `prep`,
`infer` and `write` stay flat; `read` climbs and the producer queue drains
from 100% full to 0%.

R13, current code with the default `boundless` read:

```
tile  7632   3658 tiles/min   read 0.048s   queue 100% full
tile  9576   3189 tiles/min   read 0.051s   queue   0% full   <- flips here
tile 12120   2001 tiles/min   read 0.063s   queue   0% full
```

The GPU is fine. The consumer is fine. The producers stop keeping up.

## 2. Why the reads are expensive

The producer asks GDAL for a 1170x1170 window resampled to 512x512, and
passes `boundless=True`.

`boundless` exists for one reason: windows at the raster edge run past
the image, and it keeps the returned array the right shape. But it was
applied to **every** tile, and it routes the read through a VRT wrapper
that **cannot use the file's overviews**. So instead of GDAL fetching a
decimated overview level, it reads every source pixel at full resolution
and shrinks it in RAM.

Measured per interior tile on R13:

| how the tile is read | time | bytes read |
|---|---|---|
| `boundless` — full-res through a VRT (the default) | 40.3 ms | ~4x |
| full-res, no VRT (`OVERVIEW_LEVEL=NONE`) | 13.9 ms | ~4x |
| plain path — GDAL serves an **overview** | **7.5 ms** | **1x** |

## 3. Why 4x the bytes turns into a cliff rather than a slope

GDAL keeps decoded blocks in a **global** cache — one cache for the whole
process, default **5% of RAM**, shared by every open dataset. The three
producer threads each hold their own handle on the same file and compete
for it. (GDAL's own documentation notes that multi-access reading defeats
its internal optimisation. This diagnosis is Stefan's.)

That gives a threshold, not a gradient:

- Early on, the tiles being read still fit in the cache. Reads are served
  from memory and are cheap.
- Pushing 4x the bytes fills that fixed cache 4x sooner.
- Once the working set no longer fits, reads fall to disk speed, all
  three producers slow at once, the queue empties and never refills.

This explains the two things that confused me for hours: the collapse
happens at a **consistent tile number** rather than a consistent time,
and it does not recover.

Two other explanations were tested and are **wrong**:

- *A memory leak* — RSS plateaus at 4471 MB exactly at the cliff.
- *GIL contention from the consumer* — cutting to **one** producer made
  it cliff **harder** (2897 -> 1000 tiles/min), not softer.

## 4. The experiment that proves it is the byte volume

This is the decisive one, because it changes everything *except* the
bytes.

The in-graph ONNX build moves normalisation and resampling into the model
graph, on the GPU. No skimage. No GDAL resampling. No float64. No
`out_shape`, no boundless VRT. The only thing it keeps is that GDAL
returns tiles at **native resolution** — the same byte volume as before.

R13, in-graph ONNX:

```
tile  2156   2243 tiles/min   read 0.063s   queue 100% full
tile 18456   1674 tiles/min   read 0.100s   queue   0% full
tile 99231   1214 tiles/min   read 0.137s   queue   0% full
total 118.4 min
```

It clogs. Meanwhile the overview read — which changes *only* the byte
volume — does not clog at all:

```
overview   4006 -> 3998 -> 3994 -> 3982 ...   FLAT across all 99231 tiles
           R13 prediction 23.5 min, total 88.5 min
```

Same conclusion on R12 (WEBP, 4-band): overview 4018 tiles/min flat;
in-graph 833 tiles/min and falling.

**Read volume is the cause. Where the resampling runs is irrelevant to
the clogging** — it matters for accuracy and CPU cost, which is a
separate problem.

## 5. So would more producer workers just fix it?

Partly, and it is worth being precise about "partly".

What is measured:

| producers | throughput after the cliff |
|---|---|
| 1 | 1000 tiles/min |
| 3 | 2001 tiles/min |
| 8 | **not yet measured** |

The post-cliff plateau scales at roughly 1000 tiles/min per producer, so
6-8 producers would land near 4000 — about what the overview fix
achieves. On this 12-core machine, more workers would very likely make
the symptom invisible.

**But the 8-producer run has not completed.** It was queued three times
and pre-empted each time by other experiments. Until it finishes, the
extrapolation above is an extrapolation, and today has produced two
extrapolations that were badly wrong (a 117 min vector-phase projection
that measured 65, and a 651x microbenchmark speedup that was 1.8x
*slower* end to end).

Three reasons it is masking rather than fixing, even if it works:

1. **It spends cores on redundant work.** The extra producers exist to
   decode 4x more JPEG data than the model needs. That is 6-8 cores busy
   producing pixels that are then thrown away in the downsample.
2. **It fails exactly where it is needed most.** The planner caps
   producers at 2 when `cpu_workers < 10`, so an 8-core laptop cannot get
   the extra threads at all. The overview fix helps every machine.
3. **It does not reduce cache pressure, it hides latency.** The same
   volume still flows through the same fixed global cache; more threads
   just keep more requests in flight. Whether that scales or starts to
   thrash is the open question the 8-producer run answers.

By contrast the overview read removes the cause: 4x fewer bytes, the
working set fits the cache again, and 3 producers are more than enough.
Measured flat across 99231 tiles on R13 and 75072 on R12, and — checked
on stems, not pixels — **identical output**: 455 stems, 4537.7 m,
269.47 m3, the same to every digit as the `boundless` path.

## 6. What is still open

- **The 8-producer run.** Re-queued. It converts section 5 from
  reasoning into measurement.
- **`GDAL_CACHEMAX`.** Never tested. Raising it from 5% should *delay*
  the collapse without removing it. That is a direct check of the
  mechanism in section 3 and is cheap.
- **Producer cap.** `cpu_workers // 3` silently reduces a configured 6 to
  3. The planner now prints a warning when it does this; the cap itself
  is unchanged pending the 8-producer result.
