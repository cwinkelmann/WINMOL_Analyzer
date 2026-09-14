# Runtime of this release, measured

Numbers a release note can quote. Every row was produced on one machine
— ThinkPad T14, i7-1255U (12 threads), RTX 4080 SUPER 16 GB, local NVMe,
46.75 GB RAM — on the same two orthomosaics, with the same model
(`Spruce_Deadwood`), and each figure comes from a complete end-to-end
run whose output was compared against the previous version's.

| ortho | pixels | file | compression | prediction tiles |
|---|---|---|---|---|
| Tegel R13 `result_Res1.3_COG.tif` | 392558 × 335327 | 19.7 GB | JPEG, overviews | 99231 |
| Tegel R12 `result_Res1.2_webp.tif` | 338566 × 333907 | 12.3 GB | WEBP 4-band, overviews | 75072 |

## Against the released v0.6.1.1

Both arms in the same container image, so only the code under test
differs. `main` here is the released `ghcr.io/cwinkelmann/winmol-analyzer-gpu:v0.6.1.1`
entrypoint; `this release` is the same image with this branch mounted over it.

### Tegel R13

| stage | v0.6.1.1 | this release | factor |
|---|---|---|---|
| prediction | 2187 s | 1216 s | 1.80× |
| vector phase (incl. tile split) | ~1790 s | 295 s | 6.07× |
| **total** | **4008 s** (67 min) | **1555 s** (26 min) | **2.58×** |

### Tegel R12

| stage | v0.6.1.1 | this release | factor |
|---|---|---|---|
| prediction | 1964 s | 1095 s | 1.79× |
| vector phase (incl. tile split) | ~1380 s | 140 s | 9.9× |
| **total** | **3340 s** (56 min) | **1249 s** (21 min) | **2.67×** |

Per vector tile, averaged over the run: R13 21.0 s → 2.6 s, R12 27.9 s →
1.8 s. The vector phase was 45 % of an R13 run and is now 19 %;
prediction is the majority of a run again.

**The v0.6.1.1 R12 arm had to be given a 32 GB container.** Under a
16 GB cap the kernel OOM-killed one of its fork-pool workers (6.2 GB
anonymous RSS, cgroup constraint) at 96 % of the vector phase, and the
pool then waited on a result that would never arrive — the container sat
at 0.1 % CPU until it was stopped. This release completes the same file
under a 16 GB cap with a 6.5 GiB peak. Peak container RSS: R13 8.2 →
7.1 GiB, R12 > 16 → 6.5 GiB.

## Results are unchanged

Not "comparable" — identical. Compared against the v0.6.1.1 arm's own
output:

| | R13 | R12 |
|---|---|---|
| stems / vectors | 12,718 / 122,896 (both) | 3,190 / 33,000 (both) |
| merged GPKG | identical row for row, all three layers | identical row for row |
| stem-map raster | 25.2 Gpx, 0 px differing | 19.1 Gpx, 0 px differing |

"Row for row" means geometry WKB plus every attribute, in file order —
not merely the same set of features.

## Outside the container

The QGIS plugin spawns its managed venv's Python directly, not the
image. Driving that exact command (Python 3.11, onnxruntime-gpu 1.26,
rasterio 1.4.4 / GDAL 3.10.3 wheels) gives the same picture, and the
output still matches the container's byte for byte — across two
different onnxruntime builds and two different GDAL builds:

| | R13 | R12 |
|---|---|---|
| prediction | 1232 s | 1150 s |
| vector phase | 295 s | 142 s |
| **total** | **1571 s** | **1308 s** |

## Where the vector-phase time went

Average per tile at 11 workers, from the run log's own stage summary:

| stage | before | now |
|---|---|---|
| skeletonisation | 2.9 s | 0.8 s |
| quantification | 1.7 s | 1.15 s |
| connect stems | 0.8 s | 0.33 s |
| write | 0.4 s | 0.35 s |
| serial tile split (once per run) | 89 s | 0 s (in the workers) |

The skeleton stage dominated because it scanned the
`max_tree_height`-padded tile — 7102² for a 4916 px tile, 50 M pixels —
about fifteen times per tile, for a foreground under 1 %. With eleven
workers those scans contended for memory bandwidth: the per-tile average
in a full run was 2.5× the same tile measured alone. See the commits on
`utils/Skeletonization.py`.

## How to reproduce

```
docker run --rm --gpus all --memory=16g --memory-swap=16g \
  -e WINMOL_PROCESS_TYPE=Nodes -e GDAL_CACHEMAX=2048 \
  -e WINMOL_CONFIG_OVERRIDES_JSON='{"prediction_batch_override": 1}' \
  -v <ortho.tif>:/data/input/<ortho.tif>:ro \
  -v <outdir>:/data/output -v <modeldir>:/data/models \
  ghcr.io/cwinkelmann/winmol-analyzer-gpu:<tag> Spruce_Deadwood
```

The run prints `Elapsed time` per phase and a `VECTOR SUMMARY` block
with the per-stage averages quoted above. Worker counts come from the
execution plan; nothing above pins them beyond the batch size.
