# Benchmark: original pipeline vs the full change stack

`bench_orig_vs_changed.py` runs one orthomosaic through **both** versions of the
analyzer, headless (no QGIS), N times each, and reports the **median**.

| | A · ORIGINAL | B · CHANGED |
|---|---|---|
| Code | `origin/main` | the branch you run from |
| Model | `.hdf5` (Keras) | `.onnx` |
| Runtime | TensorFlow | onnxruntime |
| `PYTHONHASHSEED` | unset — as shipped | pinned by the guard |

Both sides run the **same weights**: `General.onnx` was converted from
`model_UNet_GenDS_512_2023-02-27_211141.hdf5` with 0.0000 % binary-mask
disagreement (`docs/onnx-conversion-parity.md`). Differences in *output* are
therefore attributable to the pipeline changes — the ortho-boundary edge fix,
the determinism guard, the GDAL-read resample — not to swapping models.

The **wall-time** figure is end-to-end, old stack vs new stack: it includes the
inference-engine change, which is what an end user actually experiences after
upgrading.

## What you need

1. **A checkout** — the script creates its own git worktrees, so nothing else.
2. **An orthomosaic.** Any GeoTIFF the analyzer accepts (< 3 cm GSD preferred).
3. **Both models.** `standalone/model*` is gitignored, so fetch them:

```bash
# ONNX (this repo's release)
gh release download models-onnx-v1 --repo cwinkelmann/WINMOL_Analyzer \
   --pattern 'General.onnx' --dir standalone/model_onnx

# HDF5 — the original Keras weights, from the WINMOL/Zenodo publication
#   model_UNet_GenDS_512_2023-02-27_211141.hdf5  -> standalone/model/
```

4. **Python environments** — see the CUDA section; on one machine you often
   need two.

## Running it

```bash
python benchmark/bench_orig_vs_changed.py \
  --ortho    /data/20220212_Barnekow_4.tiff \
  --orig-model standalone/model/model_UNet_GenDS_512_2023-02-27_211141.hdf5 \
  --new-model  standalone/model_onnx/General.onnx \
  --outdir   benchmark/out/barnekow \
  --repeats  3
```

Results stream to `<outdir>/results.json` after **every** run, so a crash or an
interrupted job still leaves usable data. Re-print a summary at any time
without recomputing:

```bash
python benchmark/bench_orig_vs_changed.py --report benchmark/out/barnekow
```

## On a CUDA machine

This is where the performance number is worth trusting. On Apple Silicon the
vector phase dominates (~73 % of a run) and inference is comparatively cheap, so
the *shape* of the profile differs from a datacentre GPU — not just its
magnitude.

```bash
python benchmark/bench_orig_vs_changed.py \
  --ortho /data/20220212_Barnekow_4.tiff \
  --orig-model .../model_UNet_GenDS_512_2023-02-27_211141.hdf5 \
  --new-model  .../General.onnx \
  --onnx-providers CUDAExecutionProvider \
  --orig-python /envs/tf-cuda/bin/python \
  --new-python  /envs/ort-gpu/bin/python \
  --outdir benchmark/out/barnekow-cuda --repeats 3
```

Three things that will otherwise cost you an afternoon:

- **`--onnx-providers CUDAExecutionProvider` is not optional.** onnxruntime
  falls back to CPU *silently* if the GPU provider fails to load, and you will
  read the result as "the GPU is slow" rather than "the GPU never ran". Confirm
  in the log which provider was actually selected.
- **Use two environments.** TensorFlow-CUDA and `onnxruntime-gpu` frequently
  disagree about CUDA/cuDNN versions. `--orig-python` / `--new-python` exist so
  you do not have to reconcile them.
- **`TF_USE_LEGACY_KERAS=1` is set for you.** The shipped `.hdf5` are Keras 2
  artifacts and will not load under the Keras 3 bundled with TF ≥ 2.16.

## Reading the output

```
config        wall_s   stems  spread   volume_m3  deterministic
---------------------------------------------------------------
changed        114.2     484       0     439.228  YES
original       793.0     449       5     400.661  NO (3 distinct)
```

- **wall_s** — median wall time. The end-user-visible speedup.
- **stems / volume_m3** — median detection count and total timber volume.
  `changed` is expected to be *higher*: the edge fix recovers stems the tiled
  merge silently discarded on the orthomosaic's outer boundary.
- **spread** — max − min stem count across runs. **Non-zero means the pipeline
  is not reproducible**, which is the original's defining defect: it returns a
  different answer each time you run it on the same input.
- **deterministic** — whether stem geometry was byte-identical across runs.
  Reported as `n/a` below two successful runs, because with a single run there
  is exactly one hash and "identical" would be vacuously true.

## Why `--repeats` matters

The original has no `PYTHONHASHSEED` guard, so `connect_stems` iterates a set of
string-hashed objects in a per-process order and the stem count moves run to
run. One run per side would silently attribute that variance to the code
changes. Three is the practical minimum; the summary reports the median *and*
the full spread so the variance stays visible rather than averaged away.
