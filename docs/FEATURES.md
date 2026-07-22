# Desired Features


### load models as keras format
https://www.tensorflow.org/tutorials/keras/save_and_load
hdf5 seems to deprecated, but should still work in 2.21

.keras model loading should work too.

### onnx loading
load models as onnx format, remove dependencies to tensorflow and keras to migrate to pytorch completely.


#### Speed up inference and the vectorisation later
use some JIT or other compilation tricks to speed it up.

### fix the QGIS plugin building
it should work on windows and linux with multiple QGIS versions


### Fix bugs found in the two code reviews.


### CI on GITHUB
the tests should run in github to ensure changes are not ruining something


### Fix QGIS plugin installation
currently the plugin fails to install on a mac but also on python 3.11 on linux with cuda. The venv installation process is quite flakey. In general multipe ways of installing exist. 
A classic windows install should work for QGIS 3.44 and 4, in linux one could install qgis into a conda environment or via apt. The plugin should be installable via the QGIS plugin manager and also via a zip file. The installation process should be documented in detail. On a mac options are slightly reduced. In all cases a CUDA GPU or Apple MPS should be usable. 

### Autotune static parameters like batch size
the autotune is quite static, there is no point in running it all the time. It should be possible to run it once and then save the results for later use. The autotune should also be able to tune more parameters like batch size, tile size, number of workers etc. The results should be saved in a config file which can be loaded later. The autotune should also be able to run on a different machine than the one running QGIS.


#### API connectivity
Instead of having a analyser locally, a worker on a beefy GPU machine could be used. The QGIS plugin would then send the orthomosaic to the worker and receive the results back. This would allow to use a GPU on a different machine than the one running QGIS. The API should be secure and support authentication and authorization. The API should be documented in detail.

#### QGIS features: manual canvas analysis
it Would be great if I can draw stem outlines and then look into what the segmentation model predicts, but also what the skeletonisation predicts. This would allow to compare the two methods and also to manually correct the predictions. The manual corrections should be saved and could be used to retrain the model.
For this the plugin would need to read either a 1-channel raster or shapefile which closed polygons.


### Local Environment Setup
currently the setup seems incomplete. There are a ton of scripts inthe root folder which are not documented and have broken imports, from qgis.PyQt.QtCore import QObject, pyqtSignal

### Multiple geotiff processing in QGIS
Assuming there are multiple geotiff raster layers in QGIS, it should be possible to run the winmol analyser on all of them in one go. The plugin should allow to select multiple layers and then run the analysis on all of them. The results should be saved in a separate folder/group with the same name as the input layer. The results should be added to the QGIS canvas automatically.


### Speedup the model inference
candidates would quantization, prunning, onnx speedup Especially on cpu it should run faster
#### Do a speed and feature benchmark
Are all the optimisations faster and by how much? How does accuracy change in the end.

### Persist the batch-size autotune result — DONE
Shipped in `plugin_utils/autotune_cache.py`; `prediction_batch_autotune` is now
tri-state and defaults to `"auto"`: tune ONCE, keyed by (hardware + model file
+ execution provider + tile geometry), persist to a small JSON cache, and reuse
it on every later run. `True` forces a re-tune, `False` disables it (pinned by
the test suite and `benchmark/`), and `WINMOL_BATCH_AUTOTUNE=off|auto|force`
overrides without editing config. A corrupt or unwritable cache degrades to a
re-measurement, never a failure. See `docs/CONFIG.md` for the cache location
and how to clear it.

Measured on an M2/CoreML with the 9-tile crop fixture: tuning costs ~62 s and
buys 0.169 → 0.159 s/tile (5.9 %), i.e. ~18 s per run on the 580-tile ortho in
`docs/benchmark-full-ortho.md` — it pays for itself after ~4 runs and is free
thereafter. The optimum is per-device (batch 4-5 on this M2; a large CUDA GPU
will likely prefer a much bigger batch), which is exactly why it is measured
rather than hardcoded.

### Split semantic segmentation from vectorization

Today the two halves are welded together: `winmol_run.py … Stems` stops after
prediction and writes the binary stem-map raster, but there is **no way in** —
nothing can take an existing stem map and run only the vector stage. The only
use of `stem_map_path` is the merge-time edge fix; the vector phase always
follows a fresh inference in the same process.

That makes tuning the vector side needlessly expensive. Every experiment with
`edge_buffer_m`, the minimum stem length, the 25 cm diameter step, or the
`connect_stems` join thresholds costs a full re-inference over the whole
orthomosaic — even though the segmentation is bit-identical each time. On the
full-ortho benchmark the vector phase is already **~73 %** of a DeepLab run
(173 s of 236 s), so the iteration loop is dominated by work that did not need
redoing.

Proposal: add a `Vectorize` process type (or `--from-stem-map`) that takes a
stem-map GeoTIFF as its input instead of an orthomosaic and runs
skeletonize → build parts → connect → quantify → merge. The seam already
exists — the golden fixtures are staged exactly along it
(`tests/fixtures/stage_build_stem_parts.json.gz`, `stage_connect_stems.json.gz`,
`stage_quantified_*.json.gz`), so the stage functions are separable in practice
and the tests already prove each boundary.

Benefits beyond speed: parameter sweeps become cheap enough to automate; a
segmentation mask from *any* source can be vectorized (see the exemplar-based
backbone below, or a hand-corrected mask); and a bad vectorization can be
re-run on a stored stem map without touching the GPU. Worth pairing with a
small manifest next to the stem map recording the model, config and commit that
produced it, so a vectorization result is still traceable to its segmentation.

### Exemplar-based segmentation with a self-supervised backbone (DINO)

Instead of a U-Net trained on the 21 annotated orthomosaics, use a
self-supervised vision backbone (DINOv2/DINOv3, ideally a variant pretrained on
aerial/remote-sensing imagery) as a frozen feature extractor, let the user
click a handful of **exemplar** stems in the canvas, and build the mask by
matching patch features against those exemplars (cosine similarity in feature
space, optionally a light logistic head or k-NN over the exemplar set).

Why it is attractive here: adapting to a new site, species mix, season or sensor
currently means retraining and re-annotating. An exemplar approach adapts in
seconds with a few clicks and no labels, which fits the actual field workflow —
a forester looking at one storm event who wants *these* stems found. It also
composes well with the QGIS canvas-interaction feature already listed above.

**The honest risk is spatial resolution.** ViT backbones work on patches
(DINOv2 uses 14 px), so raw patch-level features are far coarser than the
structures being segmented: a stem at 2–4 cm GSD is only a handful of pixels
wide, and a naive patch-similarity mask would be uselessly blocky. Any serious
attempt needs one of the high-resolution adaptations — FeatUp-style feature
upsampling, sliding-window inference at overlapping offsets, or a shallow
decoder trained on the existing annotations while the backbone stays frozen.
That last option is probably the pragmatic first experiment: it reuses the
labels already available and only asks whether frozen DINO features beat a
from-scratch U-Net encoder.

Integration is cheap on our side, which lowers the cost of trying it. The
pipeline consumes a **binarised** mask, so anything that emits one is a drop-in;
`utils/onnx_runtime.py` already reads the model's declared layout and handles
**NCHW** exports (i.e. PyTorch → ONNX) alongside the NHWC Zenodo conversions,
so a DINO head exported to ONNX needs no loader changes provided it honours the
`[N,512,512,3] → [N,512,512,1]` external contract. Combined with the
segmentation/vectorization split above, this could be evaluated end-to-end
against the golden fixtures without disturbing the existing model path.

### Inference Docker Containers
Currently there is a dockerfile.blackwell, Dockefile_olive container

These need to be checked and updated if they work on these beefy machine machines with multiple GPUS

Create slimmer container which work on Blackwell multi GPU machine


## Backlog

### implement Python 3.14
the current version is python 3.14, 3.11 is end of life soon too. With Python 3.14 everywhere we would be safe for a while. Matrix testing for multiple python version could make sure that the code works on all versions. The QGIS plugin should also be tested on multiple python versions.

### Code seperation - to be planned
seperate the QGIS plugin from the stem inference code. 
