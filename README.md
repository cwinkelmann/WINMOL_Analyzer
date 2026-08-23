# WINMOL Analyzer

**WINMOL Analyzer** is an open-source QGIS plugin for the **detection and quantification of windthrown tree stems** on UAV-derived orthomosaics. It leverages deep learning and heuristics to identify, reconstruct, and quantify individual fallen trees, supporting salvage operations and sustainable forest management following storm events.

![WINMOL Analyzer Screenshot](documentation/assets/images/dji-9-2216x1662.jpeg) <!-- Replace with actual image -->

## 🌪️ Purpose

Severe storms are a major driver of biomass loss in European forests. Knowing the **amount** and **spatial distribution** of windthrown trees is essential for:
- Supporting the planning of salvage operations,
- Reduing the risk of accidents,
- Reducing follow-up biotic, abiotic, and economic damages,
- Supporting sustainable forestry and conservation strategies.

## 🛠️ Features

- Deep learning-based object detection using U-Net
- Skeletonization algorithm for stem detection
- Morphological heuristics for occluded stem reconstruction
- Diameter measurement every 25 cm along each stem
- Volume estimation via truncated cone modeling
- Pre-trained models for **Spruce**, **Beech**, and **General** (mixed stands)

## 🚀 Getting Started

### Prerequisites
- [QGIS 3.x](https://qgis.org/) (recommended: latest LTR version)
- Recommended: NVIDIA GPU with CUDA 8.x support or above
- [CUDA Drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [CdDNN](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html)

### Installation as QGIS Plugin

1. Download the latest release from the [Releases page](https://github.com/your-repo/releases) (a `.zip` file of the plugin)
2. In QGIS, go to **Plugins** > **Manage and Install Plugins**
3. Click the **Install from ZIP** tab
4. Browse to the downloaded `.zip` file and click **Install Plugin**

Once installed, the plugin will be available via the **Plugins** menu.

### Load and Use the Plugin

1. Select an UAV orthomosaic as input file (preferably <3 cm GSD) and set an output file path
2. Select a pre-trained model (**Spruce**, **Beech** or **General**) or a **Custom** model
3. Otional: Adjust the parameters
4. Run detection and quantification
5. The results are exported as geojson and added to the canvas

Refer to the [documentation](https://stefanreder.github.io/WINMOL_Analyzer/) for further information.

## 🖥️ Command-line and batch processing

Besides the QGIS plugin, the pipeline runs headless — one orthomosaic at a
time (`winmol_run.py`), a whole folder (`winmol_batch.py`), or in Docker.

### ⚠️ Prepare your orthomosaics first: they need overviews

**This is the single most important thing to get right for large
orthomosaics.** Prediction resamples every tile to the model's grid during
the GDAL read. When the file has overview levels, GDAL serves that read
from an already-decimated level. When it does not, it must read *every
source pixel* at full resolution and shrink it in RAM — roughly twice the
read time and four times the bytes, per tile.

On a small ortho this barely shows. On a large one it is the difference
between a run that finishes and one that does not. Measured on a
99231-tile orthomosaic (392558 × 335327 px):

| input | throughput |
|---|---|
| COG **with** overviews | **~4200 tiles/min, flat for the whole run** |
| same data, no overviews | starts ~2600, decays, collapses to **~80 tiles/min** |

The decay is gradual and then sudden, which makes it easy to mistake for a
hang. If a run starts fast and then gets slower and slower, check the
overviews first.

**Check whether a file has them:**

```bash
gdalinfo your_ortho.tif | grep -i overview
```

**Build them (once per file):**

```bash
gdaladdo -ro -r average your_ortho.tif 2 4 8 16 32 64 128
```

`-ro` writes a `.ovr` sidecar next to the file and leaves the original
untouched — safe for read-only or shared data. QGIS display gets faster
too.

**Better still, produce a Cloud-Optimized GeoTIFF**, which carries
overviews and is tiled for windowed reads:

```bash
gdal_translate input.tif output_cog.tif -of COG \
    -co COMPRESS=DEFLATE -co BLOCKSIZE=512
```

The pipeline warns at startup when an input over 2 GB has no overviews. It
will not build them for you unless asked (see `WINMOL_BUILD_OVERVIEWS`).

### Single orthomosaic

```bash
python -u winmol_run.py <model.onnx> <input.tif> <stem_map.tif> <out_prefix> Nodes
```

`Stems` writes only the binary stem-map raster; `Trees` / `Nodes` also
vectorise into a GeoPackage.

### A folder of orthomosaics

```bash
python winmol_batch.py Spruce_Deadwood \
    --input  ./orthos \
    --output ./results \
    --jobs 2
```

Models (`Spruce`, `Beech`, `Spruce_Deadwood`, `General`) are downloaded and
checksum-verified into `--model-dir` on first use. `--jobs N` processes N
orthomosaics concurrently, spreading them across the available GPUs; each
job gets an equal share of the CPU budget. `--merge` stitches tiled outputs
into a single GeoPackage.

### Docker

Two images, because `onnxruntime` and `onnxruntime-gpu` cannot be installed
together. Every release tag publishes both to GHCR:

```bash
docker pull ghcr.io/cwinkelmann/winmol-analyzer-gpu:latest
docker pull ghcr.io/cwinkelmann/winmol-analyzer-cpu:latest
```

Or build them yourself:

```bash
docker build -f docker/Dockerfile --build-arg VARIANT=cpu -t winmol:cpu .
docker build -f docker/Dockerfile --build-arg VARIANT=gpu -t winmol:gpu .
```

The GPU image needs only an NVIDIA driver (**r525 or newer**) and
`nvidia-container-toolkit` — no system CUDA toolkit, because the CUDA
userspace ships inside the image as wheels. It carries compiled kernels for
compute capabilities 6.0, 7.0, 7.5, 8.0, 8.6 and 9.0 and **no PTX**, so it
covers Pascal through Hopper — GTX 10-series and RTX 20/30/40-series
(desktop and laptop), T4, A100, H100 — but cannot run on Maxwell or older,
nor on Blackwell (RTX 50-series, B200). Use the CPU image there.

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v /path/to/orthos:/data/input:ro \
  -v /path/to/results:/data/output \
  -v /path/to/models:/data/models \
  winmol:gpu Spruce_Deadwood
```

`--user` is what makes the output land as **you** rather than as the image's
uid 1000: a bind mount is governed by the host's ownership, so without it
anyone whose host uid is not 1000 gets `Permission denied` writing results.
It is also why the results are yours to delete afterwards.

The models mount must be **writable**. Weights are immutable, so `:ro` is the
natural instinct, but the first run also writes a small derived graph
(`.winmol_pre_<hash>.onnx`, the in-graph resize) next to them; on a read-only
mount that falls back to a temp dir and is rebuilt every run.

On a **rootless** Docker daemon, container uids map into your subuid range
instead, so `--user $(id -u)` does not name your host account — run as
`--user 0:0` there, which maps to the host user who owns the daemon.

The container sizes itself to the resources **it** has rather than the
host's, so `--memory` and `--cpus` are respected instead of ignored:

```bash
docker run --rm --memory=8g --cpus=4 ... winmol:cpu Spruce_Deadwood
```

Useful environment variables (all optional — everything is autodetected):

| variable | meaning |
|---|---|
| `WINMOL_JOBS` | orthomosaics in parallel (default: from GPU count and memory) |
| `WINMOL_PROCESS_TYPE` | `Stems`, `Trees` or `Nodes` (default `Nodes`) |
| `WINMOL_BUILD_OVERVIEWS=1` | run `gdaladdo` on inputs that lack overviews (needs a writable input mount) |
| `GDAL_CACHEMAX` | GDAL block cache in MB. Defaults to ~20% of the memory budget per job — **do not set this small**: a starved cache is what makes large runs collapse |
| `WINMOL_MODEL_DIR` | where models are downloaded and cached |

## 📖 Related Publications

Please cite the following peer-reviewed studies if you use WINMOL Analyzer in your work:

1. **Reder, S., Kruse, M., Miranda, L., Voss, N., & Mund, J.-P. (2025).**  
   *Unveiling wind-thrown trees: Detection and quantification of wind-thrown tree stems on UAV orthomosaics based on UNet and a heuristic stem reconstruction.*  
   *Forest Ecology and Management, 578, 122411.*  
   [https://doi.org/10.1016/j.foreco.2024.122411](https://doi.org/10.1016/j.foreco.2024.122411)

2. **Reder, S., Mund, J.-P., Albert, N., & Miranda, L. (2024).**  
   *Detection of windthrown tree stems on UAV-orthomosaics using U-Net convolutional networks.*  
   *Remote Sensing.*  
   [https://doi.org/10.3390/rs16244710](https://doi.org/10.3390/rs16244710)

## 🤝 Contributing

We welcome contributions from the community! Whether you want to:
- Report a bug 🐞
- Suggest a new feature 💡
- Improve the documentation ✍️
- Submit a pull request 🔧

…your input is appreciated!

- Open an [issue](https://github.com/your-repo/issues)
- Fork the repository and submit a pull request

## 🙏 Acknowledgements

Developed as part of the WINMOL project. The plugin is trained and validated using 21 UAV orthomosaics of spruce, beech, and mixed stands, with 1747 stems manually annotated and 710 trees measured for validation.

WINMOL Analyzer supports forest managers, ecologists, and researchers in monitoring post-disturbance biomass and improving sustainable forest planning.

---

📬 Questions or feedback? Open a GitHub issue or visit the [official website](https://stefanreder.github.io/WINMOL_Analyzer/)
