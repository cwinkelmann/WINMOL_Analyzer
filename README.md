# WINMOL Analyzer

**WINMOL Analyzer** is an open-source QGIS plugin for the **detection and quantification of windthrown tree stems** on UAV-derived orthomosaics. It leverages deep learning and heuristics to identify, reconstruct, and quantify individual fallen trees, supporting salvage operations and sustainable forest management following storm events.

![WINMOL Analyzer Screenshot](documentation/assets/images/dji-9-2216x1662.jpeg)

## 🌪️ Purpose

Severe storms are a major driver of biomass loss in European forests. Knowing the **amount** and **spatial distribution** of windthrown trees is essential for:
- Supporting the planning of salvage operations,
- Reducing the risk of accidents,
- Reducing follow-up biotic, abiotic, and economic damages,
- Supporting sustainable forestry and conservation strategies.

## 🛠️ Features

- Deep learning-based object detection using U-Net
- Skeletonization algorithm for stem detection
- Morphological heuristics for occluded stem reconstruction
- Diameter measurement every 25 cm along each stem
- Volume estimation via truncated cone modeling
- Pre-trained models for **Spruce**, **Beech**, **Spruce Deadwood**, and **General** (mixed stands)

## 🚀 Getting Started

WINMOL Analyzer ships as a **QGIS plugin** and a **standalone command-line
tool**. Inference runs on **ONNX models via onnxruntime** — no TensorFlow, and
no CUDA toolchain required.

### Prerequisites

- **QGIS 3.22 or newer** (LTR recommended) — Windows, macOS, or Linux.
- **Internet access on first run** to download the compute environment — or an
  existing **Python 3.11** interpreter if the machine is offline (see the custom
  environment option in step 2).
- **GPU is optional.** Apple Silicon uses CoreML automatically; NVIDIA users can
  opt into CUDA (see the per-platform table). Inference falls back to CPU
  everywhere else.

### 1. Install the plugin — Windows · macOS · Linux

The steps are identical on all three platforms:

1. Download the latest `WINMOL_Analyzer.zip` from the
   [Releases page](https://github.com/StefanReder/WINMOL_Analyzer/releases).
2. In QGIS, open **Plugins → Manage and Install Plugins → Install from ZIP**.
3. Browse to the downloaded ZIP and click **Install Plugin**.

The plugin then appears in the **Plugins** menu and toolbar.

> **Developers:** `make deploy` copies the full plugin (including the compute
> core) into your QGIS profile's plugin folder:
> `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` (Windows),
> `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
> (macOS), or `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
> (Linux).

### 2. Set up the compute environment (first run)

WINMOL runs inference in its **own Python environment** (onnxruntime + the
geo-stack), isolated from QGIS's interpreter. The first time you open the plugin
it asks how to provide one — reachable any time afterwards via the
**Environment** button in the dialog:

- **Create for me (default).** WINMOL downloads a self-contained **CPython
  3.11** (via
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone),
  no admin rights) and installs `requirements/plugin.txt` into a private venv.
  Identical on Windows, macOS and Linux; the download happens once and is cached.
- **Choose interpreter… — custom environment.** Point WINMOL at an existing
  **Python 3.11** interpreter: a conda env, or any venv with
  `requirements/plugin.txt` installed. Ideal for **offline / firewalled**
  machines, or to reuse a GPU-enabled environment. Your choice is stored in the
  QGIS setting `winmol/python_executable` and applied live — no QGIS restart.

| Platform | Environment notes |
| --- | --- |
| **Windows** | Automatic mode brings its own Python — nothing to pre-install. NVIDIA CUDA: use a custom env with `onnxruntime-gpu`. |
| **macOS** | Nothing to pre-install; Apple Silicon uses the CoreML GPU / Neural-Engine automatically. |
| **Linux** | Automatic mode is self-contained. Pointing at a *system* Python needs `python3-venv` (`sudo apt install python3-venv`). NVIDIA CUDA: custom env with `onnxruntime-gpu`. |

### 3. Standalone / command line (optional)

For batch processing or scripting without QGIS (Python 3.11):

```shell
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements/base.txt                # geo-stack + onnxruntime (no TensorFlow)
python -u winmol_run.py <model.onnx> <input.tif> <stem_map.tif> <out_prefix> <Stems|Trees|Nodes>
```

See [`docs/SETUP.md`](docs/SETUP.md) for GPU providers and model-conversion
details.

### Load and use the plugin

1. Select a UAV orthomosaic as input (preferably **< 3 cm GSD**) — choose a
   loaded layer or browse to a file.
2. Choose a pre-trained model (**Spruce**, **Beech**, **Spruce Deadwood** or
   **General**) or a custom `.onnx` model.
3. *Optional:* adjust the detection and quantification parameters.
4. Run the analysis — results are written to a temporary workspace by default.
5. The detected stems are added to the map canvas; use **Export** to save them
   as a **GeoPackage**.

Refer to the [documentation](https://stefanreder.github.io/WINMOL_Analyzer/) for
further information.

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

- Open an [issue](https://github.com/StefanReder/WINMOL_Analyzer/issues)
- Fork the repository and submit a pull request

## 🙏 Acknowledgements

Developed as part of the WINMOL project. The plugin is trained and validated using 21 UAV orthomosaics of spruce, beech, and mixed stands, with 1747 stems manually annotated and 710 trees measured for validation.

WINMOL Analyzer supports forest managers, ecologists, and researchers in monitoring post-disturbance biomass and improving sustainable forest planning.

---

📬 Questions or feedback? Open a GitHub issue or visit the [official website](https://stefanreder.github.io/WINMOL_Analyzer/)
