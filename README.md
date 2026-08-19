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

1. Download the latest release from the [Releases page](https://github.com/StefanReder/WINMOL_Analyzer/releases) (a `.zip` file of the plugin)
2. In QGIS, go to **Plugins** > **Manage and Install Plugins**
3. Click the **Install from ZIP** tab
4. Browse to the downloaded `.zip` file and click **Install Plugin**

Once installed, the plugin will be available via the **Plugins** menu.

### Load and Use the Plugin

1. Select an UAV orthomosaic as input file (preferably <3 cm GSD) and set an output file path
2. Select a pre-trained model (**Spruce**, **Beech** or **General**) or a **Custom** model
3. Optional: Adjust the parameters
4. Run detection and quantification
5. The results are exported as geojson and added to the canvas

Refer to the [documentation](https://stefanreder.github.io/WINMOL_Analyzer/) for further information.

## 📖 Related Publications

Please cite the following peer-reviewed studies if you use WINMOL Analyzer in your work:

1. **Reder, S., Kruse, M., Miranda, L., Voss, N., & Mund, J.-P. (2025).**  
   *Unveiling wind-thrown trees: Detection and quantification of wind-thrown tree stems on UAV-orthomosaics based on UNet and a heuristic stem reconstruction.*  
   *Forest Ecology and Management, 578, 122411.*  
   [https://doi.org/10.1016/j.foreco.2024.122411](https://doi.org/10.1016/j.foreco.2024.122411)

2. **Reder, S., Mund, J.-P., Albert, N., Waßermann, L., & Miranda, L. (2022).**  
   *Detection of windthrown tree stems on UAV-orthomosaics using U-Net convolutional networks.*  
   *Remote Sensing, 14(1), 75.*  
   [https://doi.org/10.3390/rs14010075](https://doi.org/10.3390/rs14010075)

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
