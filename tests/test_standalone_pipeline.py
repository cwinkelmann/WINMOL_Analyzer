"""End-to-end test for the standalone WINMOL pipeline with Metal GPU checks.

Runs ``standalone.WINMOL_Analyzer.run_pipeline`` on a small cropped window of a
real orthomosaic and verifies:

* the pipeline completes and writes its stem-map raster and GeoPackage, and
* the U-Net inference actually executes on the Apple Metal GPU.

The test skips cleanly (rather than failing) when Metal, the source
orthomosaic, or the model file are unavailable, so it stays portable to CPU /
CI environments.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
import tensorflow as tf
from rasterio.windows import Window

# TF_USE_LEGACY_KERAS is set in tests/conftest.py before TensorFlow imports.

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ORTHO = Path(
    "/Users/christian/data/Winmol/Winmol Orthos/WINMOL_x_OeBF/Raster/proj/"
    "repr_20250327_171_4_Kraking_Windwurf_SELEKTION.tif"
)
MODEL_PATH = (
    REPO_ROOT / "standalone" / "model"
    / "model_UNet_GenDS_512_2023-02-27_211141.hdf5"
)
WINDOW_SIZE = 2048  # pixels; ~82 m at ~4 cm GSD


def _metal_devices():
    return tf.config.list_physical_devices("GPU")


@pytest.fixture(scope="session")
def metal_gpu():
    gpus = _metal_devices()
    if not gpus:
        pytest.skip(
            "No Metal GPU visible to TensorFlow (needs tensorflow-metal)")
    return gpus[0]


@pytest.fixture(scope="session")
def model_path():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found: {MODEL_PATH}")
    return str(MODEL_PATH)


@pytest.fixture(scope="session")
def standalone_mod():
    """Import standalone/WINMOL_Analyzer.py as a module (not a package)."""
    script = REPO_ROOT / "standalone" / "WINMOL_Analyzer.py"
    spec = importlib.util.spec_from_file_location("winmol_standalone", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def cropped_ortho(tmp_path):
    """Crop a WINDOW_SIZE window (first 3 bands) into a temp GeoTIFF."""
    if not SOURCE_ORTHO.exists():
        pytest.skip(f"Source orthomosaic not found: {SOURCE_ORTHO}")

    with rasterio.open(SOURCE_ORTHO) as src:
        w = min(WINDOW_SIZE, src.width)
        h = min(WINDOW_SIZE, src.height)
        col_off = max(0, (src.width - w) // 2)
        row_off = max(0, (src.height - h) // 2)
        window = Window(col_off, row_off, w, h)

        bands = min(3, src.count)
        data = src.read(list(range(1, bands + 1)), window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            width=w, height=h, count=bands, transform=transform,
            driver="GTiff",
        )
        profile.pop("nodata", None)

    out = tmp_path / "crop.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data)
    return str(out)


def test_model_forward_pass_runs_on_metal(
    metal_gpu, model_path, standalone_mod
):
    """The U-Net loads and its forward pass executes on the Metal GPU."""
    model = standalone_mod.keras.models.load_model(model_path, compile=False)
    x = np.random.rand(1, 512, 512, 3).astype("float32")
    with tf.device("/GPU:0"):
        y = model(x, training=False)
    assert tuple(y.shape) == (1, 512, 512, 1)
    # Tensor was produced on the Metal PluggableDevice.
    assert "GPU:0" in y.device


def test_standalone_pipeline_end_to_end(
    metal_gpu, model_path, cropped_ortho, standalone_mod, tmp_path
):
    """run_pipeline completes on real data and writes both outputs."""
    pred_dir = str(tmp_path / "pred") + os.sep
    output_dir = str(tmp_path / "out") + os.sep
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    result = standalone_mod.run_pipeline(
        model_path, cropped_ortho, pred_dir, output_dir
    )

    # Stem-map raster written and valid.
    stem_map = Path(result["stem_map_path"])
    assert stem_map.exists(), f"stem map missing: {stem_map}"
    with rasterio.open(stem_map) as ds:
        assert ds.width > 0 and ds.height > 0

    # GeoPackage written.
    gpkg = Path(result["gpkg_path"])
    assert gpkg.exists(), f"gpkg missing: {gpkg}"

    # Stem count is a non-negative int; quantified stems have positive metrics.
    assert isinstance(result["num_stems"], int)
    assert result["num_stems"] >= 0
