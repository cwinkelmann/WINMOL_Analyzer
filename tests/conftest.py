"""Make the repo root importable no matter where pytest is invoked
from (repo root or tests/). Several test modules also do this insert
themselves; this covers the ones that import winmol_batch / utils /
plugin_utils directly."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture(autouse=True)
def _isolated_autotune_cache(tmp_path, monkeypatch):
    """Give every test its own throwaway autotune cache file.

    Without this, ``_autotune_batch_size`` (utils/Prediction.py) resolves
    ``plugin_utils.autotune_cache.cache_path()`` to the SAME per-user
    location a real WINMOL run uses (e.g. ``~/Library/Caches/winmol`` on
    macOS) whenever a test does not set ``$WINMOL_AUTOTUNE_CACHE`` itself.
    That would (a) leave real files behind on the machine running the
    suite, and (b) let two unrelated tests whose (fake model, Config) hash
    to the same cache key leak a batch size between them. A test that
    wants a specific cache file (or wants to assert the real per-user
    default) still wins by setting ``WINMOL_AUTOTUNE_CACHE`` itself after
    this fixture runs -- monkeypatch layers cleanly.
    """
    monkeypatch.setenv(
        "WINMOL_AUTOTUNE_CACHE", str(tmp_path / "autotune-test-cache.json"))


def build_tiny_unet(path):
    """1-conv sigmoid segmenter, NHWC [b,512,512,3] -> [b,512,512,1].
    Shared by the compute-contract and read-strategy tests."""
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper
    s = 512
    rng = np.random.default_rng(0)
    w = helper.make_tensor("w", TensorProto.FLOAT, [1, 3, 1, 1],
                           rng.normal(size=3).astype(np.float32))
    nodes = [
        helper.make_node("Transpose", ["input"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node("Conv", ["nchw", "w"], ["c"]),
        helper.make_node("Sigmoid", ["c"], ["nchw_out"]),
        helper.make_node("Transpose", ["nchw_out"], ["output"],
                         perm=[0, 2, 3, 1]),
    ]
    graph = helper.make_graph(
        nodes, "segmenter",
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["b", s, s, 3])],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["b", s, s, 1])],
        [w])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return str(path)


def build_test_geotiff(path):
    """600x600 px, 3-band uint8, EPSG:32633, 5 cm pixels."""
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin
    rng = np.random.default_rng(42)
    data = rng.integers(1, 255, size=(3, 600, 600), dtype=np.uint8)
    profile = {
        "driver": "GTiff", "width": 600, "height": 600, "count": 3,
        "dtype": "uint8", "crs": rasterio.crs.CRS.from_epsg(32633),
        "transform": from_origin(400000.0, 5900000.0, 0.05, 0.05),
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data)
    return str(path)


@pytest.fixture(scope="session")
def tiny_unet_file(tmp_path_factory):
    """One model build per session; build_preprocessed_model's mtime-keyed
    wrap cache then hits across tests instead of re-wrapping per test."""
    return build_tiny_unet(tmp_path_factory.mktemp("model") / "m.onnx")


@pytest.fixture(scope="session")
def test_geotiff_file(tmp_path_factory):
    return build_test_geotiff(tmp_path_factory.mktemp("raster") / "ortho.tif")
