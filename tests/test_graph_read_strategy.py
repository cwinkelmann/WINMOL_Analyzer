"""The in-graph (v0.5-equivalent) resize is the default read strategy.

docs/resize-mechanics.md is the why. These tests pin the wiring: the
config default, the WINMOL_BENCH_READ override kept for benchmarking,
model wrapping at load time, and the stream feeding native uint8 tiles
to the wrapped model when nothing is overridden.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def test_config_defaults_to_graph_strategy():
    from classes.Config import Config
    assert Config.prediction_read_strategy == "graph"


def test_env_var_overrides_config_for_benching(monkeypatch):
    from classes.Config import Config
    from utils.Prediction import resolve_read_strategy
    cfg = Config()
    monkeypatch.delenv("WINMOL_BENCH_READ", raising=False)
    assert resolve_read_strategy(cfg) == "graph"
    monkeypatch.setenv("WINMOL_BENCH_READ", "overview")
    assert resolve_read_strategy(cfg) == "overview"
    # the historical bench name for the in-graph path stays an alias
    monkeypatch.setenv("WINMOL_BENCH_READ", "onnx_gpu")
    assert resolve_read_strategy(cfg) == "graph"


def test_cupy_strategy_is_recognized_but_guarded(tmp_path, monkeypatch):
    """`cupy` is a valid flag value (rc12's path, for A/B on CUDA boxes),
    but selecting it without the port/hardware fails fast and clearly."""
    pytest.importorskip("rasterio")
    monkeypatch.setenv("WINMOL_BENCH_READ", "cupy")
    from classes.Config import Config
    from utils import Prediction as Pred
    assert Pred.resolve_read_strategy(Config()) == "cupy"
    uav = _build_geotiff(tmp_path / "ortho.tif")
    with pytest.raises(RuntimeError, match="[Cc]uPy"):
        Pred.predict_stream_to_raster(
            uav, str(tmp_path / "stem.tif"), _SpyModel(), Config())


def _build_model(path):
    """1-conv sigmoid segmenter, NHWC [b,512,512,3] -> [b,512,512,1]."""
    onnx = pytest.importorskip("onnx")
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


def test_load_wraps_model_for_uint8_any_size_input(tmp_path, monkeypatch):
    """By default the loaded segmenter takes NHWC uint8 at ANY tile size
    and resizes in-graph -- the wrapped contract, not the raw model's."""
    pytest.importorskip("onnxruntime")
    monkeypatch.delenv("WINMOL_BENCH_READ", raising=False)
    from utils.IO import load_model_from_path
    seg = load_model_from_path(_build_model(tmp_path / "m.onnx"))
    out = seg.predict_on_batch(
        np.zeros((1, 299, 299, 3), dtype=np.uint8))
    assert out.shape[1:3] == (512, 512)


def _build_geotiff(path):
    """600x600 px, 3-band uint8, 5 cm pixels -> px_per_tile-1 = 299."""
    rasterio = pytest.importorskip("rasterio")
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


class _SpyModel:
    """Records what the stream feeds it; answers like a 512 segmenter."""

    def __init__(self):
        self.batches = []

    def predict_on_batch(self, x):
        self.batches.append((x.dtype, x.shape))
        return np.zeros((x.shape[0], 512, 512, 1), dtype=np.float32)


@pytest.mark.parametrize("env,want_dtype,want_h", [
    # graph default: native uint8 (window = px_per_tile-1 = 299 at 5 cm)
    (None, np.uint8, 299),
    ("overview", np.float32, 512),  # bench override still wins
])
def test_stream_feeds_model_per_strategy(tmp_path, monkeypatch,
                                         env, want_dtype, want_h):
    pytest.importorskip("rasterio")
    if env is None:
        monkeypatch.delenv("WINMOL_BENCH_READ", raising=False)
    else:
        monkeypatch.setenv("WINMOL_BENCH_READ", env)
    from classes.Config import Config
    from utils import Prediction as Pred
    uav = _build_geotiff(tmp_path / "ortho.tif")
    out = str(tmp_path / "stem.tif")
    spy = _SpyModel()
    Pred.predict_stream_to_raster(uav, out, spy, Config())
    assert os.path.exists(out)
    assert spy.batches, "model was never called"
    dtype, shape = spy.batches[0]
    assert dtype == want_dtype
    assert shape[1] == want_h
