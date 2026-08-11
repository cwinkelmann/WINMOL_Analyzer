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
    # the redundant GDAL bench variants were removed 2026-08-11; only
    # `overview` stays as the flag-gated fast path
    for gone in ("fullres", "boundless"):
        monkeypatch.setenv("WINMOL_BENCH_READ", gone)
        with pytest.raises(ValueError):
            resolve_read_strategy(cfg)


def test_graph_aa_wraps_antialiased_and_pins_cpu(tiny_unet_file, monkeypatch):
    """`graph_aa` = the same in-graph Resize with antialias=1: its wrap
    must produce different downsampled pixels than `graph`, and it must
    pin the CPU provider — onnxruntime's CUDA EP mis-executes the
    opset-18 antialias Resize (measured 2026-08-10, ORT 1.19.2 / RTX
    4080: 82 stems vs 478 on the CPU EP, same model+ortho)."""
    pytest.importorskip("onnxruntime")
    from utils.IO import load_model_from_path
    monkeypatch.setenv("WINMOL_BENCH_READ", "graph_aa")
    seg_aa = load_model_from_path(tiny_unet_file)
    assert seg_aa.providers == ["CPUExecutionProvider"]
    monkeypatch.setenv("WINMOL_BENCH_READ", "graph")
    seg = load_model_from_path(tiny_unet_file)
    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, (1, 1024, 1024, 3), dtype=np.uint8)
    d = float(np.abs(seg_aa.predict_on_batch(x)
                     - seg.predict_on_batch(x)).max())
    assert d > 1e-4, "antialias attribute had no effect on the wrapped graph"


def test_cupy_strategy_is_recognized_but_guarded(monkeypatch):
    """`cupy` is a reserved flag value (rc12's path, unported); selecting
    it fails fast at the resolver — the single chokepoint — so no entry
    point can silently degrade it into another strategy's code path."""
    monkeypatch.setenv("WINMOL_BENCH_READ", "cupy")
    from classes.Config import Config
    from utils import Prediction as Pred
    assert "cupy" in Pred._READ_STRATEGIES
    with pytest.raises(RuntimeError, match="[Cc]uPy"):
        Pred.resolve_read_strategy(Config())


def test_load_wraps_model_for_uint8_any_size_input(tiny_unet_file,
                                                   monkeypatch):
    """By default the loaded segmenter takes NHWC uint8 at ANY tile size
    and resizes in-graph -- the wrapped contract, not the raw model's."""
    pytest.importorskip("onnxruntime")
    monkeypatch.delenv("WINMOL_BENCH_READ", raising=False)
    from utils.IO import load_model_from_path
    seg = load_model_from_path(tiny_unet_file)
    out = seg.predict_on_batch(
        np.zeros((1, 299, 299, 3), dtype=np.uint8))
    assert out.shape[1:3] == (512, 512)


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
    ("graph_aa", np.uint8, 299),    # AA variant feeds the model identically
    ("overview", np.float32, 512),  # bench override still wins
])
def test_stream_feeds_model_per_strategy(tmp_path, test_geotiff_file,
                                         monkeypatch, env, want_dtype,
                                         want_h):
    pytest.importorskip("rasterio")
    if env is None:
        monkeypatch.delenv("WINMOL_BENCH_READ", raising=False)
    else:
        monkeypatch.setenv("WINMOL_BENCH_READ", env)
    from classes.Config import Config
    from utils import Prediction as Pred
    out = str(tmp_path / "stem.tif")
    spy = _SpyModel()
    Pred.predict_stream_to_raster(test_geotiff_file, out, spy, Config())
    assert os.path.exists(out)
    assert spy.batches, "model was never called"
    dtype, shape = spy.batches[0]
    assert dtype == want_dtype
    assert shape[1] == want_h
