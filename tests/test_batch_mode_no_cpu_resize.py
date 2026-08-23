"""Batch (multi-GPU) prediction must resize in-graph, like the stream does.

Measured on carrot 2026-08-23, 8xH100, `--jobs 1` on one orthomosaic:

    Multi-GPU prediction 481/1089 | 106.0 tiles/min | avg infer 1.974s

against 8569 tiles/min aggregate for the streaming path. The raw model
runs a 512^2 tile in 2.97 ms on the same card, so ~99.8% of that "infer"
was not inference: `prediction_worker` loaded the model with
wrap_preprocess=False and `_predict_batch` forced read_strategy="native",
so every tile's normalize + 1250^2 -> 512^2 bicubic resample ran on the
CPU -- exactly the work the in-graph resize exists to avoid -- inside the
block the timer calls inference.

The stream producer's contract under `graph` is the one to match: the
IMAGE stays native uint8 and is resized on device; only the one-channel
validity mask is resized (nearest) by the producer.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class RecordingModel:
    """Stands in for the wrapped session and records what it was fed."""

    def __init__(self, out_hw):
        self.out_hw = out_hw
        self.seen_dtype = None
        self.seen_shape = None

    def predict_on_batch(self, x):
        x = np.asarray(x)
        self.seen_dtype = x.dtype
        self.seen_shape = x.shape
        return np.zeros((x.shape[0], self.out_hw[0], self.out_hw[1], 1),
                        dtype=np.float32)


def test_batch_mode_feeds_native_uint8_to_the_wrapped_model():
    from classes.Config import Config
    from utils import PredictWorkers as PW

    cfg = Config()
    native = 1250
    model = RecordingModel((cfg.img_height, cfg.img_width))
    tiles = [np.zeros((native, native, 3), dtype=np.uint8)]
    # Producer-resized, as the graph contract requires.
    masks = [np.ones((cfg.img_height, cfg.img_width), dtype=bool)]

    PW._predict_batch(tiles, masks, model, cfg)

    assert model.seen_dtype == np.uint8, (
        f"batch mode CPU-converted the tile to {model.seen_dtype}")
    assert tuple(model.seen_shape[1:3]) == (native, native), (
        f"batch mode CPU-resized the tile to {model.seen_shape[1:3]}")


def test_producer_resizes_only_the_mask_under_the_graph_strategy(
        test_geotiff_file):
    """The graph contract: image native, mask on the model grid.

    `_prepare_inference_batch` returns the mask UNRESIZED under `graph`
    ("Producers already resized the masks to the model grid"), so if the
    batch producer hands it a native-sized mask the binarize step indexes
    a 1250^2 mask against a 512^2 prediction.
    """
    import rasterio

    from classes.Config import Config
    from utils import PredictWorkers as PW

    cfg = Config()
    out_size = PW._graph_out_size(cfg)
    assert out_size == (cfg.img_height, cfg.img_width), \
        "graph is the default strategy, so a model grid is expected"

    job = {"src_row": 0, "src_col": 0, "src_width": 300, "src_height": 300,
           "dst_row": 0, "dst_col": 0, "tile_index": 0}

    with rasterio.open(test_geotiff_file) as src:
        indexes = [1, 2, 3]
        tiles, masks, _ = PW._read_batch_jobs(src, indexes, [job], out_size)

    assert tiles[0].shape[:2] == (300, 300), "the image must stay native"
    assert masks[0].shape == out_size, "the mask must be on the model grid"


def test_non_graph_strategies_keep_native_masks(monkeypatch):
    """Only the graph path changes; `native` must behave as it always did."""
    from classes.Config import Config
    from utils import PredictWorkers as PW

    monkeypatch.setenv("WINMOL_BENCH_READ", "native")
    assert PW._graph_out_size(Config()) is None
