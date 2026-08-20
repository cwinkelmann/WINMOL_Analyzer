"""Proves the CLI prediction path is TensorFlow-free.

Five groups:
1. ``_resize_batch`` on imagery (order=3, bicubic-like) resizes NHWC batches
   and takes an identity fast path when already at the target size.
2. ``_resize_batch`` on masks (order=0, nearest) keeps values binary.
3. ``_predict_batch_adaptive`` halves the micro-batch on OOM-shaped
   ``MemoryError``/``RuntimeError`` and re-raises anything else.
4. ``utils.Prediction`` / ``utils.PredictWorkers`` import cleanly with
   TensorFlow poisoned out of ``sys.modules`` -- proving neither module
   needs it, even though this conda env has TensorFlow installed.
5. ``TileBatchProducer(out_size=...)`` resamples tiles onto the model grid
   during the GDAL read (cubic imagery / nearest mask), so tiles arrive
   already sized and ``_resize_batch``'s identity fast path short-circuits
   the skimage resize -- including for a window clipped at the raster edge.
"""
import queue
import sys

import numpy as np
import pytest


# --- Group 1 & 2: _resize_batch -------------------------------------------

def test_resize_batch_imagery_resizes_to_target_shape():
    from utils.Prediction import _resize_batch

    batch = np.random.rand(2, 700, 700, 3).astype(np.float32)
    out = _resize_batch(batch, (512, 512), order=3)

    assert out.shape == (2, 512, 512, 3)
    assert out.dtype == np.float32


def test_resize_batch_identity_fast_path_returns_same_object():
    from utils.Prediction import _resize_batch

    batch = np.random.rand(2, 512, 512, 3).astype(np.float32)
    out = _resize_batch(batch, (512, 512), order=3)

    assert out is batch


def test_resize_batch_masks_nearest_stays_binary():
    from utils.Prediction import _resize_batch

    mask = np.zeros((2, 20, 20, 1), dtype=np.float32)
    mask[:, :10, :10, :] = 1.0
    out = _resize_batch(mask, (8, 8), order=0)

    assert out.shape == (2, 8, 8, 1)
    assert set(np.unique(out).tolist()).issubset({0.0, 1.0})


# --- Group 3: _predict_batch_adaptive OOM retry ----------------------------

def test_predict_batch_adaptive_halves_batch_on_memory_error(monkeypatch):
    from utils import Prediction as Pred

    calls = []

    def fake_core(raw_tiles, raw_masks, model, config):
        calls.append(len(raw_tiles))
        if len(raw_tiles) > 1:
            raise MemoryError("Unable to allocate 4.00 GiB for an array")
        return [f"core-for-{len(raw_tiles)}"]

    monkeypatch.setattr(Pred, "_predict_batch_core", fake_core)

    result, used = Pred._predict_batch_adaptive(
        ["t"] * 4, ["m"] * 4, object(), object(), 4)

    assert used == 1
    # All FOUR tiles come back. This assertion used to read
    # `result == ["core-for-1"]`: the back-off recursed on
    # raw_tiles[:reduced] and silently dropped the remainder, so a 4-tile
    # batch that halved to 1 returned a single core. Harmless while only
    # the autotune called this (it keeps timings, not predictions), but it
    # became holes in the stem map once the streaming loop started using
    # the back-off -- see tests/test_prediction_oom_backoff.py.
    assert result == ["core-for-1"] * 4
    # 4 and 2 fail; the first tile succeeds at 1 and the working size is
    # then carried forward, so the remaining three go straight to 1.
    assert calls == [4, 2, 1, 1, 1, 1]


def test_predict_batch_adaptive_halves_batch_on_runtime_oom_message(
    monkeypatch,
):
    from utils import Prediction as Pred

    calls = []

    def fake_core(raw_tiles, raw_masks, model, config):
        calls.append(len(raw_tiles))
        if len(raw_tiles) > 1:
            raise RuntimeError("CUDA error: out of memory")
        return ["ok"]

    monkeypatch.setattr(Pred, "_predict_batch_core", fake_core)

    result, used = Pred._predict_batch_adaptive(
        ["t"] * 4, ["m"] * 4, object(), object(), 4)

    assert used == 1
    assert result == ["ok"] * 4     # every tile, not just the first slice
    assert calls == [4, 2, 1, 1, 1, 1]


def test_predict_batch_adaptive_halves_on_arena_alloc_failure(monkeypatch):
    """onnxruntime's CUDA BFC-arena OOM ("Failed to allocate memory for
    requested buffer of size N") carries neither 'oom' nor 'out of memory',
    so it slipped past the back-off and aborted the run. It must now halve
    like any other OOM (issue #40)."""
    from utils import Prediction as Pred

    calls = []

    def fake_core(raw_tiles, raw_masks, model, config):
        calls.append(len(raw_tiles))
        if len(raw_tiles) > 1:
            raise RuntimeError(
                "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status "
                "code returned while running Conv node. bfc_arena.cc "
                "AllocateRawInternal Failed to allocate memory for requested "
                "buffer of size 604127488")
        return ["ok"]

    monkeypatch.setattr(Pred, "_predict_batch_core", fake_core)

    result, used = Pred._predict_batch_adaptive(
        ["t"] * 4, ["m"] * 4, object(), object(), 4)

    assert used == 1
    assert result == ["ok"] * 4     # every tile, not just the first slice
    assert calls == [4, 2, 1, 1, 1, 1]


def test_oom_detection_catches_arena_failure_but_not_cudnn_failure():
    """The normalizer must recognize the arena allocation failure (#40) yet
    NOT misclassify a cuDNN execution failure (issue #24) as OOM -- that would
    send the batch loop halving to 1 and still fail, hiding the real cause."""
    from utils import onnx_runtime as ort_mod

    assert ort_mod._looks_like_oom(
        "Failed to allocate memory for requested buffer of size 604127488")
    assert ort_mod._looks_like_oom("CUDA error: out of memory")
    assert not ort_mod._looks_like_oom(
        "CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED")
    assert not ort_mod._looks_like_oom("some unrelated runtime error")


def test_predict_batch_adaptive_reraises_non_oom_runtime_error(monkeypatch):
    from utils import Prediction as Pred

    def fake_core(raw_tiles, raw_masks, model, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(Pred, "_predict_batch_core", fake_core)

    with pytest.raises(RuntimeError, match="boom"):
        Pred._predict_batch_adaptive(
            ["t"] * 4, ["m"] * 4, object(), object(), 4)


def test_predict_batch_adaptive_reraises_oom_at_batch_size_one(monkeypatch):
    """Can't halve below 1 -- an OOM that persists at batch_size=1 must
    still propagate instead of looping forever."""
    from utils import Prediction as Pred

    def fake_core(raw_tiles, raw_masks, model, config):
        raise MemoryError("Unable to allocate")

    monkeypatch.setattr(Pred, "_predict_batch_core", fake_core)

    with pytest.raises(MemoryError):
        Pred._predict_batch_adaptive(
            ["t"], ["m"], object(), object(), 1)


# --- Group 4: TensorFlow-free import ---------------------------------------

def test_prediction_and_predictworkers_import_without_tensorflow():
    """Poison sys.modules["tensorflow"] the way test suites conventionally
    block an accidental import (`import tensorflow` raises ImportError
    immediately), force a fresh (re)import of both modules, and confirm
    neither needs TensorFlow -- even though the conda env running this
    test suite has TensorFlow installed."""
    names_to_clear = [
        name for name in sys.modules
        if name == "tensorflow" or name.startswith("tensorflow.")
        or name in ("utils.Prediction", "utils.PredictWorkers")
    ]
    saved = {name: sys.modules.pop(name) for name in names_to_clear}
    sys.modules["tensorflow"] = None
    try:
        import utils.Prediction  # noqa: F401
        import utils.PredictWorkers  # noqa: F401

        sys.modules.pop("tensorflow", None)
        assert "tensorflow" not in sys.modules
    finally:
        sys.modules.pop("utils.Prediction", None)
        sys.modules.pop("utils.PredictWorkers", None)
        sys.modules.update(saved)


# --- Group 5: resample-in-read (TileBatchProducer out_size) ----------------

def _write_synthetic_geotiff(path, width=700, height=700, seed=0):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    data = rng.integers(1, 255, size=(3, height, width), dtype=np.uint8)
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 3,
        "dtype": "uint8", "crs": rasterio.crs.CRS.from_epsg(32633),
        "transform": from_origin(400000.0, 5900000.0, 0.05, 0.05),
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data)
    return str(path)


class _FakeConfig:
    img_height = 512
    img_width = 512


def _run_producer(uav_path, jobs, out_size):
    from utils.Prediction import TileBatchProducer

    q = queue.Queue()
    producer = TileBatchProducer(
        uav_path=uav_path, chunk_size=max(1, len(jobs)), jobs=jobs,
        n_channels=3, out_queue=q, out_size=out_size,
    )
    producer.run()  # synchronous call -- no thread needed for the test
    assert producer.error is None

    items = []
    while True:
        payload = q.get_nowait()
        if payload.get('producer_done'):
            break
        items.extend(payload['items'])
    return items


def test_tile_batch_producer_resamples_tiles_to_model_grid(tmp_path):
    uav_path = _write_synthetic_geotiff(tmp_path / "ortho.tif")

    job = {'src_col': 0, 'src_row': 0, 'src_width': 300, 'src_height': 300,
           'tile_index': 0, 'dst_row': 0, 'dst_col': 0}
    items = _run_producer(uav_path, [job], out_size=(512, 512))

    assert len(items) == 1
    _, tile, valid_mask = items[0]
    assert tile.shape == (512, 512, 3)
    assert valid_mask.shape == (512, 512)
    # A window fully inside the raster has no boundless padding: every
    # resampled pixel is valid.
    assert valid_mask.all()


def test_tile_batch_producer_native_read_unchanged_without_out_size(
    tmp_path,
):
    """out_size=None (the default) must keep reading at native resolution --
    proving the resample-in-read branch is opt-in, not a behavior change for
    any other caller of TileBatchProducer."""
    uav_path = _write_synthetic_geotiff(tmp_path / "ortho.tif")

    job = {'src_col': 0, 'src_row': 0, 'src_width': 300, 'src_height': 300,
           'tile_index': 0, 'dst_row': 0, 'dst_col': 0}
    items = _run_producer(uav_path, [job], out_size=None)

    assert len(items) == 1
    _, tile, valid_mask = items[0]
    assert tile.shape == (300, 300, 3)
    assert valid_mask.shape == (300, 300)


def test_tile_batch_producer_bypasses_skimage_resize(tmp_path, monkeypatch):
    from utils import Prediction as Pred

    uav_path = _write_synthetic_geotiff(tmp_path / "ortho.tif")
    job = {'src_col': 0, 'src_row': 0, 'src_width': 300, 'src_height': 300,
           'tile_index': 0, 'dst_row': 0, 'dst_col': 0}
    items = _run_producer(uav_path, [job], out_size=(512, 512))
    _, tile, valid_mask = items[0]

    monkeypatch.setenv("WINMOL_BENCH_READ", "overview")

    def failing_resize(*args, **kwargs):
        raise AssertionError(
            "skimage resize must not run: tiles already arrive on the "
            "model grid via the GDAL read"
        )

    monkeypatch.setattr(Pred, "resize", failing_resize)

    tile_batch, mask_resized = Pred._prepare_inference_batch(
        [tile], [valid_mask], _FakeConfig())

    assert tile_batch.shape == (1, 512, 512, 3)
    assert tile_batch.dtype == np.float32
    assert mask_resized.shape == (1, 512, 512, 1)


def test_tile_batch_producer_clipped_edge_window_still_yields_model_grid(
    tmp_path,
):
    """A window that runs past the raster boundary (boundless + fill_value=0
    padding) must still resample to exactly out_size -- this is where a
    ratio/clipping bug would silently shrink or distort the output tile."""
    width = height = 700
    uav_path = _write_synthetic_geotiff(tmp_path / "ortho.tif", width, height)

    # Window starts inside the raster but extends 300px past both edges
    # (raster is 700x700; the window covers native cols/rows 600..900).
    job = {'src_col': 600, 'src_row': 600, 'src_width': 300,
           'src_height': 300, 'tile_index': 0, 'dst_row': 0, 'dst_col': 0}
    items = _run_producer(uav_path, [job], out_size=(512, 512))

    assert len(items) == 1
    _, tile, valid_mask = items[0]
    # The clipped window still resamples to exactly the model grid -- not a
    # smaller or distorted shape.
    assert tile.shape == (512, 512, 3)
    assert valid_mask.shape == (512, 512)
    # Top-left corner is deep inside real data (native px 600, 600); the
    # bottom-right corner is deep inside the boundless fill_value=0 padding
    # (native px ~900, 900, well past the 700x700 raster).
    assert valid_mask[0, 0]
    assert not valid_mask[-1, -1]
    assert np.array_equal(tile[-1, -1], [0, 0, 0])
