"""Golden inference test: crop_input.tif + ONNX U-Net -> stem map.

Marked slow (loads onnxruntime + a ~124 MB model and predicts the whole crop).
No TensorFlow. Forces CPU onnxruntime so it matches the CPU-generated fixtures;
comparison is agreement-based (isolated boundary-pixel flips are expected across
onnxruntime builds). Everything downstream is pinned exactly because the stage
tests start from the SAVED stem map, not a fresh prediction.
"""

import json
import os

# Deterministic + CPU onnxruntime, to match how the fixtures were generated.
os.environ.setdefault("WINMOL_ONNX_FORCE_CPU", "1")

import numpy as np   # noqa: E402
import pytest        # noqa: E402

import helpers       # noqa: E402

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "standalone", "model_onnx", "General.onnx")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"ONNX model not found: {MODEL_PATH} "
                    f"(run scripts/convert_models_to_onnx.py)")
    pytest.importorskip("onnxruntime")
    from utils import IO
    return IO.load_model_from_path(MODEL_PATH)


def test_model_identity_matches_fixture_generation(fixtures_dir):
    """The committed fixtures record which model produced them."""
    with open(os.path.join(fixtures_dir, "manifest.json")) as f:
        manifest = json.load(f)
    if not os.path.exists(MODEL_PATH):
        pytest.skip("ONNX model file absent")
    from generate_fixtures import sha16
    assert sha16(MODEL_PATH) == manifest["model"]["sha256_16"], (
        "local ONNX model differs from the one that generated the fixtures — "
        "regenerate fixtures or restore the model")


def test_prediction_agrees_with_golden_stem_map(
        model, crop_input_path, stem_map, pipeline_config):
    from utils import IO
    from utils import Prediction as Pred

    golden_pred, golden_profile = stem_map
    img, profile = IO.load_orthomosaic(crop_input_path, pipeline_config)
    pred, out_profile = Pred.predict_with_resampling_per_tile(
        img, dict(profile), model, pipeline_config)
    pred = pred.astype(np.uint8)

    assert pred.shape == golden_pred.shape
    agreement = helpers.raster_agreement(pred, golden_pred)
    assert agreement >= 0.999, (
        f"pixel agreement {agreement:.5f} < 0.999 vs golden stem map")

    fg_new = int((pred > 0).sum())
    fg_gold = int((golden_pred > 0).sum())
    assert abs(fg_new - fg_gold) <= max(0.005 * fg_gold, 50), (
        f"foreground count drifted: {fg_new} vs golden {fg_gold}")

    # the resampled georeferencing must be identical
    assert out_profile["transform"] == golden_profile["transform"]
