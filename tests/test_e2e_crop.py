"""End-to-end golden test: standalone run_pipeline on the fixture crop.

Slow (TF + model + full chain). The comparison is canonical/order-insensitive
because run_pipeline's internal quantification may run pooled (callback
append order varies), and the fresh prediction may differ from the golden
stem map at isolated pixels — so stem counts and totals get small tolerances
rather than exact pinning. The exact pinning lives in the per-stage tests.

Note: run_pipeline does NOT forward config to quantify_stems (review A-18);
this test passes the snapshot config for the stages that do receive it.
"""

import importlib.util
import os
import sys

import pytest

import helpers

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    REPO_ROOT, "standalone", "model_onnx", "General.onnx")

os.environ.setdefault("WINMOL_ONNX_FORCE_CPU", "1")   # match the fixtures

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def standalone_mod():
    pytest.importorskip("onnxruntime")
    path = os.path.join(REPO_ROOT, "standalone", "WINMOL_Analyzer.py")
    spec = importlib.util.spec_from_file_location(
        "winmol_standalone_e2e", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["winmol_standalone_e2e"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_pipeline_end_to_end(standalone_mod, crop_input_path,
                                 pipeline_config, fixtures_dir, tmp_path):
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"model not found: {MODEL_PATH}")

    pred_dir = str(tmp_path / "pred")
    out_dir = str(tmp_path / "out") + os.sep  # A-18b: bare string concat
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    result = standalone_mod.run_pipeline(
        MODEL_PATH, crop_input_path, pred_dir, out_dir,
        config=pipeline_config)

    assert os.path.exists(result["stem_map_path"])
    gpkg = result["gpkg_path"]
    assert os.path.exists(gpkg)
    assert sorted(helpers.gpkg_layers(gpkg)) == ["nodes", "stems", "vectors"]

    actual = helpers.gpkg_stems_canonical(gpkg)
    golden = helpers.gpkg_stems_canonical(
        os.path.join(fixtures_dir, "golden_stems.gpkg"))

    # Prediction pixel flips can split/merge a stem or two at the margin —
    # allow a small count band, but totals must stay close.
    assert abs(len(actual) - len(golden)) <= max(2, int(0.05 * len(golden))), (
        f"stem count {len(actual)} vs golden {len(golden)}")

    total_len_a = sum(r["length"] for r in actual)
    total_len_g = sum(r["length"] for r in golden)
    total_vol_a = sum(r["volume"] for r in actual)
    total_vol_g = sum(r["volume"] for r in golden)
    assert total_len_a == pytest.approx(total_len_g, rel=0.05), (
        f"total length {total_len_a:.1f} vs golden {total_len_g:.1f}")
    assert total_vol_a == pytest.approx(total_vol_g, rel=0.10), (
        f"total volume {total_vol_a:.2f} vs golden {total_vol_g:.2f}")
