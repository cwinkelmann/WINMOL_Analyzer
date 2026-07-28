"""Model conformance: prove that DIFFERENT models run the pipeline the SAME
way — same loader entry point, same I/O contract, same prediction grid, same
vector chain, same export structure. Weights (and therefore detections) may
differ; pipeline behavior must not.

Models come from tests/models_manifest.json (override with
WINMOL_MODELS_MANIFEST=<path>). Each model runs in a FRESH subprocess via
tests/conformance_driver.py because mixed formats cannot coexist in one
process: a Keras-2 HDF5 needs TF_USE_LEGACY_KERAS=1 while a Keras-3 HDF5
fails under it. The parent sniffs the format and sets the env accordingly.

A side-by-side comparison report is written to
tests/conformance_report.md after the run.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(TESTS_DIR, "conformance_driver.py")
REPORT_PATH = os.path.join(TESTS_DIR, "conformance_report.md")

pytestmark = pytest.mark.slow

_RESULTS = {}  # name -> result dict, for the report
_PREDS = {}    # name -> saved stem-map path, for cross-model agreement


def _load_manifest():
    path = os.environ.get(
        "WINMOL_MODELS_MANIFEST",
        os.path.join(TESTS_DIR, "models_manifest.json"))
    if not os.path.exists(path):
        return []
    with open(path) as f:
        manifest = json.load(f)
    entries = []
    for entry in manifest.get("models", []):
        p = entry["path"]
        if not os.path.isabs(p):
            p = os.path.join(REPO_ROOT, p)
        entries.append({**entry, "path": p})
    return entries


def _keras_env_for(model_path):
    """Decide TF_USE_LEGACY_KERAS for the subprocess by model format.

    - .onnx: TF unused; value irrelevant.
    - .keras: always Keras 3 -> disable the legacy shim.
    - .hdf5/.h5: sniff the keras_version attribute; '2.x' artifacts need the
      legacy shim, '3.x' artifacts break under it.
    """
    lower = model_path.lower()
    if lower.endswith(".onnx"):
        return {"TF_USE_LEGACY_KERAS": "0"}
    if lower.endswith(".keras"):
        return {"TF_USE_LEGACY_KERAS": "0"}
    try:
        import h5py
        with h5py.File(model_path, "r") as h:
            version = h.attrs.get("keras_version", b"")
            if isinstance(version, bytes):
                version = version.decode()
    except Exception:
        version = ""
    legacy = "1" if str(version).startswith("2") else "0"
    return {"TF_USE_LEGACY_KERAS": legacy, "_keras_version": str(version)}


MODELS = _load_manifest()


@pytest.fixture(scope="module")
def preds_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("conformance_preds")


@pytest.mark.parametrize(
    "entry", MODELS, ids=[m["name"] for m in MODELS] or ["no-models"])
def test_model_conformance(entry, tmp_path, preds_dir):
    if not MODELS:
        pytest.skip("no models manifest found")
    if not entry["path"].lower().endswith(".onnx"):
        # The runtime loads only .onnx now (the legacy Keras/.hdf5 path was
        # removed). Convert with scripts/convert_models_to_onnx.py and add the
        # resulting .onnx to the manifest to conformance-test it.
        pytest.skip(
            "runtime loads only .onnx; convert .hdf5/.keras with "
            "scripts/convert_models_to_onnx.py")
    if not os.path.exists(entry["path"]):
        pytest.skip(f"model file missing: {entry['path']}")

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    keras_env = _keras_env_for(entry["path"])
    env["TF_USE_LEGACY_KERAS"] = keras_env["TF_USE_LEGACY_KERAS"]

    proc = subprocess.run(
        [sys.executable, DRIVER, entry["path"], str(tmp_path)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)

    result_path = tmp_path / "result.json"
    assert result_path.exists(), (
        f"driver produced no result (exit {proc.returncode}).\n"
        f"stderr tail:\n{proc.stderr[-2000:]}")
    with open(result_path) as f:
        result = json.load(f)
    result["keras_env"] = keras_env
    _RESULTS[entry["name"]] = result

    pred_tif = tmp_path / "pred.tif"
    if pred_tif.exists():
        import shutil
        kept = preds_dir / f"{entry['name']}.tif"
        shutil.copyfile(pred_tif, kept)
        _PREDS[entry["name"]] = str(kept)

    assert proc.returncode == 0 and result.get("ok"), (
        f"{entry['name']} failed conformance.\n"
        f"checks: {json.dumps(result.get('checks', {}), indent=1)}\n"
        f"error: {result.get('error')}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}")

    failed = [k for k, v in result["checks"].items() if v is not True]
    assert not failed, f"{entry['name']} failed checks: {failed}"


def _write_report():
    lines = [
        "# Model conformance report",
        "",
        "Same crop (`tests/fixtures/crop_input.tif`), same config snapshot,",
        "same pipeline for every model — only the model file differs.",
        "Detections legitimately differ (different weights); every",
        "structural check passed for the models listed as `ok`.",
        "",
        "| model | loader | legacy keras | ok | load s | predict s |"
        " stem px % | parts | stems | length m | volume m³ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in _RESULTS.items():
        m = r.get("metrics", {})
        lines.append(
            f"| {name} | {m.get('loader', '?')} "
            f"| {r.get('keras_env', {}).get('TF_USE_LEGACY_KERAS', '?')} "
            f"| {'✅' if r.get('ok') else '❌'} "
            f"| {m.get('load_s', '—')} | {m.get('predict_s', '—')} "
            f"| {m.get('stem_px_pct', '—')} | {m.get('n_parts', '—')} "
            f"| {m.get('n_stems', '—')} | {m.get('total_length_m', '—')} "
            f"| {m.get('total_volume_m3', '—')} |")
    lines.append("")
    lines.extend(_agreement_section())
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))


def _agreement_section():
    """Pairwise stem-map comparison: IoU of foreground + pixel agreement.

    IoU ~1.0 means two models segment the SAME stems (e.g. the same weights
    exported to two formats); low IoU means genuinely different detectors.
    """
    if len(_PREDS) < 2:
        return []
    import numpy as np
    import rasterio

    names = list(_PREDS)
    arrays = {}
    for n in names:
        with rasterio.open(_PREDS[n]) as src:
            arrays[n] = src.read(1) > 0

    lines = ["## Pairwise stem-map agreement", "",
             "| model A | model B | IoU (foreground) | pixel agreement |",
             "|---|---|---|---|"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = arrays[names[i]], arrays[names[j]]
            if a.shape != b.shape:
                lines.append(f"| {names[i]} | {names[j]} | shape mismatch |")
                continue
            inter = float(np.logical_and(a, b).sum())
            union = float(np.logical_or(a, b).sum())
            iou = inter / union if union else 1.0
            agree = float(np.mean(a == b))
            lines.append(f"| {names[i]} | {names[j]} "
                         f"| {iou:.3f} | {agree:.4f} |")
    lines.append("")
    return lines


@pytest.fixture(scope="module", autouse=True)
def _report_at_end():
    yield
    if _RESULTS:
        _write_report()
        print(f"\nconformance report -> {REPORT_PATH}")
