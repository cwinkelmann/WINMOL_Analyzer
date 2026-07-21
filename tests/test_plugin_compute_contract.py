"""Plugin compute-contract test: winmol_run.py (the subprocess the QGIS plugin
launches) runs end-to-end on an ONNX model with TensorFlow UNAVAILABLE, and
writes a stem map + GeoPackage.

Runs winmol_run.py in a child process whose imports of ``tensorflow`` raise —
the strongest proof that the plugin's compute path is TF-free. CI-friendly: no
venv/pip; uses the current interpreter (which in the CI image has no TF anyway).
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, "standalone", "model_onnx", "General.onnx")
CROP = os.path.join(REPO, "tests", "fixtures", "crop_input.tif")

_BLOCK_TF = (
    "import builtins\n"
    "_orig = builtins.__import__\n"
    "def _b(name, *a, **k):\n"
    "    if name == 'tensorflow' or name.startswith('tensorflow.'):\n"
    "        raise ImportError('tensorflow blocked (plugin is TF-free)')\n"
    "    return _orig(name, *a, **k)\n"
    "builtins.__import__ = _b\n"
)


def test_winmol_run_is_tf_free_and_produces_outputs(tmp_path):
    if not os.path.exists(MODEL):
        pytest.skip(f"ONNX model not found: {MODEL}")
    pytest.importorskip("onnxruntime")

    # a sitecustomize that blocks tensorflow imports in the child
    blockdir = tmp_path / "block"
    blockdir.mkdir()
    (blockdir / "sitecustomize.py").write_text(_BLOCK_TF)

    stem_map = tmp_path / "stem_map.tif"
    out_prefix = tmp_path / "out"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(blockdir) + os.pathsep + env.get("PYTHONPATH", "")
    env["WINMOL_ONNX_FORCE_CPU"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # "Nodes" is what the plugin's default selection (all three output
    # products checked) resolves to: one run, stem map + every vector layer.
    proc = subprocess.run(
        [sys.executable, "-u", "winmol_run.py", MODEL, CROP,
         str(stem_map), str(out_prefix), "Nodes"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900)

    assert proc.returncode == 0, (
        f"winmol_run failed (exit {proc.returncode}).\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}")
    # never fell back to a TF import
    assert "tensorflow blocked" not in (proc.stdout + proc.stderr)

    assert stem_map.exists(), "stem-map raster not written"
    gpkg = out_prefix.with_suffix(".gpkg")
    assert gpkg.exists(), "GeoPackage not written"

    import pyogrio
    n = len(pyogrio.read_dataframe(str(gpkg), layer="stems"))
    assert n > 0, "no stems detected"

    # All three products from a single invocation: the stem-map raster plus
    # the stems / vectors / nodes layers in one GeoPackage.
    layers = set(pyogrio.list_layers(str(gpkg))[:, 0])
    for expected in ("stems", "vectors", "nodes"):
        assert expected in layers, (
            f"layer '{expected}' missing from {gpkg}; got {sorted(layers)}")
