"""End-to-end compute contract for the TF-free pipeline: a real
``winmol_run.py <model.onnx> <img.tif> <stem.tif> <prefix> Stems`` child
process must succeed with TensorFlow imports hard-blocked, using only the
vendored ONNX runtime. The model is a tiny on-the-fly 512x512 segmenter
(random weights, so no stem-count assertions) and the image a synthetic
georeferenced RGB GeoTIFF."""
import os
import subprocess
import sys

import pytest

from conftest import build_test_geotiff, build_tiny_unet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TF_BLOCKER = '''\
import sys


class _TFBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ImportError("TensorFlow blocked by contract test")
        return None


sys.meta_path.insert(0, _TFBlocker())
'''


def test_stems_run_end_to_end_without_tensorflow(tmp_path):
    blocker_dir = tmp_path / "tf_blocker"
    blocker_dir.mkdir()
    (blocker_dir / "sitecustomize.py").write_text(TF_BLOCKER)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(blocker_dir) + os.pathsep + \
        env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = "0"
    env["WINMOL_ONNX_FORCE_CPU"] = "1"
    env["WINMOL_CONFIG_OVERRIDES_JSON"] = \
        '{"prediction_batch_autotune": false}'

    # The blocker itself must work: importing TF in the child errors out.
    probe = subprocess.run(
        [sys.executable, "-c", "import tensorflow"],
        capture_output=True, text=True, env=env, timeout=120)
    assert probe.returncode != 0
    assert "TensorFlow blocked by contract test" in probe.stderr

    model = build_tiny_unet(tmp_path / "segmenter.onnx")
    image = build_test_geotiff(tmp_path / "ortho.tif")
    stem_map = tmp_path / "out" / "stem_map.tif"

    proc = subprocess.run(
        [sys.executable, "-u", "winmol_run.py", model, image,
         str(stem_map), str(tmp_path / "out" / "trees"), "Stems"],
        capture_output=True, text=True, env=env, cwd=REPO, timeout=300)
    assert proc.returncode == 0, (
        f"winmol_run.py failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    assert "CPUExecutionProvider" in proc.stdout

    rasterio = pytest.importorskip("rasterio")
    assert stem_map.exists()
    with rasterio.open(str(stem_map)) as src:
        assert src.crs.to_epsg() == 32633
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.width > 0 and src.height > 0
        # Same origin as the input: georeferencing survived the pipeline.
        assert src.transform.c == 400000.0
        assert src.transform.f == 5900000.0
