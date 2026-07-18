"""Regression test for cross-run stem-count determinism.

The vector stage's ``connect_stems`` joins stems in the set-iteration order of
string-hashed ``Part`` objects (docs/CODE_REVIEW_2.md A-4). Python salts string
hashing per process, so *without* a fixed ``PYTHONHASHSEED`` the SAME
orthomosaic yielded a slightly different number of stems on every run.
``winmol_run.py`` pins the seed (re-exec with ``PYTHONHASHSEED=0``) so results
are reproducible across processes.

This runs the real entry point three times, deliberately handing each run a
DIFFERENT incoming ``PYTHONHASHSEED`` (unset / "1" / "2"). With the guard, all
three re-exec to seed 0 and must produce byte-identical stems. If the guard is
removed, the three seeds take effect and the outputs diverge -> this test fails.

Marked slow: loads onnxruntime + a model and runs the whole tiled pipeline 3x.
"""
import hashlib
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, "standalone", "model_onnx", "General.onnx")
CROP = os.path.join(REPO, "tests", "fixtures", "crop_input.tif")

pytestmark = pytest.mark.slow


def _stem_signature(gpkg):
    """(feature count, md5 of the sorted per-stem WKB) — order-independent."""
    fiona = pytest.importorskip("fiona")
    shapely_geometry = pytest.importorskip("shapely.geometry")
    with fiona.open(gpkg) as src:
        wkb = sorted(shapely_geometry.shape(f["geometry"]).wkb_hex
                     for f in src)
    return len(wkb), hashlib.md5("".join(wkb).encode()).hexdigest()


def _run(tmp_path, tag, incoming_seed):
    env = dict(os.environ)
    if incoming_seed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = incoming_seed
    prefix = tmp_path / f"run_{tag}"
    stem_map = tmp_path / f"run_{tag}_sm.tif"
    subprocess.run(
        [sys.executable, "winmol_run.py", MODEL, CROP,
         str(stem_map), str(prefix), "Trees"],
        cwd=REPO, env=env, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gpkg = f"{prefix}.gpkg"
    assert os.path.exists(gpkg), f"run {tag} produced no GeoPackage"
    return _stem_signature(gpkg)


def test_trees_stem_count_is_deterministic_across_processes(tmp_path):
    pytest.importorskip("onnxruntime")
    if not os.path.exists(MODEL):
        pytest.skip(f"model missing: {MODEL} "
                    f"(run scripts/convert_models_to_onnx.py)")

    # Different incoming hash seeds; the guard must normalize them all to 0,
    # so the stem count + geometry must come out identical.
    sigs = {
        "unset": _run(tmp_path, "unset", None),
        "seed1": _run(tmp_path, "seed1", "1"),
        "seed2": _run(tmp_path, "seed2", "2"),
    }
    counts = {tag: count for tag, (count, _) in sigs.items()}
    assert len(set(sigs.values())) == 1, (
        f"non-deterministic stems across processes: {counts} "
        f"(is the PYTHONHASHSEED re-exec guard still at the top of "
        f"winmol_run.py?)")
