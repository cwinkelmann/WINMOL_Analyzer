"""Mask input: run the downstream stages on an existing stem mask
(hand-drawn or predicted) from the batch CLI and the plugin, instead of
predicting one from an orthophoto."""

import json
import os
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import winmol_batch
import winmol_from_labels as cli
from plugin_utils import mask_input as MI
from test_manual_labels import CRS, X0, Y0, write_scene
from utils import ManualLabels as ML


def _layer_stats(gpkg):
    stems = gpd.read_file(gpkg, layer="stems", engine="pyogrio")
    return len(stems), round(float(stems.length.sum()), 3)


# --- refusing masks that are not in metres ---------------------------------


def test_geographic_raster_mask_is_refused(tmp_path):
    path = tmp_path / "mask_4326.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=20,
        height=20,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(13.2, 52.6, 1e-5, 1e-5),
    ) as dst:
        dst.write(np.ones((1, 20, 20), np.uint8))
    with pytest.raises(ValueError, match="projected"):
        ML.load_mask(str(path), pixel_size=0.03)


def test_geographic_polygons_without_reference_are_refused(tmp_path):
    path = tmp_path / "mask_4326.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(13.2, 52.6, 13.2001, 52.6001)], crs="EPSG:4326"
    ).to_file(path, engine="pyogrio")
    with pytest.raises(ValueError, match="projected"):
        ML.load_mask(str(path), pixel_size=0.03)


def test_raster_mask_with_reference_is_refused(tmp_path):
    # A raster mask keeps its own grid; accepting --reference and then
    # ignoring it would yield metrics on a grid the user did not ask for.
    path = tmp_path / "mask.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=20,
        height=20,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=from_origin(X0, Y0, 0.05, 0.05),
    ) as dst:
        dst.write(np.ones((1, 20, 20), np.uint8))
    with pytest.raises(ValueError, match="reference"):
        ML.load_mask(str(path), reference=str(tmp_path / "ref.tif"))


def test_geographic_polygons_on_a_projected_reference_are_reprojected(tmp_path):
    poly = gpd.GeoDataFrame(
        geometry=[box(X0, Y0, X0 + 2, Y0 + 1)], crs=CRS
    ).to_crs("EPSG:4326")
    path = tmp_path / "mask_4326.gpkg"
    poly.to_file(path, engine="pyogrio")
    ref = tmp_path / "ref.tif"
    with rasterio.open(
        ref,
        "w",
        driver="GTiff",
        width=200,
        height=100,
        count=1,
        dtype="uint8",
        crs=CRS,
        transform=from_origin(X0 - 1, Y0 + 1.5, 0.02, 0.02),
    ) as dst:
        dst.write(np.zeros((1, 100, 200), np.uint8))
    mask, grid = ML.load_mask(str(path), reference=str(ref), pixel_size=0.02)
    assert str(grid["crs"]) == CRS
    assert mask.sum() * 0.02**2 == pytest.approx(2.0, rel=0.05)


# --- winmol_from_labels CLI ------------------------------------------------


def test_importing_the_cli_does_not_reexec_the_process():
    # The PYTHONHASHSEED pin must only re-exec when run as a script; on
    # import it would restart the importing process (e.g. pytest).
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    out = subprocess.run(
        [sys.executable, "-c", "import winmol_from_labels; print('ok')"],
        cwd=os.path.dirname(os.path.abspath(cli.__file__)),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.stdout.strip() == "ok"


def test_lines_route_applies_env_config_overrides(tmp_path, monkeypatch):
    mask, lines = write_scene(tmp_path)
    monkeypatch.setenv(
        "WINMOL_CONFIG_OVERRIDES_JSON", '{"measuring_point_spacing_m": 5.0}'
    )
    cli.main(
        ["--mask", mask, "--lines", lines, "--out", str(tmp_path / "o" / "m")]
    )
    nodes = gpd.read_file(
        tmp_path / "o" / "m.gpkg", layer="nodes", engine="pyogrio"
    )
    assert len(nodes) == 5  # 20 m line at 5 m spacing, both ends kept


def test_dotted_output_prefix_is_kept(tmp_path):
    mask, lines = write_scene(tmp_path)
    cli.main(
        ["--mask", mask, "--lines", lines, "--out", str(tmp_path / "plot.v2")]
    )
    assert (tmp_path / "plot.v2.gpkg").exists()
    assert (tmp_path / "plot.v2_summary.json").exists()


# --- mask-only route (no centre lines) -------------------------------------


def test_mask_only_mode_writes_all_layers_and_a_summary(tmp_path):
    mask, _ = write_scene(tmp_path)
    cli.main(["--mask", mask, "--out", str(tmp_path / "out" / "m")])
    summary = json.loads((tmp_path / "out" / "m_summary.json").read_text())
    assert summary["mode"] == "mask_only"
    assert summary["stems"] >= 1
    nodes = gpd.read_file(
        tmp_path / "out" / "m.gpkg", layer="nodes", engine="pyogrio"
    )
    assert len(nodes) > 0


# --- batch CLI -------------------------------------------------------------


def test_batch_mask_mode_matches_winmol_from_labels(tmp_path):
    mask, lines = write_scene(tmp_path)
    cli.main(
        [
            "--mask",
            mask,
            "--lines",
            lines,
            "--out",
            str(tmp_path / "ref" / "mask"),
        ]
    )
    code = winmol_batch.main(
        ["--mask", mask, "--lines", lines, "--output", str(tmp_path / "out")]
    )
    assert code == 0
    assert _layer_stats(tmp_path / "out" / "mask.gpkg") == _layer_stats(
        tmp_path / "ref" / "mask.gpkg"
    )
    assert (tmp_path / "out" / "mask_summary.json").exists()


def test_batch_forwards_pixel_size(tmp_path):
    mask, lines = write_scene(tmp_path)
    code = winmol_batch.main(
        [
            "--mask",
            mask,
            "--lines",
            lines,
            "--pixel-size",
            "0.05",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    summary = json.loads(
        (tmp_path / "out" / "mask_summary.json").read_text()
    )
    assert summary["pixel_size_m"] == pytest.approx(0.05)


def test_batch_refuses_model_and_mask_together(tmp_path, capsys):
    mask, _ = write_scene(tmp_path)
    with pytest.raises(SystemExit):
        winmol_batch.main(["Beech", "--mask", mask])
    assert "either a model or --mask" in capsys.readouterr().err


def test_batch_without_model_or_mask_names_both(capsys):
    with pytest.raises(SystemExit):
        winmol_batch.main([])
    assert "--mask" in capsys.readouterr().err


def test_batch_lines_without_mask_is_refused(tmp_path, capsys):
    _, lines = write_scene(tmp_path)
    with pytest.raises(SystemExit):
        winmol_batch.main(["Beech", "--lines", lines])
    assert "--lines needs --mask" in capsys.readouterr().err


# --- plugin helper (Qt-free) -----------------------------------------------


def test_mask_command_with_lines():
    cmd = MI.mask_command(
        "py",
        "/plug",
        "/d/m.tif",
        "/o/x",
        lines="/d/l.gpkg",
        lines_layer="centrelines",
        id_field="ID",
    )
    assert cmd == [
        "py",
        "-u",
        os.path.join("/plug", MI.FROM_LABELS),
        "--mask",
        "/d/m.tif",
        "--out",
        "/o/x",
        "--lines",
        "/d/l.gpkg",
        "--lines-layer",
        "centrelines",
        "--id-field",
        "ID",
    ]


def test_mask_command_polygon_layer_without_lines():
    cmd = MI.mask_command(
        "py", "/plug", "/d/m.gpkg", "/o/x", mask_layer="stems"
    )
    assert cmd == [
        "py",
        "-u",
        os.path.join("/plug", MI.FROM_LABELS),
        "--mask",
        "/d/m.gpkg",
        "--out",
        "/o/x",
        "--mask-layer",
        "stems",
    ]


def test_layer_source_is_split_into_path_and_layer():
    assert MI.split_layer_source("/d/a.gpkg|layername=stems") == (
        "/d/a.gpkg",
        "stems",
    )
    assert MI.split_layer_source("/d/a.tif") == ("/d/a.tif", None)
    assert MI.split_layer_source("/d/a.gpkg|layerid=0") == ("/d/a.gpkg", None)


def test_validate_mask_input(tmp_path):
    m = tmp_path / "m.tif"
    m.write_bytes(b"x")
    assert (
        MI.validate_mask_input("")
        == "No stem mask selected. Pick a raster or polygon layer or file."
    )
    assert "not found" in MI.validate_mask_input(str(tmp_path / "nope.tif"))
    assert "Centre-line file was not found" in MI.validate_mask_input(
        str(m), str(tmp_path / "nope.gpkg")
    )
    assert MI.validate_mask_input(str(m)) is None
