"""Mask input: run the downstream stages on an existing stem mask (hand-drawn or predicted) from the batch CLI and
the plugin, instead of predicting one from an orthophoto."""
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


# --- refusing masks that are not in metres ---------------------------------------------------------------------

def test_geographic_raster_mask_is_refused(tmp_path):
    path = tmp_path / "mask_4326.tif"
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(13.2, 52.6, 1e-5, 1e-5)) as dst:
        dst.write(np.ones((1, 20, 20), np.uint8))
    with pytest.raises(ValueError, match="projected"):
        ML.load_mask(str(path), pixel_size=0.03)


def test_geographic_polygons_without_reference_are_refused(tmp_path):
    path = tmp_path / "mask_4326.gpkg"
    gpd.GeoDataFrame(geometry=[box(13.2, 52.6, 13.2001, 52.6001)], crs="EPSG:4326").to_file(path, engine="pyogrio")
    with pytest.raises(ValueError, match="projected"):
        ML.load_mask(str(path), pixel_size=0.03)


def test_geographic_polygons_on_a_projected_reference_are_reprojected(tmp_path):
    poly = gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + 2, Y0 + 1)], crs=CRS).to_crs("EPSG:4326")
    path = tmp_path / "mask_4326.gpkg"
    poly.to_file(path, engine="pyogrio")
    ref = tmp_path / "ref.tif"
    with rasterio.open(ref, "w", driver="GTiff", width=200, height=100, count=1, dtype="uint8",
                       crs=CRS, transform=from_origin(X0 - 1, Y0 + 1.5, 0.02, 0.02)) as dst:
        dst.write(np.zeros((1, 100, 200), np.uint8))
    mask, grid = ML.load_mask(str(path), reference=str(ref), pixel_size=0.02)
    assert str(grid["crs"]) == CRS
    assert mask.sum() * 0.02 ** 2 == pytest.approx(2.0, rel=0.05)


# --- batch CLI ----------------------------------------------------------------------------------------------------

def test_batch_mask_mode_matches_winmol_from_labels(tmp_path):
    mask, lines = write_scene(tmp_path)
    cli.main(["--mask", mask, "--lines", lines, "--out", str(tmp_path / "ref" / "mask")])
    code = winmol_batch.main(["--mask", mask, "--lines", lines, "--output", str(tmp_path / "out")])
    assert code == 0
    assert _layer_stats(tmp_path / "out" / "mask.gpkg") == _layer_stats(tmp_path / "ref" / "mask.gpkg")
    assert (tmp_path / "out" / "mask_summary.json").exists()


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


# --- plugin helper (Qt-free) --------------------------------------------------------------------------------------

def test_mask_command_with_lines():
    cmd = MI.mask_command("py", "/plug", "/d/m.tif", "/o/x", lines="/d/l.gpkg", lines_layer="centrelines",
                          id_field="ID")
    assert cmd == ["py", "-u", "/plug/winmol_from_labels.py", "--mask", "/d/m.tif", "--out", "/o/x",
                   "--lines", "/d/l.gpkg", "--lines-layer", "centrelines", "--id-field", "ID"]


def test_mask_command_polygon_layer_without_lines():
    cmd = MI.mask_command("py", "/plug", "/d/m.gpkg", "/o/x", mask_layer="stems")
    assert cmd == ["py", "-u", "/plug/winmol_from_labels.py", "--mask", "/d/m.gpkg", "--out", "/o/x",
                   "--mask-layer", "stems"]


def test_layer_source_is_split_into_path_and_layer():
    assert MI.split_layer_source("/d/a.gpkg|layername=stems") == ("/d/a.gpkg", "stems")
    assert MI.split_layer_source("/d/a.tif") == ("/d/a.tif", None)
    assert MI.split_layer_source("/d/a.gpkg|layerid=0") == ("/d/a.gpkg", None)


def test_validate_mask_input(tmp_path):
    m = tmp_path / "m.tif"
    m.write_bytes(b"x")
    assert MI.validate_mask_input("") == "No stem mask selected. Pick a raster or polygon layer or file."
    assert "not found" in MI.validate_mask_input(str(tmp_path / "nope.tif"))
    assert "Centre-line file was not found" in MI.validate_mask_input(str(m), str(tmp_path / "nope.gpkg"))
    assert MI.validate_mask_input(str(m)) is None
