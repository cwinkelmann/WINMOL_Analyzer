"""Downstream metrics from manual labels, on small scenes with a known answer."""
import json
import math

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString, MultiPolygon, box

import winmol_from_labels as cli
from classes.Config import Config
from utils import ManualLabels as ML

CRS = "EPSG:32633"
PIXEL = 0.02
X0, Y0 = 380000.0, 5828000.0


def write_scene(tmp_path, lines=None, field="ID", ids=None):
    """A 0.4 m wide, 20 m long horizontal stem, and its centre line."""
    polygons = gpd.GeoDataFrame(geometry=[box(X0, Y0 - 0.2, X0 + 20, Y0 + 0.2)], crs=CRS)
    mask_path = tmp_path / "mask.gpkg"
    polygons.to_file(mask_path, engine="pyogrio")
    lines = lines or [LineString([(X0, Y0), (X0 + 20, Y0)])]
    lines_gdf = gpd.GeoDataFrame({field: ids or list(range(1, len(lines) + 1))},
                                 geometry=lines, crs=CRS)
    lines_path = tmp_path / "lines.gpkg"
    lines_gdf.to_file(lines_path, engine="pyogrio")
    return str(mask_path), str(lines_path)


def test_rasterised_mask_has_the_polygon_area():
    polygons = gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + 2, Y0 + 1)], crs=CRS)
    grid = ML.grid_from_bounds(polygons.total_bounds, PIXEL, polygons.crs)
    mask = ML.rasterize_polygons(polygons, grid)
    assert mask.sum() * PIXEL ** 2 == pytest.approx(2.0, rel=0.02)
    assert mask.dtype == np.uint8


def test_resampling_uses_the_measuring_spacing_and_keeps_the_ends():
    line = LineString([(0, 0), (10, 0), (10, 10)])
    path = ML.resample_line(line, 0.5)
    assert len(path.coords) == 41
    assert path.coords[0] == (0.0, 0.0)
    assert path.coords[-1] == (10.0, 10.0)
    assert path.length == pytest.approx(20.0)


def test_touching_lines_with_one_id_become_one_stem():
    lines = gpd.GeoDataFrame({"ID": [7, 7]}, geometry=[
        LineString([(0, 0), (5, 0)]), LineString([(5, 0), (10, 0)])], crs=CRS)
    stems = ML.stems_from_lines(lines, "ID", 0.5)
    assert len(stems) == 1
    assert stems[0].source_id == "7"
    assert stems[0].path.length == pytest.approx(10.0)


def test_separated_lines_with_one_id_stay_parts_of_one_stem():
    lines = gpd.GeoDataFrame({"ID": [7, 7]}, geometry=[
        LineString([(0, 0), (5, 0)]), LineString([(8, 0), (12, 0)])], crs=CRS)
    stems = ML.stems_from_lines(lines, "ID", 0.5)
    assert [s.source_id for s in stems] == ["7", "7"]
    assert [s.part for s in stems] == [0, 1]


def test_crossing_lines_with_one_id_are_not_cut():
    lines = gpd.GeoDataFrame({"ID": [7, 7]}, geometry=[
        LineString([(0, 0), (10, 0)]), LineString([(5, -5), (5, 5)])], crs=CRS)
    stems = ML.stems_from_lines(lines, "ID", 0.5)
    assert sorted(round(s.path.length, 6) for s in stems) == [10.0, 10.0]


def test_a_branch_touching_mid_line_does_not_cut_the_stem():
    lines = gpd.GeoDataFrame({"ID": [7, 7]}, geometry=[
        LineString([(0, 0), (10, 0)]), LineString([(5, 0), (5, 3)])], crs=CRS)
    stems = ML.stems_from_lines(lines, "ID", 0.5)
    assert sorted(round(s.path.length, 6) for s in stems) == [3.0, 10.0]


def test_three_lines_meeting_at_an_endpoint_stay_separate():
    lines = gpd.GeoDataFrame({"ID": [7, 7, 7]}, geometry=[
        LineString([(0, 0), (5, 0)]), LineString([(5, 0), (10, 0)]), LineString([(5, 0), (5, 3)])],
        crs=CRS)
    assert len(ML.stems_from_lines(lines, "ID", 0.5)) == 3


def test_a_line_clipped_into_two_pieces_stays_two_parts():
    lines = gpd.GeoDataFrame({"ID": [7]}, geometry=[LineString([(0, 0), (10, 0)])], crs=CRS)
    aoi = MultiPolygon([box(0, -1, 3, 1), box(6, -1, 10, 1)])
    stems = ML.stems_from_lines(lines, "ID", 0.5, aoi=aoi)
    assert sorted(round(s.path.length, 6) for s in stems) == [3.0, 4.0]


def test_null_ids_are_never_merged():
    lines = gpd.GeoDataFrame({"ID": [None, None]}, geometry=[
        LineString([(0, 0), (5, 0)]), LineString([(5, 0), (10, 0)])], crs=CRS)
    stems = ML.stems_from_lines(lines, "ID", 0.5)
    assert len({s.source_id for s in stems}) == 2


def test_missing_id_field_is_an_error_not_a_renumbering():
    lines = gpd.GeoDataFrame({"Tree ID": [1]}, geometry=[LineString([(0, 0), (5, 0)])], crs=CRS)
    with pytest.raises(ValueError, match="Tree ID"):
        ML.stems_from_lines(lines, "ID", 0.5)


def test_aoi_clips_the_lines():
    lines = gpd.GeoDataFrame({"ID": [1]}, geometry=[LineString([(0, 0), (10, 0)])], crs=CRS)
    stems = ML.stems_from_lines(lines, "ID", 0.5, aoi=box(0, -1, 4, 1))
    assert stems[0].path.length == pytest.approx(4.0)


def test_lines_give_the_true_diameter_length_and_volume(tmp_path):
    mask, lines = write_scene(tmp_path)
    out = tmp_path / "out" / "trees"
    cli.main(["--mask", mask, "--lines", lines, "--pixel-size", str(PIXEL), "--out", str(out)])

    stems = ML.read_layer(f"{out}.gpkg", "stems")
    assert list(stems["source_id"]) == ["1"]
    assert stems["length"].iloc[0] == pytest.approx(20.0, abs=0.01)
    assert stems["volume"].iloc[0] == pytest.approx(math.pi * 0.2 ** 2 * 20, rel=0.03)

    nodes = ML.read_layer(f"{out}.gpkg", "nodes")
    assert len(nodes) == 41

    summary = json.load(open(f"{out}_summary.json"))
    assert summary["mode"] == "mask_and_lines"
    assert summary["stems"] == 1
    assert summary["diameter_median_m"] == pytest.approx(0.4, abs=PIXEL)
    assert summary["nodes"] == 41


def test_a_line_off_the_mask_reports_zero_diameters(tmp_path):
    mask, lines = write_scene(tmp_path, lines=[LineString([(X0, Y0 + 5), (X0 + 20, Y0 + 5)])])
    grid_ref = tmp_path / "ref.tif"
    grid = ML.grid_from_bounds((X0, Y0 - 1, X0 + 20, Y0 + 6), PIXEL, CRS)
    with rasterio.open(grid_ref, "w", **ML.mask_profile(
            np.zeros((grid["height"], grid["width"]), dtype=np.uint8), grid)) as dst:
        dst.write(np.zeros((grid["height"], grid["width"]), dtype=np.uint8), 1)
    out = tmp_path / "trees"
    cli.main(["--mask", mask, "--lines", lines, "--reference", str(grid_ref), "--out", str(out)])
    summary = json.load(open(f"{out}_summary.json"))
    assert summary["nodes_zero_diameter"] == summary["nodes"]
    assert summary["volume_m3"] == 0.0
    assert summary["diameter_median_m"] is None


def test_a_raster_mask_is_accepted(tmp_path):
    _, lines = write_scene(tmp_path)
    polygons = gpd.GeoDataFrame(geometry=[box(X0, Y0 - 0.2, X0 + 20, Y0 + 0.2)], crs=CRS)
    grid = ML.grid_from_bounds(polygons.total_bounds, PIXEL, polygons.crs)
    tif = ML.write_mask(ML.rasterize_polygons(polygons, grid), grid, str(tmp_path / "m.tif"))
    out = tmp_path / "trees"
    cli.main(["--mask", tif, "--lines", lines, "--out", str(out)])
    assert json.load(open(f"{out}_summary.json"))["diameter_median_m"] == pytest.approx(0.4, abs=PIXEL)


def test_custom_id_field(tmp_path):
    mask, lines = write_scene(tmp_path, field="Tree ID", ids=[42])
    out = tmp_path / "trees"
    cli.main(["--mask", mask, "--lines", lines, "--id-field", "Tree ID",
              "--pixel-size", str(PIXEL), "--out", str(out)])
    assert list(ML.read_layer(f"{out}.gpkg", "stems")["source_id"]) == ["42"]


def test_default_pixel_size_is_the_stem_map_scale():
    config = Config()
    assert config.tile_size / config.img_width == pytest.approx(0.0293, abs=1e-4)
