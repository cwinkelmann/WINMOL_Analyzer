"""Golden tests for the export layer: Stems -> GeoDataFrames -> GeoPackage.

All GPKG reads go through pyogrio; the fiona read path is broken with the
pinned geopandas 0.14 + installed pandas/fiona (review A-8).
"""

import os

import helpers
from utils import IO


def _quantified_stems(golden):
    return helpers.stems_from_canonical(golden("stage_quantified_contour"))


def test_stems_to_gdf_matches_golden(golden, stem_map):
    _, profile = stem_map
    stems = _quantified_stems(golden)
    gdf = IO.stems_to_gdf(stems, profile)
    records = golden("stage_quantified_contour")
    assert len(gdf) == len(records)
    assert {"length", "volume"}.issubset(gdf.columns)
    # totals match the quantified fixture
    import numpy as np
    assert np.isclose(gdf["length"].sum(),
                      sum(sum(r["lengths"]) for r in records), rtol=1e-6)
    assert np.isclose(gdf["volume"].sum(),
                      sum(sum(r["volumes"]) for r in records), rtol=1e-6)


def test_nodes_to_gdf_matches_golden(golden, stem_map):
    _, profile = stem_map
    stems = _quantified_stems(golden)
    gdf = IO.nodes_to_gdf(stems, profile)
    records = golden("stage_quantified_contour")
    assert len(gdf) == sum(len(r["diameters"]) for r in records)


def test_write_all_layers_roundtrip_matches_golden_gpkg(
        golden, stem_map, tmp_path, fixtures_dir):
    _, profile = stem_map
    stems = _quantified_stems(golden)
    out = IO.write_all_layers_to_gpkg(
        stems, profile, os.path.join(str(tmp_path), "roundtrip"))
    assert sorted(helpers.gpkg_layers(out)) == ["nodes", "stems", "vectors"]
    helpers.assert_gpkg_stems_match(
        out, os.path.join(fixtures_dir, "golden_stems.gpkg"))
    golden_gpkg = os.path.join(fixtures_dir, "golden_stems.gpkg")
    assert (helpers.gpkg_layer_count(out, "nodes")
            == helpers.gpkg_layer_count(golden_gpkg, "nodes"))


def test_golden_gpkg_has_crs(fixtures_dir):
    import pyogrio
    gdf = pyogrio.read_dataframe(
        os.path.join(fixtures_dir, "golden_stems.gpkg"), layer="stems")
    assert gdf.crs is not None and gdf.crs.to_epsg() == 25833
