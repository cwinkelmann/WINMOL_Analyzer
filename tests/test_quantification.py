"""Golden tests for quantify_stems (diameters, lengths, frustum volumes).

Input = the connect_stems fixture (fresh Stem objects, empty measurement
lists); pred/profile = the golden stem map. Both diameter methods are pinned:
contour (default) and EDT.

Known behavior pinned on purpose: calc_d returns 0.0 when no chord passes a
node (A-10) and diameters are capped by the 2 m probe (contour method).
"""

import numpy as np
import pytest

import helpers
from utils import Quantification as Quant


@pytest.fixture()
def connect_stage_stems(golden):
    return helpers.stems_from_canonical(golden("stage_connect_stems"))


def test_quantify_contour_matches_golden(connect_stage_stems, stem_map,
                                         pipeline_config, golden):
    pred, profile = stem_map
    stems = Quant.quantify_stems(
        connect_stage_stems, pred, profile, pipeline_config)
    helpers.assert_stems_match_golden(
        stems, golden("stage_quantified_contour"))


def test_quantify_edt_matches_golden(connect_stage_stems, stem_map,
                                     pipeline_config, golden):
    import copy
    pred, profile = stem_map
    config = copy.copy(pipeline_config)
    config.diameter_method = "edt"
    stems = Quant.quantify_stems(connect_stage_stems, pred, profile, config)
    helpers.assert_stems_match_golden(stems, golden("stage_quantified_edt"))


def test_quantified_totals_are_positive(golden):
    records = golden("stage_quantified_contour")
    total_len = sum(sum(r["lengths"]) for r in records)
    total_vol = sum(sum(r["volumes"]) for r in records)
    assert total_len > 0 and total_vol > 0
    # per-stem: one length/volume per path segment, one diameter per node
    for r in records:
        assert len(r["lengths"]) == len(r["path"]) - 1
        assert len(r["volumes"]) == len(r["path"]) - 1
        assert len(r["diameters"]) == len(r["path"])


def test_quantify_empty_input(stem_map, pipeline_config):
    pred, profile = stem_map
    assert Quant.quantify_stems([], pred, profile, pipeline_config) == []


def test_edt_and_contour_disagree(golden):
    """The two diameter methods are different estimators; identical outputs
    would indicate the config switch stopped working."""
    contour = golden("stage_quantified_contour")
    edt = golden("stage_quantified_edt")
    d_contour = np.array([d for r in contour for d in r["diameters"]])
    d_edt = np.array([d for r in edt for d in r["diameters"]])
    assert d_contour.shape == d_edt.shape
    assert not np.allclose(d_contour, d_edt)


def test_shapes_foreground_mask_equals_postfiltered_shapes(stem_map):
    """get_diameters now passes mask=foreground to rasterio.features.shapes
    instead of polygonizing everything and filtering raster_val == 1 post
    hoc. The foreground shape set (holes included) must be unchanged."""
    import rasterio.features
    from shapely.geometry import shape

    pred, profile = stem_map
    pred_bin = Quant._as_binary_mask(pred).astype(np.int16, copy=False)
    transform = profile["transform"]

    unmasked = [
        shape(geom).wkb
        for geom, value in rasterio.features.shapes(
            pred_bin, mask=None, transform=transform)
        if value == 1
    ]
    masked = [
        shape(geom).wkb
        for geom, value in rasterio.features.shapes(
            pred_bin, mask=pred_bin.astype(bool), transform=transform)
    ]
    assert masked == unmasked
