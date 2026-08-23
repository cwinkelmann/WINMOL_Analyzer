"""An orthomosaic with no detections is a result, not a crash.

Measured on carrot 2026-08-23: a clean container run reported
``Total stems: 0`` and then died in the merge with
``FileNotFoundError: No .gpkg files found``, which marked the whole
orthomosaic FAILED. Treeless ground is a legitimate answer, and in a large
batch it must not abort the ortho that produced it.

The merge already has a zero-feature path (it prints MERGE SUMMARY with
zeros and creates no GeoPackage) for the case where tiles exist but hold
nothing; this pins that no-tiles-at-all lands there too.
"""
from utils import IO


def test_merge_of_a_run_with_no_detections_is_not_an_error(tmp_path):
    out = IO.merge_and_filter_tiled_results(str(tmp_path))

    assert out is not None


def test_merge_of_a_run_with_no_detections_creates_no_geopackage(tmp_path):
    import os

    out = IO.merge_and_filter_tiled_results(str(tmp_path))

    assert not os.path.exists(out)
