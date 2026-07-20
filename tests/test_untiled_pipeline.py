"""End-to-end test for the un-tiled vector mode (plan.vector_mode='untiled').

Runs ImageProcessing.run_vector_phase on the golden stem-map fixture with
Config.vector_processing='untiled' and verifies:

* the legacy reference stem count (58 — golden_stems.gpkg was produced by the
  same whole-raster chain), content-matching the untiled golden, and
* the GeoPackage contract matches the tiled merge output exactly (layer
  names, stems schema/dtypes, CRS) as pinned by golden_merged.gpkg.

Requires PYTHONHASHSEED=0 (enforced by tests/conftest.py's re-exec guard;
connect_stems is set-iteration-order sensitive, docs/CODE_REVIEW_2.md A-4).
"""

import os

import pytest

import helpers
from classes.HardwareInfo import HardwareInfo

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def untiled_gpkg(tmp_path_factory):
    from winmol_run import ImageProcessing

    fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "fixtures")
    stem_map_path = os.path.join(fixtures_dir, "stem_map.tif")
    if not os.path.exists(stem_map_path):
        pytest.skip("fixture stem_map.tif missing — run generate_fixtures.py")

    out_path = str(tmp_path_factory.mktemp("untiled") / "untiled.gpkg")
    ip = ImageProcessing(
        model_path="unused-by-vector-phase",
        uav_path=stem_map_path,
        stem_path=stem_map_path,
        trees_path=out_path,
        process_type="Trees",
    )
    # Force the untiled path regardless of this machine's RAM so the test is
    # deterministic; hardware is mocked for the same reason.
    ip.config.vector_processing = "untiled"
    hardware = HardwareInfo(
        cpu_count=4, total_ram_gb=16.0, gpu_count=0,
        gpu_names=[], gpu_memory_gb=[],
    )
    plan = ip.build_plan(hardware)
    assert plan.vector_mode == "untiled"

    result = ip.run_vector_phase(plan, pred_path=stem_map_path)
    assert result == out_path
    assert os.path.exists(out_path)
    return out_path


def _fixtures_path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fixtures", name)


def test_untiled_reproduces_legacy_stem_count(untiled_gpkg):
    assert helpers.gpkg_layer_count(untiled_gpkg, "stems") == 58


def test_untiled_matches_untiled_golden_content(untiled_gpkg):
    helpers.assert_gpkg_stems_match(
        untiled_gpkg, _fixtures_path("golden_stems.gpkg"))


def test_untiled_gpkg_layers_match_tiled_merge_contract(untiled_gpkg):
    # golden_merged.gpkg is the pinned output of the tiled merge path.
    assert (helpers.gpkg_layers(untiled_gpkg)
            == helpers.gpkg_layers(_fixtures_path("golden_merged.gpkg")))


def test_untiled_stems_schema_matches_tiled_merge_schema(untiled_gpkg):
    import pyogrio
    actual = pyogrio.read_dataframe(untiled_gpkg, layer="stems")
    tiled = pyogrio.read_dataframe(
        _fixtures_path("golden_merged.gpkg"), layer="stems")
    assert list(actual.columns) == list(tiled.columns)
    assert ({c: str(t) for c, t in actual.dtypes.items()}
            == {c: str(t) for c, t in tiled.dtypes.items()})
    assert actual.crs == tiled.crs


def test_untiled_child_layers_are_consistent(untiled_gpkg):
    import pyogrio
    stems = pyogrio.read_dataframe(untiled_gpkg, layer="stems")
    nodes = pyogrio.read_dataframe(untiled_gpkg, layer="nodes")
    vectors = pyogrio.read_dataframe(untiled_gpkg, layer="vectors")
    # One node per path coordinate, across all stems.
    n_coords = int(sum(len(g.coords) for g in stems.geometry))
    assert len(nodes) == n_coords
    # Measurement vectors: at most one per node, none orphaned.
    assert 0 < len(vectors) <= n_coords
    assert set(vectors["stem_id"]) <= set(stems["stem_id"])
