"""A per-tile stem count must never read as the run's answer.

``connect_stems`` runs once per vector tile, so its counts are
intermediates. It used to end each tile with a bare "final number of
stems 60" while the run's real answer ("Total stems written:   215")
only appeared much later, in the merge summary -- which is exactly the
"is it 60 or 215?" confusion this pins shut.

The contract:

* a per-tile ``connect_stems`` line names its tile and marks itself as an
  intermediate;
* the un-tiled call (whole raster) carries no tile prefix, because there
  is no tile;
* the merged total is labelled as the final result and stays parseable by
  ``benchmark/benchmark_fixes.py``.
"""
import os

import pytest

from classes.Config import Config
from utils import Log
import utils.Vectorization as Vec


@pytest.fixture(autouse=True)
def _debug_level():
    previous = Log.current_level()
    Log.set_level(Log.DEBUG)
    yield
    Log.set_level(previous)


def _connect(capsys, **kwargs):
    """Run connect_stems on an empty input purely for its log output."""
    Vec.connect_stems([], Config(), **kwargs)
    return capsys.readouterr().out


def test_per_tile_line_identifies_its_tile(capsys):
    out = _connect(capsys, scope="Tile raster_r00000_c00001")
    assert "Tile raster_r00000_c00001: 0 stems after connect" in out


def test_per_tile_line_marks_itself_as_intermediate(capsys):
    out = _connect(capsys, scope="Tile raster_r00000_c00001")
    line = next(ln for ln in out.splitlines() if "after connect" in ln)
    assert "intermediate" in line
    # "final" is reserved for the run's answer.
    assert "final" not in line.lower()


def test_no_line_claims_to_be_the_final_stem_count(capsys):
    """The old wording -- 'final number of stems 60' -- is gone."""
    out = _connect(capsys, scope="Tile raster_r00000_c00001")
    assert "final number of stems" not in out.lower()


def test_untiled_call_has_no_tile_prefix(capsys):
    """connect_stems over the whole raster must not pretend to be a tile."""
    out = _connect(capsys)
    assert "Stems after connect: 0" in out
    assert "Tile " not in out
    assert "intermediate" not in out


def test_every_tile_scoped_line_carries_the_tile_id(capsys):
    """Not just the count -- the whole per-tile block is attributable."""
    scope = "Tile raster_r00007_c00003"
    out = _connect(capsys, scope=scope)
    body = [
        ln for ln in out.splitlines()
        if ln.strip() and not ln.startswith("#")
        and "Elapsed time" not in ln
    ]
    assert body, "connect_stems produced no debug output"
    assert all(ln.startswith(scope) for ln in body), body


def test_merge_summary_labels_the_final_total(capsys):
    """The merged answer is announced as final and stays machine-readable."""
    import utils.IO as IO

    summary = "\n".join(
        ln for ln in _merge_summary_source(IO).splitlines()
        if "MERGE SUMMARY" in ln or "Total stems written" in ln
    )
    assert "MERGE SUMMARY (final result for this run)" in summary
    # Every summary header is followed by a parseable stems line.
    assert summary.count("MERGE SUMMARY (final result for this run)") == \
        summary.count("Total stems written")


def test_the_benchmark_harness_can_read_the_real_merge_summary(
        capsys, tmp_path):
    """The two-sided contract, executed end to end.

    ``benchmark/benchmark_fixes.py`` greps the run's stem count out of
    stdout, so the producer (utils.IO) and the consumer (the harness's
    own ``_parse_run``) have to agree on one line format. Both halves
    are the real thing here: IO prints the merge summary, and the
    harness's parser -- imported, not restated -- reads it back.
    """
    import importlib.util
    import utils.IO as IO

    spec = importlib.util.spec_from_file_location(
        "benchmark_fixes",
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "benchmark", "benchmark_fixes.py"))
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    IO.merge_selected_tile_results(
        [], str(tmp_path / "out.gpkg"), raster_profile=None)
    stdout = capsys.readouterr().out

    assert "Total stems written" in stdout, "the producer changed its line"
    assert harness._parse_run(stdout)["stems_written"] == 0, (
        "benchmark_fixes.py can no longer read the merge summary; the "
        "'Total stems written:' line format and its regex have diverged")


def _merge_summary_source(module):
    import inspect
    return inspect.getsource(module)
