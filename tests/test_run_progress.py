"""The run progress bar must track real work, not stdout line counts.

Regression: the bar read ~78 % before the first inference because
``tasks_threads.Worker`` divided lines-seen by a hardcoded constant and
``winmol_run.py`` prints ~91 setup lines before "Loading Model...".
"""

import ast
import itertools
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_utils.run_progress import (  # noqa: E402
    BANDS, SETUP_END, RunProgress,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The chatty preamble winmol_run.py emits before "Loading Model..." — the
# thing that used to drive the bar to 78 %.
PREAMBLE = [
    "Start timer",
    "Hardware detected: CPUs=10, RAM=32.0 GB, GPUs=1",
    "Visible GPUs: ['Apple M2 Pro GPU']",
    "Execution plan:",
    "  prediction_mode  = stream",
    "  prediction_batch = 4",
    "Command-line arguments:",
    "Model Path: /models/General.onnx",
] + [f"  config_attr_{i:02d}            value" for i in range(58)]


def _feed_all(progress, lines):
    return [progress.feed(line) for line in lines]


def test_setup_chatter_leaves_the_bar_at_zero():
    progress = RunProgress("Trees")
    emitted = [v for v in _feed_all(progress, PREAMBLE) if v is not None]
    assert emitted == []
    assert progress.percent == 0


def test_zero_percent_at_loading_model():
    progress = RunProgress("Trees")
    _feed_all(progress, PREAMBLE)
    assert progress.feed("\nLoading Model...".strip()) is None
    assert progress.percent == 0


@pytest.mark.parametrize("process_type", ["Stems", "Trees", "Nodes"])
def test_prediction_band_tracks_written_tiles(process_type):
    progress = RunProgress(process_type)
    _feed_all(progress, PREAMBLE + ["Loading Model..."])
    pred_end = BANDS[process_type][0]

    progress.feed("Written tile 1/100 | 1.0% | 12.0 tiles/min | ETA 00m 10s")
    assert progress.percent == SETUP_END        # ~0 work done
    progress.feed("Written tile 50/100 | 50.0% | 12.0 tiles/min | ETA 1m")
    assert progress.percent == pytest.approx(
        SETUP_END + (pred_end - SETUP_END) // 2, abs=1)
    progress.feed("Written tile 100/100 | 100.0% | 12.0 tiles/min | ETA 0s")
    assert progress.percent == pred_end


def test_multi_gpu_prediction_line_is_understood():
    progress = RunProgress("Trees")
    progress.feed("Multi-GPU prediction 5/10 | 50.0% | 30.0 tiles/min | ETA 1m")
    assert progress.percent == pytest.approx(
        SETUP_END + (BANDS["Trees"][0] - SETUP_END) // 2, abs=1)
    assert progress.matched


def test_vector_and_merge_phases_advance_to_ninety_nine():
    progress = RunProgress("Trees")
    progress.feed("Written tile 8/8 | 100.0% | 12.0 tiles/min | ETA 00m 00s")
    assert progress.percent == BANDS["Trees"][0]

    progress.feed("Prepared 4/9 vector tiles with foreground | skipped 5")
    progress.feed("Vector tiles 2/4 | 50.0% | 3.0 tiles/min | ETA 1m")
    assert BANDS["Trees"][0] < progress.percent < BANDS["Trees"][1]
    progress.feed("Vector tiles 4/4 | 100.0% | 3.0 tiles/min | ETA 0s")
    assert progress.percent == BANDS["Trees"][1]

    for i in range(4):
        progress.feed(f"MERGE TILE READ | tile {i} | file /t/{i}.gpkg | "
                      "stems 3 | nodes 9 | vectors 3 | raster /t/r.tif")
    assert progress.percent == BANDS["Trees"][2] == 99
    assert progress.finish(ok=True) == 100


def test_merge_without_a_prepared_line_does_not_divide_by_zero():
    progress = RunProgress("Trees")
    progress.feed("Vector tiles 1/1 | 100.0% | 1.0 tiles/min | ETA 0s")
    before = progress.percent
    progress.feed("MERGE TILE READ | tile 0 | file /t/0.gpkg | stems 1 "
                  "| nodes 2 | vectors 1")
    assert progress.percent == before


def test_empty_vector_stage_line_does_not_crash():
    progress = RunProgress("Trees")
    progress.feed("Vector tiles 0/0 | no foreground tiles queued")
    assert progress.percent == BANDS["Trees"][0]


def test_autotune_candidates_tick_inside_the_setup_band():
    progress = RunProgress("Trees")
    progress.feed("Prediction micro-batch autotune candidate 1/13: b4 = "
                  "0.169s/tile")
    progress.feed("Prediction micro-batch autotune candidate 13/13: b16 = "
                  "0.381s/tile")
    assert progress.percent == SETUP_END
    # Still below the first real prediction step.
    progress.feed("Written tile 1/2 | 50.0% |")
    assert progress.percent >= SETUP_END


def test_progress_is_monotonic_and_capped_at_99_before_finish():
    progress = RunProgress("Trees")
    lines = PREAMBLE + [
        "Loading Model...",
        "Written tile 1/4 | 25.0% |",
        "Written tile 4/4 | 100.0% |",
        "Written tile 2/4 | 50.0% |",        # out-of-order / stale line
        "Prepared 2/4 vector tiles with foreground | skipped_empty 2",
        "Vector tiles 1/2 | 50.0% |",
        "Vector tiles 2/2 | 100.0% |",
        "MERGE TILE READ | tile 0 | file a | stems 1 | nodes 1 | vectors 1",
        "MERGE TILE READ | tile 1 | file b | stems 1 | nodes 1 | vectors 1",
    ]
    seen = [0]
    for line in lines:
        value = progress.feed(line)
        if value is not None:
            seen.append(value)
    assert seen == sorted(seen)
    assert max(seen) <= 99
    assert progress.finish() == 100


def test_unrecognised_stream_leaves_the_bar_alone():
    progress = RunProgress("Trees")
    for line in PREAMBLE:
        assert progress.feed(line) is None
    assert not progress.matched
    assert progress.percent == 0
    assert progress.finish(ok=False) == 0


def test_feed_tolerates_blank_and_none_lines():
    progress = RunProgress("Stems")
    assert progress.feed("") is None
    assert progress.feed(None) is None


# --- the contract with the producing modules -------------------------------

@pytest.mark.parametrize("relpath,needle", [
    ("utils/Prediction.py", 'f"Written tile {done}/{total_tiles} | "'),
    ("utils/PredictWorkers.py",
     'f"Multi-GPU prediction {done}/{total_tiles} | "'),
    ("utils/VectorTilePipeline.py",
     "f'Vector tiles {done}/{total} | "),
    ("winmol_run.py", 'f"Prepared {len(tile_paths)}/{len(jobs)} vector tiles '),
    ("utils/IO.py", 'f"MERGE TILE READ | tile '),
])
def test_producers_still_emit_the_parsed_formats(relpath, needle):
    with open(os.path.join(REPO, relpath), encoding="utf-8") as handle:
        source = handle.read()
    assert needle in source, (
        f"{relpath} no longer emits the format plugin_utils/run_progress.py "
        "parses; update BOTH or the progress bar silently freezes.")


def test_merge_tile_counter_survives_normal_verbosity(capsys):
    """The merge counter must not be gated behind ``--debug``.

    A source-level needle is not enough here: utils/Log.py can suppress a
    line that is still present in the source, which freezes the merge band
    of the bar without failing any other test.
    """
    from utils import IO as _IO
    from utils import Log

    previous = Log.current_level()
    try:
        Log.set_level("normal")
        _IO._log_merge_tile_read("t_0_0", "/tmp/t.gpkg", [], [], [])
        emitted = capsys.readouterr().out
    finally:
        Log.set_level(previous)

    progress = RunProgress("Trees")
    progress.feed("Prepared 2/2 vector tiles with foreground | skipped 0")
    assert progress.feed(emitted.strip()) == 97, (
        "utils/IO.py's MERGE TILE READ line is invisible at normal "
        "verbosity; run_progress.py counts it to advance the merge band.")


def test_worker_no_longer_counts_lines():
    with open(os.path.join(REPO, "tasks_threads.py"), encoding="utf-8") as fh:
        source = fh.read()
    assert "get_total_lines" not in source
    assert re.search(r"RunProgress\(", source)


# --- the lines the producers ACTUALLY emit, end to end ---------------------
#
# The source-substring pins above catch a renamed prefix, but they would
# happily pass an edit that keeps the substring and still moves the counter
# (e.g. text inserted between {total} and the '|'). These tests rebuild the
# line each producer really prints, straight from its f-string, and run it
# through the parser — which is the only thing that proves the bar still
# advances.

def _render_joined(node, values):
    """Render an ast.JoinedStr with placeholders for its {expressions}."""
    out = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            out.append(str(part.value))
        else:
            out.append(next(values))
    return "".join(out)


def _emitted_line(relpath, head, done="156", total="182"):
    """The line ``relpath`` prints, with ``done``/``total`` substituted
    for its first two {expressions} and 9 for the rest."""
    with open(os.path.join(REPO, relpath), encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        line = _render_joined(
            node, itertools.chain([done, total], itertools.repeat("9")))
        if line.startswith(head):
            return line
    pytest.fail(f"{relpath} no longer emits a line starting {head!r}")


PRODUCER_LINES = [
    ("utils/Prediction.py", "Written tile "),
    ("utils/PredictWorkers.py", "Multi-GPU prediction "),
    ("utils/VectorTilePipeline.py", "Vector tiles "),
    ("winmol_run.py", "Prepared "),
    ("utils/IO.py", "MERGE TILE READ | tile "),
]


@pytest.mark.parametrize("relpath,head", PRODUCER_LINES)
def test_the_line_the_producer_emits_is_still_recognised(relpath, head):
    line = _emitted_line(relpath, head)
    progress = RunProgress("Trees")
    progress.feed(line)
    if head == "Prepared ":
        # Prepared only sets the merge denominator; it moves nothing.
        assert progress._merge_total == 156
        return
    assert progress.matched, (
        f"{relpath} emits {line!r}, which plugin_utils/run_progress.py no "
        "longer parses — the progress bar would silently freeze")


def test_the_new_unit_labels_do_not_move_the_counters():
    """Prediction tiles and vector tiles now say which they are. The
    counter still has to sit in exactly the same place."""
    progress = RunProgress("Trees")
    progress.feed(
        "Written tile 50/100 | prediction tile | 50.0% | 12.0 tiles/min | "
        "ETA 1m | src 727x727 -> out 504x504")
    assert progress.percent == pytest.approx(
        SETUP_END + (BANDS["Trees"][0] - SETUP_END) // 2, abs=1)

    progress.feed(
        "Written tile 100/100 | prediction tile | 100.0% | 12 tiles/min")
    assert progress.percent == BANDS["Trees"][0]

    progress.feed("Prepared 2/9 vector tiles with foreground | skipped 7")
    progress.feed(
        "Vector tiles 1/2 | vector tile ~4144x4144 px | 50.0% | 0.1 "
        "tiles/min | ETA 1m | wrote 1 | empty 0 | no_output 0 | avg total "
        "12.707s quant 1.008s connect 2.854s")
    assert BANDS["Trees"][0] < progress.percent < BANDS["Trees"][1]
    progress.feed(
        "Vector tiles 2/2 | vector tile ~4144x4144 px | 100.0% | 0.1 "
        "tiles/min | ETA 0s | wrote 2 | empty 0 | no_output 0 | avg total "
        "12.707s quant 1.008s connect 2.854s")
    assert progress.percent == BANDS["Trees"][1]

    for i in range(2):
        progress.feed(f"MERGE TILE READ | tile {i} | vector tile")
    assert progress.percent == BANDS["Trees"][2] == 99


def test_the_empty_vector_stage_line_still_parses_with_its_unit():
    progress = RunProgress("Trees")
    progress.feed("Vector tiles 0/0 | vector tile | no foreground tiles "
                  "queued")
    assert progress.percent == BANDS["Trees"][0]


def test_the_vector_stage_names_its_tile_size():
    """VectorTilePipeline gets the size handed down from winmol_run.py
    (it must not reopen a raster just to log), and renders it into every
    'Vector tiles' line and its own phase header."""
    size = _emitted_line("utils/VectorTilePipeline.py", "vector tile ~")
    assert size == "vector tile ~156x182 px"
    header = _emitted_line("utils/VectorTilePipeline.py", "VECTOR PHASE")
    assert "vector tiles" in header


@pytest.mark.parametrize("relpath,head,unit", [
    ("utils/Prediction.py", "PREDICTION PHASE", "prediction tiles"),
    ("winmol_run.py", "VECTOR PHASE", "vector tiles"),
])
def test_each_phase_announces_its_unit_and_tile_size(relpath, head, unit):
    """The user's complaint: '182 tiles' and '3 tiles' were the same
    word for a 727 px inference tile and a ~4144 px vector tile. Each
    phase now opens by saying which unit it is counting and how big it
    is. These headers are new and standalone — the parser ignores them,
    so they cannot freeze the bar."""
    line = _emitted_line(relpath, head)
    assert unit in line
    assert "px" in line
    progress = RunProgress("Trees")
    assert progress.feed(line) is None
    assert not progress.matched
