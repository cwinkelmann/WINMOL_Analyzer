"""The run progress bar must track real work, not stdout line counts.

Each producer's counter line (copied verbatim from the pipeline's own
print format strings) must parse to the right (done, total) and land in
the right phase band; unknown lines must be ignored; the sequence must be
monotonic 0->100.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_utils.run_progress import (  # noqa: E402
    BANDS, SETUP_END, RunProgress,
)

# Chatty setup preamble winmol_run.py prints before "Loading Model...".
PREAMBLE = [
    "Start timer",
    "Hardware detected: CPUs=10, RAM=32.0 GB, GPUs=1",
    "Execution plan:",
    "  prediction_mode  = stream",
    "Command-line arguments:",
] + [f"  config_attr_{i:02d}            value" for i in range(10)]


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
    assert progress.feed("Loading Model...") is None
    assert progress.percent == 0


@pytest.mark.parametrize("process_type", ["Stems", "Trees", "Nodes"])
def test_prediction_band_tracks_written_tiles(process_type):
    # utils/Prediction.py:913 -- f"Written tile {done}/{total_tiles} | "
    progress = RunProgress(process_type)
    _feed_all(progress, PREAMBLE + ["Loading Model..."])
    pred_end = BANDS[process_type][0]

    progress.feed(
        "Written tile 1/100 | 1.0% | 12.0 tiles/min | ETA 00m 10s | "
        "avg read 0.012s prep 0.034s infer 0.056s write 0.008s | batch 4 "
        "| queue 12% full | producers 2 | src 727x727 -> out 504x504")
    assert progress.percent == SETUP_END
    progress.feed(
        "Written tile 50/100 | 50.0% | 12.0 tiles/min | ETA 00m 05s | "
        "avg read 0.012s prep 0.034s infer 0.056s write 0.008s | batch 4 "
        "| queue 12% full | producers 2 | src 727x727 -> out 504x504")
    assert progress.percent == pytest.approx(
        SETUP_END + (pred_end - SETUP_END) // 2, abs=1)
    progress.feed(
        "Written tile 100/100 | 100.0% | 12.0 tiles/min | ETA 00m 00s | "
        "avg read 0.012s prep 0.034s infer 0.056s write 0.008s | batch 4 "
        "| queue 12% full | producers 2 | src 727x727 -> out 504x504")
    assert progress.percent == pred_end


def test_multi_gpu_prediction_line_is_understood():
    # utils/PredictWorkers.py:483 -- f"Multi-GPU prediction {done}/{total} | "
    progress = RunProgress("Trees")
    progress.feed(
        "Multi-GPU prediction 5/10 | 50.0% | 30.0 tiles/min | ETA 00m 10s "
        "| avg read 0.012s infer 0.045s write 0.008s")
    assert progress.percent == pytest.approx(
        SETUP_END + (BANDS["Trees"][0] - SETUP_END) // 2, abs=1)
    assert progress.matched


def test_vector_and_merge_phases_advance_to_ninety_nine():
    progress = RunProgress("Trees")
    progress.feed(
        "Written tile 8/8 | 100.0% | 12.0 tiles/min | ETA 00m 00s")
    assert progress.percent == BANDS["Trees"][0]

    # winmol_run.py:214 -- f"Prepared {n}/{m} vector tiles with foreground "
    progress.feed(
        "Prepared 4/9 vector tiles with foreground | skipped_empty 5")
    # utils/VectorTilePipeline.py:389 -- f'Vector tiles {done}/{total} | '
    progress.feed(
        "Vector tiles 2/4 | 50.0% | 3.0 tiles/min | ETA 00m 20s | wrote 2 "
        "| empty 1 | no_output 0 | avg total 12.707s quant 1.008s connect "
        "2.854s")
    assert BANDS["Trees"][0] < progress.percent < BANDS["Trees"][1]
    progress.feed(
        "Vector tiles 4/4 | 100.0% | 3.0 tiles/min | ETA 00m 00s | wrote 4 "
        "| empty 1 | no_output 0 | avg total 12.707s quant 1.008s connect "
        "2.854s")
    assert progress.percent == BANDS["Trees"][1]

    # utils/IO.py:1101 -- f"MERGE TILE READ | tile {tile_id} | file {gpkg_path}"
    for i in range(4):
        progress.feed(
            f"MERGE TILE READ | tile t_{i}_0 | file /t/{i}.gpkg | stems 3 "
            "| nodes 9 | vectors 3 | raster /t/r.tif")
    assert progress.percent == BANDS["Trees"][2] == 99
    assert progress.finish(ok=True) == 100


def test_merge_without_a_prepared_line_does_not_divide_by_zero():
    progress = RunProgress("Trees")
    progress.feed("Vector tiles 1/1 | 100.0% | 1.0 tiles/min | ETA 00m 00s")
    before = progress.percent
    progress.feed(
        "MERGE TILE READ | tile t_0_0 | file /t/0.gpkg | stems 1 | nodes 2 "
        "| vectors 1 | raster /t/r.tif")
    assert progress.percent == before


def test_empty_vector_stage_line_does_not_crash():
    # utils/VectorTilePipeline.py:463
    progress = RunProgress("Trees")
    progress.feed("Vector tiles 0/0 | no foreground tiles queued")
    assert progress.percent == BANDS["Trees"][0]


def test_autotune_regex_is_dead_but_harmless():
    """No producer in this lineage prints "autotune candidate i/n" -- the
    regex is kept for parity with the RR6 lineage where it is live, but
    never fires against our pipeline's actual stdout."""
    with open(os.path.join(REPO, "utils", "Prediction.py"),
              encoding="utf-8") as handle:
        assert "autotune candidate" not in handle.read()

    # The regex itself still behaves correctly if fed a matching line --
    # it is dead code here, not broken code.
    progress = RunProgress("Trees")
    progress.feed("Prediction micro-batch autotune candidate 13/13: "
                  "0.169s/tile")
    assert progress.matched
    assert progress.percent == SETUP_END


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
        "MERGE TILE READ | tile t_0_0 | file a | stems 1 | nodes 1 "
        "| vectors 1 | raster r",
        "MERGE TILE READ | tile t_1_0 | file b | stems 1 | nodes 1 "
        "| vectors 1 | raster r",
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


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("relpath,needle", [
    ("utils/Prediction.py", 'f"Written tile {done}/{total_tiles} | "'),
    ("utils/PredictWorkers.py",
     'f"Multi-GPU prediction {done}/{total_tiles} | "'),
    ("utils/VectorTilePipeline.py",
     "f'Vector tiles {done}/{total} | "),
    ("winmol_run.py",
     'f"Prepared {len(tile_paths)}/{len(jobs)} vector tiles '),
    ("utils/IO.py", 'f"MERGE TILE READ | tile '),
])
def test_producers_still_emit_the_parsed_formats(relpath, needle):
    with open(os.path.join(REPO, relpath), encoding="utf-8") as handle:
        source = handle.read()
    assert needle in source, (
        f"{relpath} no longer emits the format plugin_utils/run_progress.py "
        "parses; update BOTH or the progress bar silently freezes.")
