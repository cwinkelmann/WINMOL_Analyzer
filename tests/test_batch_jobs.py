"""Parallel batch processing must run every orthomosaic, exactly once.

A scheduler that silently drops or duplicates work is worse than the sequential
loop it replaces, so these pin the invariants rather than the speed. They also
pin the failure behaviour: the previous loop raised out on the first bad file
and abandoned the rest of the batch.
"""
import subprocess

import pytest

import winmol_batch


@pytest.fixture()
def fake_run(monkeypatch):
    """Record calls instead of launching winmol_run.py."""
    calls = []

    def _run(input_image, model_path, output_folder, gpu_id=None):
        calls.append({"ortho": input_image, "gpu": gpu_id})
        if "boom" in input_image:
            raise subprocess.CalledProcessError(1, "winmol_run.py")

    monkeypatch.setattr(winmol_batch, "run_winmol", _run)
    return calls


def test_every_ortho_runs_exactly_once_in_parallel(fake_run, monkeypatch):
    monkeypatch.setattr(winmol_batch, "detect_gpu_count", lambda: 4)
    orthos = [f"/in/o{i}.tif" for i in range(10)]

    failures = winmol_batch.process_orthos(orthos, "/m.onnx", "/out", jobs=4)

    assert failures == []
    assert sorted(c["ortho"] for c in fake_run) == sorted(orthos)


def test_jobs_are_spread_across_the_available_gpus(fake_run, monkeypatch):
    monkeypatch.setattr(winmol_batch, "detect_gpu_count", lambda: 4)
    orthos = [f"/in/o{i}.tif" for i in range(8)]

    winmol_batch.process_orthos(orthos, "/m.onnx", "/out", jobs=4)

    # Every job pinned to a real device, and all four devices used.
    assert {c["gpu"] for c in fake_run} == {0, 1, 2, 3}


def test_no_gpu_means_no_pinning(fake_run, monkeypatch):
    monkeypatch.setattr(winmol_batch, "detect_gpu_count", lambda: 0)

    winmol_batch.process_orthos(["/in/a.tif", "/in/b.tif"], "/m.onnx", "/out",
                                jobs=2)

    assert {c["gpu"] for c in fake_run} == {None}


def test_one_failure_does_not_abandon_the_batch(fake_run, monkeypatch):
    """The old loop raised on the first bad file, losing every later one."""
    monkeypatch.setattr(winmol_batch, "detect_gpu_count", lambda: 2)
    orthos = ["/in/a.tif", "/in/boom.tif", "/in/c.tif", "/in/d.tif"]

    failures = winmol_batch.process_orthos(orthos, "/m.onnx", "/out", jobs=2)

    assert [f[0] for f in failures] == ["/in/boom.tif"]
    # the other three still ran
    assert sorted(c["ortho"] for c in fake_run) == sorted(orthos)


def test_sequential_path_still_reports_failures(fake_run, monkeypatch):
    orthos = ["/in/a.tif", "/in/boom.tif", "/in/c.tif"]

    failures = winmol_batch.process_orthos(orthos, "/m.onnx", "/out", jobs=1)

    assert [f[0] for f in failures] == ["/in/boom.tif"]
    assert len(fake_run) == 3
    assert {c["gpu"] for c in fake_run} == {None}
