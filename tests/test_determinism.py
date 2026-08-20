"""winmol_run.py must normalize PYTHONHASHSEED=0 before anything hashes into
a set (connect_stems joins stems in Part-hash set-iteration order, which
Python otherwise salts per process -- see the re-exec guard at the top of
winmol_run.py).
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_guard_precedes_project_imports():
    with open(os.path.join(REPO, "winmol_run.py")) as f:
        source = f.read()

    guard_idx = source.index("os.execv(sys.executable")
    first_project_import = min(
        idx for idx in (
            source.index("from classes"),
            source.index("import json"),
        ) if idx != -1
    )
    assert guard_idx < first_project_import, (
        "the PYTHONHASHSEED re-exec guard must run before any project "
        "import (or anything else) could hash into a set")


@pytest.mark.parametrize("incoming_seed", [None, "1", "2"])
def test_reexec_normalizes_hash_seed(tmp_path, incoming_seed):
    env = os.environ.copy()
    if incoming_seed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = incoming_seed

    proc = subprocess.run(
        [sys.executable, "winmol_run.py", "/nonexistent/model.onnx",
         "/nonexistent/in.tif", str(tmp_path / "sm.tif"),
         str(tmp_path / "pfx"), "Stems"],
        capture_output=True, text=True, env=env, cwd=REPO, timeout=120)

    # The run fails later on the bogus model -- expected, not asserted here.
    # What matters is that the seed was normalized before that failure.
    assert "Determinism: PYTHONHASHSEED=0" in proc.stdout, (
        f"incoming seed {incoming_seed!r} was not normalized\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
