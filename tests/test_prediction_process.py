"""Prediction runs in a child that exits before vectorisation.

The point of utils.PredictionProcess is that the CUDA session's ~10 GB
virtual reservation dies WITH the child, because the driver never
returns it while the process lives. That cannot be tested without a
GPU, but two things that keep the mechanism honest can be:

  * a failure in the child reaches the parent as an exception carrying
    the child's traceback -- otherwise a broken prediction would leave
    the parent vectorising a stem map that was never written;
  * the child is a SPAWNED process, not a fork -- a forked child inherits
    the parent's mappings, which defeats the purpose.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def test_child_failure_reaches_the_parent_with_its_traceback(tmp_path):
    from classes.Config import Config
    from utils.PredictionProcess import predict_in_child
    # A model path that does not exist fails inside the child, in
    # IO.load_model_from_path, before any CUDA is touched.
    with pytest.raises(RuntimeError) as exc:
        predict_in_child(
            str(tmp_path / "no_such_model.onnx"),
            str(tmp_path / "no_such_ortho.tif"),
            str(tmp_path / "stem_map.tif"),
            Config(),
        )
    msg = str(exc.value)
    assert "prediction child process failed" in msg
    # the CHILD's traceback, not a bare exit code: the operator must see
    # where it died, not just that it did
    assert "Traceback" in msg


def test_child_uses_the_spawn_start_method():
    import inspect
    from utils import PredictionProcess
    src = inspect.getsource(PredictionProcess.predict_in_child)
    assert "get_context('spawn')" in src, (
        "must spawn: a forked child inherits the parent's CUDA mappings")
