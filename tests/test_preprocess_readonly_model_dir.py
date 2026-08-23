"""The wrapped graph must not require a writable model directory.

Measured on carrot 2026-08-23: mounting the models read-only --
``-v /models:/data/models:ro``, the natural instinct for a directory of
immutable weights -- killed the run at
``onnx_preprocess.build_preprocessed_model``:

    OSError: [Errno 30] Read-only file system:
      '/data/models/.winmol_pre_8fc44c50446c9a52.onnx'

The wrap is a derived cache artifact, so a read-only weights mount should
degrade to a fallback location rather than fail the run.
"""
import os

import pytest

from conftest import build_tiny_unet
from utils.onnx_preprocess import build_preprocessed_model


@pytest.fixture()
def readonly_model_dir(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = build_tiny_unet(model_dir / "m.onnx")
    os.chmod(model_dir, 0o555)
    # chmod does not make a DIRECTORY read-only on Windows -- it only
    # touches the read-only attribute on files -- and root ignores the
    # bits anyway. Check that the precondition actually holds instead of
    # asserting behaviour the platform cannot produce: measured on
    # windows-latest, the wrap simply landed in the models dir and the
    # fallback assertion failed on a perfectly correct build.
    if os.access(model_dir, os.W_OK):
        os.chmod(model_dir, 0o755)
        pytest.skip("cannot make a directory read-only here "
                    "(Windows, or running as root)")
    yield model, model_dir
    os.chmod(model_dir, 0o755)


def test_wrap_succeeds_when_the_model_dir_is_read_only(readonly_model_dir):
    model, _ = readonly_model_dir

    out = build_preprocessed_model(model, (8, 8))

    assert os.path.exists(out)


def test_wrap_falls_back_outside_a_read_only_model_dir(readonly_model_dir):
    model, model_dir = readonly_model_dir

    out = build_preprocessed_model(model, (8, 8))

    assert os.path.dirname(os.path.realpath(out)) != str(model_dir)


def test_a_failed_wrap_leaves_no_cache_entry_behind(tmp_path, monkeypatch):
    """A wrap that dies mid-write must not leave a loadable-looking file.

    Measured on carrot 2026-08-23 with --jobs 8: every job derives the same
    wrap hash and writes the same path in the shared models dir, so one job
    loaded another's half-written file and died with

        INVALID_PROTOBUF : Load model from .winmol_pre_<hash>.onnx failed

    The cache-hit check is a bare os.path.exists, so a partial file is
    indistinguishable from a good one. Publishing the wrap atomically is
    what makes concurrent jobs safe; this pins the observable consequence.
    """
    import onnx

    model = build_tiny_unet(tmp_path / "m.onnx")
    real_save = onnx.save

    def save_then_die(m, path):
        real_save(m, path)
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(onnx, "save", save_then_die)

    with pytest.raises(RuntimeError):
        build_preprocessed_model(model, (8, 8))

    assert list(tmp_path.glob(".winmol_pre_*.onnx")) == []
