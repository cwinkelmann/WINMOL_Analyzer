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
