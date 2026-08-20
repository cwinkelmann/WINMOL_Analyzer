"""The nodata-boundary margin in _binarize_prediction_core.

At a nodata cliff the U-Net fires on the transition itself, leaving a thin
rim of foreground on VALID pixels that `pred & mask` keeps and
vectorization turns into stems tracing the ortho outline. The margin
erodes the validity mask inward from real nodata only.

The property that is easy to get wrong: every tile is a crop, so the edges
of `mask_core` are tile seams, not boundaries. Eroding those would punch a
hole at every tile join across the whole ortho.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import Prediction as Pred  # noqa: E402

_binarize_prediction_core = Pred._binarize_prediction_core


def test_margin_zero_is_the_old_behaviour():
    pred = np.full((8, 8), 0.9)
    mask = np.ones((8, 8), bool)
    mask[:, :2] = False
    out = _binarize_prediction_core(pred, mask, edge_margin=0)
    assert out.sum() == mask.sum()


def test_fully_valid_tile_is_untouched():
    """The common case: no nodata in the tile, so nothing to erode --
    and no cost, since the whole ortho interior takes this path."""
    pred = np.full((8, 8), 0.9)
    mask = np.ones((8, 8), bool)
    out = _binarize_prediction_core(pred, mask, edge_margin=4)
    assert out.all()


def test_tile_seams_are_not_eroded():
    """A tile whose only invalid pixels are a nodata block: the crop's own
    borders must survive, or every tile join loses a margin."""
    pred = np.full((16, 16), 0.9)
    mask = np.ones((16, 16), bool)
    mask[:, :4] = False                      # nodata on the left
    out = _binarize_prediction_core(pred, mask, edge_margin=2)
    assert out[0, -1] == 1, "top-right corner (a tile seam) was eroded"
    assert out[-1, -1] == 1, "bottom-right corner (a tile seam) was eroded"
    assert out[8, -1] == 1, "right edge (a tile seam) was eroded"


def test_rim_next_to_nodata_is_dropped():
    pred = np.full((16, 16), 0.9)
    mask = np.ones((16, 16), bool)
    mask[:, :4] = False
    out = _binarize_prediction_core(pred, mask, edge_margin=2)
    assert out[:, 4:6].sum() == 0, "the 2 px rim beside nodata survived"
    assert out[:, 6:].all(), "erosion reached further than the margin"


def test_prediction_below_threshold_still_excluded():
    pred = np.full((8, 8), 0.2)
    mask = np.ones((8, 8), bool)
    mask[0, 0] = False
    out = _binarize_prediction_core(pred, mask, edge_margin=1)
    assert out.sum() == 0
