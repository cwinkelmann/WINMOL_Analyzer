"""Nodata-aware edge fill for U-Net segmentation tiles.

Shared by the single-GPU stream producer (utils/Prediction.py) and the
multi-GPU reader (utils/PredictWorkers.py). Kept in its own leaf module so
the multiprocessing workers in PredictWorkers can import it without pulling
in the heavy utils.Prediction dependency graph.

See docs/superpowers/specs/2026-07-28-segmentation-edge-fill-design.md.
"""
import numpy as np


def fill_invalid_with_nearest(tile, valid_mask):
    """Replace invalid pixels with their nearest valid neighbour.

    The U-Net must not see the hard black cliff at a nodata / out-of-bounds
    boundary: its receptive field bleeds spurious stem activations onto the
    valid pixels just inside the edge. Replicating the nearest valid pixel
    into the invalid region removes the cliff before inference; the output is
    still masked afterwards, so filled regions produce no stems.

    tile:       (H, W, C) array as read from the source (any dtype).
    valid_mask: (H, W) bool -- True where the pixel is real data.

    Returns a NEW tile with invalid pixels replaced; valid pixels untouched.
    Returns the input unchanged when the mask is all-valid or all-invalid.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        return tile
    if valid.all() or not valid.any():
        return tile
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(
        ~valid, return_distances=False, return_indices=True)
    return tile[tuple(idx)]
