import numpy as np

from utils.edge_fill import fill_invalid_with_nearest


def test_all_valid_is_noop():
    tile = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    mask = np.ones((2, 2), dtype=bool)
    out = fill_invalid_with_nearest(tile, mask)
    assert np.array_equal(out, tile)


def test_all_invalid_is_noop():
    tile = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=bool)
    out = fill_invalid_with_nearest(tile, mask)
    assert np.array_equal(out, tile)


def test_invalid_pixels_take_nearest_valid_value():
    # Row of 3: left pixel valid, middle+right invalid (black).
    tile = np.zeros((1, 3, 3), dtype=np.uint8)
    tile[0, 0] = (10, 20, 30)
    mask = np.array([[True, False, False]])
    out = fill_invalid_with_nearest(tile, mask)
    assert tuple(out[0, 0]) == (10, 20, 30)   # valid untouched
    assert tuple(out[0, 1]) == (10, 20, 30)   # nearest-valid replicate
    assert tuple(out[0, 2]) == (10, 20, 30)


def test_does_not_mutate_input():
    tile = np.zeros((1, 2, 3), dtype=np.uint8)
    tile[0, 0] = (5, 6, 7)
    mask = np.array([[True, False]])
    original = tile.copy()
    _ = fill_invalid_with_nearest(tile, mask)
    assert np.array_equal(tile, original)
