"""Quantification must give the same answer however it is parallelised.

The stem-level stages were farmed out to a PROCESS pool, which pickled every
Stem (shapely geometry + diameter lists) to a worker and back. Measured on one
4096-px tile with 1567 stems, that made quantification ~10x SLOWER than doing
it serially (45.2 s vs 4.6 s) — the transport cost dwarfed 4.6 s of actual
work. These tests pin the invariant that matters while that is changed: the
result must not depend on the worker count.
"""
import copy

import numpy as np
from affine import Affine
from shapely.geometry import LineString, Point

from classes.Stem import Stem
from utils import Quantification as Quant


class _Cfg:
    cpu_workers = 1
    diameter_method = "contour"
    diameter_vector_half_length_m = 1.0


def _scene():
    """A tiny raster of horizontal bars with one stem along each bar."""
    pred = np.zeros((64, 64), dtype=np.uint8)
    profile = {"transform": Affine(0.1, 0.0, 0.0, 0.0, -0.1, 0.0)}
    stems = []
    for r0, r1 in ((10, 14), (24, 27), (38, 42), (52, 55)):
        pred[r0:r1, 4:60] = 1
        y = -0.1 * (r0 + r1) / 2.0
        xs = np.linspace(0.6, 5.8, 6)
        path = LineString([(float(x), y) for x in xs])
        stems.append(Stem(start=Point(path.coords[0]),
                          stop=Point(path.coords[-1]), path=path, vector=[],
                          segment_diameter_list=[], segment_length_list=[],
                          segment_volume_list=[]))
    return stems, pred, profile


def _key(stems):
    """imap_unordered does not preserve order; compare order-independently."""
    return sorted((s.path.wkb_hex, tuple(s.segment_diameter_list),
                   tuple(s.segment_length_list),
                   tuple(s.segment_volume_list)) for s in stems)


def _quantify_with_workers(stems, pred, profile, workers):
    cfg = _Cfg()
    cfg.cpu_workers = workers
    return Quant.quantify_stems(copy.deepcopy(stems), pred, profile, cfg)


def test_single_and_multi_worker_agree():
    """Whatever the pool does, one worker and several must agree."""
    stems, pred, profile = _scene()
    one = _quantify_with_workers(stems, pred, profile, 1)
    many = _quantify_with_workers(stems, pred, profile, 4)

    assert len(one) == len(stems)
    assert any(d > 0 for s in one for d in s.segment_diameter_list)
    assert _key(one) == _key(many)


def test_get_diameters_single_and_multi_worker_agree():
    """The diameter stage has its own pool; pin it independently."""
    stems, pred, profile = _scene()
    cfg_one, cfg_many = _Cfg(), _Cfg()
    cfg_many.cpu_workers = 4
    one = Quant.get_diameters(copy.deepcopy(stems), pred, profile, cfg_one)
    many = Quant.get_diameters(copy.deepcopy(stems), pred, profile, cfg_many)

    key = sorted((s.path.wkb_hex, tuple(s.segment_diameter_list))
                 for s in one)
    assert key == sorted((s.path.wkb_hex, tuple(s.segment_diameter_list))
                         for s in many)
