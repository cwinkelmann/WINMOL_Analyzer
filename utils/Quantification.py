#!/usr/bin/env python

################################################################################
"""Imports"""

import math
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import rasterio.features
import scipy.ndimage as ndi
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point

from classes.Stem import Stem
from classes.Timer import Timer
from utils.Geometry import create_vector

# System epsilon
epsilon = np.finfo(float).eps


def _as_binary_mask(pred):
    arr = np.asarray(pred)
    if arr.dtype == np.bool_:
        return arr
    if arr.dtype == np.uint8 and arr.size and arr.max() <= 1:
        return arr.astype(bool, copy=False)
    return arr >= 0.5


def _worker_count(config=None):
    proc = mp.current_process()
    if proc.name != "MainProcess":
        return 1

    value = getattr(config, 'cpu_workers', None) \
        if config is not None else None
    if value is None:
        value = max(mp.cpu_count() - 1, 1)
    try:
        return max(1, int(value))
    except Exception:
        return 1


################################################################################
"""Stem quantification operations"""


def quantify_stems(stems: List[Stem], pred, profile, config=None):
    t = Timer()
    t.start()

    stems_ = []
    stems__ = []

    print("#######################################################")
    print("Quantifying stems")
    if not stems:
        print("0 measurements of diameters where conducted")
        print("Volume of  0  stems calculated")
        t.stop()
        print("#######################################################")
        print("")
        return []
    stems = get_diameters(stems, pred, profile, config=config)
    workers = min(_worker_count(config), max(len(stems), 1))
    if workers <= 1 or len(stems) <= 1:
        for stem in stems:
            stems_.append(clean_diameter(stem))
        for stem in stems_:
            stems__.append(quantify_stem(stem))
    else:
        # ThreadPool, not mp.Pool. These stages map over individual Stem
        # objects (shapely geometry + diameter lists), so a process pool has to
        # pickle every stem out to a worker and the result back. Measured on
        # one 4096 px tile with 1567 stems: process pool 45.2 s vs 4.6 s
        # serial -- ~10x SLOWER, with the cost flat in worker count (2/4/7 all
        # ~45 s), i.e. pure transport overhead against 4.6 s of real work.
        # Threads share memory, so the transport disappears; measured 4.8 s
        # with byte-identical output. Coarse-grained pools that hand whole
        # tiles to workers (VectorTilePipeline) are left as processes.
        with ThreadPool(workers) as pool:
            for stem in pool.imap_unordered(clean_diameter, stems):
                stems_.append(stem)
            for stem in pool.imap_unordered(quantify_stem, stems_):
                stems__.append(stem)

    print("Volume of ", len(stems__), " stems calculated")
    t.stop()
    print("#######################################################")
    print("")
    return stems__


def get_diameters(stems: List[Stem], pred, profile, config=None):
    transform = profile['transform']
    pred_bin = _as_binary_mask(pred).astype(np.int16, copy=False)

    diameter_method = str(getattr(config, 'diameter_method', 'contour'))\
        .lower() if config is not None else 'contour'

    diam_count = 0
    measured_stems = []

    def return_callback(measured_stem):
        measured_stems.append(measured_stem)
        nonlocal diam_count
        diam_count = diam_count + len(measured_stem.segment_diameter_list)

    def error_callback(error):
        print(error, flush=True)

    workers = min(_worker_count(config), max(len(stems), 1))

    if diameter_method == 'edt':
        edt_map = _distance_transform_m(pred_bin, profile)
        if workers <= 1 or len(stems) <= 1:
            for stem in stems:
                try:
                    return_callback(
                        calc_v_d_edt(stem, edt_map, profile, config=config))
                except Exception as error:
                    error_callback(error)
        else:
            # EDT array pickling can be expensive;
            #  default to serial unless many stems
            for stem in stems:
                try:
                    return_callback(
                        calc_v_d_edt(stem, edt_map, profile, config=config))
                except Exception as error:
                    error_callback(error)
    else:
        mask = None
        pred_shapes_ = (
            {'properties': {'raster_val': value}, 'geometry': geom}
            for geom, value in rasterio.features.shapes(
                pred_bin,
                mask=mask,
                transform=transform,
            )
        )
        pred_shapes = list(pred_shapes_)
        pred_shapes = gpd.GeoDataFrame.from_features(pred_shapes)
        pred_shapes = pred_shapes[pred_shapes['raster_val'] == 1]
        # One STRtree for the whole stage instead of one per calc_d call.
        pred_shapes = ContourIndex(pred_shapes)

        if workers <= 1 or len(stems) <= 1:
            for stem in stems:
                try:
                    return_callback(calc_v_d_contour(
                        stem, pred_shapes, config=config))
                except Exception as error:
                    error_callback(error)
        else:
            # ThreadPool for the same reason as quantify_stems: a process pool
            # would pickle every Stem AND the full contour list to each worker.
            with ThreadPool(workers) as pool:
                r = []
                for stem in stems:
                    r.append(pool.apply_async(
                        calc_v_d_contour,
                        args=(stem, pred_shapes, config),
                        callback=return_callback,
                        error_callback=error_callback))
                for r_ in r:
                    r_.wait()

    print(diam_count, " measurements of diameters where conducted")
    return measured_stems


def quantify_stem(stem: Stem):
    stem.segment_length_list = []
    stem.segment_volume_list = []
    for i in range(0, len(stem.path.coords) - 1):
        seg_l, seg_vol = calc_l_v(
            stem.path.coords[i],
            stem.path.coords[i + 1],
            stem.segment_diameter_list[i],
            stem.segment_diameter_list[i + 1]
        )
        stem.segment_length_list.append(seg_l)
        stem.segment_volume_list.append(seg_vol)
    return stem


# --- Helper functions ---

def _pixel_size(profile) -> Tuple[float, float]:
    return abs(profile['transform'][0]), abs(profile['transform'][4])


def clean_diameter(stem):
    if len(stem.segment_diameter_list) < 2:
        return stem
    if len(stem.segment_diameter_list) == 2:
        return stem
    q1 = np.quantile(stem.segment_diameter_list, 0.25)
    q3 = np.quantile(stem.segment_diameter_list, 0.75)
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    if len(stem.segment_diameter_list) > 4:
        for i in range(1, len(stem.segment_diameter_list) - 2):
            i_uw = stem.segment_diameter_list[i] > uw
            i_lw = stem.segment_diameter_list[i] < lw
            if i_uw or i_lw:
                wd1 = stem.segment_diameter_list[i - 1] * abs(
                    Point(stem.path.coords[i]).distance(Point(
                        stem.path.coords[i + 1])))
                wd2 = stem.segment_diameter_list[i + 1] * abs(
                    Point(stem.path.coords[i - 1]).distance(Point(
                        stem.path.coords[i])))
                d12 = abs(Point(stem.path.coords[i - 1]).distance(
                    Point(stem.path.coords[i + 1])))
                if d12 > epsilon:
                    stem.segment_diameter_list[i] = (wd1 + wd2) / d12
        if (
            stem.segment_diameter_list[0] > uw
            or stem.segment_diameter_list[0] < lw
        ):
            stem.segment_diameter_list[0] = stem.segment_diameter_list[1]

        if (
            stem.segment_diameter_list[-1] > uw
            or stem.segment_diameter_list[-1] < lw
        ):
            stem.segment_diameter_list[-1] = stem.segment_diameter_list[-2]
    return stem


def _local_normal(coords, idx):
    if len(coords) < 2:
        return (0.0, 1.0)
    if idx == 0:
        v = create_vector((coords[0], coords[1]))
    elif idx == len(coords) - 1:
        v = create_vector((coords[-2], coords[-1]))
    else:
        v = create_vector((coords[idx - 1], coords[idx + 1]))
    return (-float(v[1]), float(v[0]))


def _measurement_vector(node_xy, normal_xy, half_len):
    nx, ny = normal_xy
    x, y = node_xy
    p1 = Point(x - nx * half_len, y - ny * half_len)
    p2 = Point(x + nx * half_len, y + ny * half_len)
    return LineString([p1, p2])


def calc_v_d_contour(stem, contours, config=None):
    coords = list(stem.path.coords)
    half_len = float(getattr(config, 'diameter_vector_half_length_m', 1.0)) \
        if config is not None else 1.0
    stem.vector = []
    stem.segment_diameter_list = []

    for i, xy in enumerate(coords):
        normal = _local_normal(coords, i)
        vector = _measurement_vector(xy, normal, half_len)
        stem.segment_diameter_list.append(calc_d(xy, vector, contours))
        stem.vector.append(vector)
    return stem


def _distance_transform_m(pred_bin, profile):
    px, py = _pixel_size(profile)
    return ndi.distance_transform_edt(pred_bin.astype(bool),
                                      sampling=(py, px))


def _xy_to_rowcol(x, y, profile) -> Tuple[int, int]:
    transform = profile['transform']
    inv = ~transform
    col, row = inv * (x, y)
    return int(round(row)), int(round(col))


def calc_v_d_edt(stem, edt_map, profile, config=None):
    coords = list(stem.path.coords)
    default_half = \
        float(getattr(config, 'diameter_vector_half_length_m', 1.0)) \
        if config is not None else 1.0
    clip_max = getattr(config, 'edt_clip_max_m', None) \
        if config is not None else None
    stem.vector = []
    stem.segment_diameter_list = []

    h, w = edt_map.shape
    for i, xy in enumerate(coords):
        row, col = _xy_to_rowcol(xy[0], xy[1], profile)
        if row < 0 or row >= h or col < 0 or col >= w:
            radius = 0.0
        else:
            radius = float(edt_map[row, col])
            if clip_max is not None:
                radius = min(radius, float(clip_max))
        diameter = max(0.0, 2.0 * radius)
        normal = _local_normal(coords, i)
        half_len = max(default_half, radius)
        stem.vector.append(_measurement_vector(xy, normal, half_len))
        stem.segment_diameter_list.append(diameter)
    return stem


# Backward-compatible name
calc_v_d = calc_v_d_contour


class ContourIndex:
    """Contour geometries plus ONE STRtree over them.

    ``GeoDataFrame.sindex`` is documented as a cached spatial index, and calc_d
    was written assuming that. Measured, it is not: one 4096 px tile produced
    **10,595** STRtree constructions, essentially one per calc_d call. That is
    why the original sindex prefilter measured only ~1.3x instead of the
    predicted 40-65% — the index was rebuilt and thrown away every time, so the
    prefilter kept paying for itself.

    Building it once here is what the comment always claimed was happening.
    """

    __slots__ = ("geoms", "tree")

    def __init__(self, contours):
        self.geoms = np.asarray(contours.geometry.values)
        self.tree = STRtree(self.geoms)


def calc_d(node, line, contours):
    node = Point(node)
    d = 0
    # Only intersect the line against polygons whose bounding box overlaps it —
    # a line cannot intersect a polygon whose bbox it misses, so the set of
    # non-empty intersections (and thus d, a max) is identical to intersecting
    # against all polygons. Bit-identical result, fewer GEOS intersections.
    if isinstance(contours, ContourIndex):
        idx = contours.tree.query(line)
        if len(idx) == 0:
            return d
        candidates = contours.geoms[idx]
    else:
        # Backwards-compatible path for callers still passing a GeoDataFrame.
        idx = contours.sindex.query(line)
        if len(idx) == 0:
            return d
        candidates = np.asarray(contours.geometry.iloc[idx].values)

    intersects = shapely.intersection(candidates, line)
    intersects = intersects[~shapely.is_empty(intersects)]

    for i in intersects:
        if node.distance(i) < 0.01:
            if i.geom_type == 'MultiLineString':
                for i_ in i.geoms:
                    if node.distance(i_) < 0.01:
                        d = max(d, i_.length)
            else:
                d = max(d, i.length)
    return d


def calc_l_v(p1, p2, d1, d2):
    length = math.dist(p1, p2)
    v = 1 / 3 * math.pi * (
        (d1 / 2) ** 2 + (d1 / 2) * (d2 / 2) + (d2 / 2) ** 2
    ) * length
    return length, v
