#!/usr/bin/env python

################################################################################
"""Imports"""

import math
import multiprocessing as mp
from typing import Any, List, Tuple

import numpy as np
import scipy.ndimage.measurements
from numpy import ndarray
from skimage import morphology

from classes.Part import Part
from classes.Timer import Timer
from utils.Geometry import ang


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
"""Skeleton operations"""


def find_segments(pred, config, profile) -> (List[Part], List[Tuple[int]]):
    t = Timer()
    t.start()

    print("#######################################################")
    print("Skeletonize Image")

    px_size = abs(profile['transform'][0])
    min_length = math.floor((config.min_length / 4) / px_size)
    padding = int(config.max_tree_height / px_size) + 1

    # Coordinates downstream (restore_geoinformation) are in the frame of
    # the tile padded by `padding` on every side. That frame used to be
    # materialised: a 4916 px tile became 7102 px, 50M pixels, and every
    # stage ran over all of them although well under 1% are live. Now the
    # frame is only a coordinate offset. The work happens on a crop around
    # the foreground, and the parts come out in the padded frame -- an
    # integer shift, hence exact.
    pred = _as_binary_mask(pred)
    padded_shape = (pred.shape[0] + 2 * padding, pred.shape[1] + 2 * padding)
    cropped = _crop_for_skeleton(pred, padding)
    if cropped is None:
        t.stop()
        return []
    crop, offset = cropped

    skel = _skeletonize_sparse(crop)

    t.stop()
    print("#######################################################")
    print("")

    end_nodes, skel = get_nodes(skel)
    segments, skel = find_skeleton_segments(
        skel, end_nodes, math.floor(min_length / 4),
        padding, config=config, offset=offset, out_shape=padded_shape,
    )
    measuring_point_spacing = math.floor(
        min(config.min_length, config.measuring_point_spacing_m) / px_size)

    segments = refine_skeleton_segments(
        segments, skel,
        measuring_point_spacing,
        min_length, config=config
    )

    return segments


# get nodes
#: Rows/cols kept around the foreground bounding box. Every operation in
#: get_nodes is local -- a 2x2 erosion and 3x3 neighbour counts -- so a
#: handful of rows is ample; 8 is far more than any of them reach.
_NODE_MARGIN = 8


def _foreground_bbox(skel: np.ndarray, margin: int = _NODE_MARGIN):
    """Slices around the live pixels, or None when the skeleton is empty."""
    ys, xs = np.nonzero(skel)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(skel.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(skel.shape[1], int(xs.max()) + margin + 1)
    return y0, y1, x0, x1


def _crop_for_skeleton(mask: np.ndarray, padding: int,
                       margin: int = _NODE_MARGIN):
    """The foreground of `mask` with `margin` zero pixels on every side,
    and the offset that maps crop coordinates into the frame of the tile
    padded by `padding` -- or None when there is no foreground.

    Every stage of the skeleton pipeline is local (3x3 neighbourhoods, a
    2x2 erosion, tracing along skeleton pixels), so nothing further than
    one pixel from the foreground can influence a result. The margin is
    kept at the tile edge as well, by padding the crop, so no stage ever
    sees an array boundary where the padded frame had zeros.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    cy0, cy1 = max(0, y0 - margin), min(mask.shape[0], y1 + margin)
    cx0, cx1 = max(0, x0 - margin), min(mask.shape[1], x1 + margin)
    sub = mask[cy0:cy1, cx0:cx1]
    pad_t, pad_b = margin - (y0 - cy0), margin - (cy1 - y1)
    pad_l, pad_r = margin - (x0 - cx0), margin - (cx1 - x1)
    crop = np.pad(sub, ((pad_t, pad_b), (pad_l, pad_r)))
    offset = (cy0 - pad_t + padding, cx0 - pad_l + padding)
    return crop, offset


_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def _skeletonize_sparse(mask: np.ndarray) -> np.ndarray:
    """morphology.skeletonize, applied per connected component.

    Thinning removes a pixel based on its 3x3 neighbourhood only, and two
    8-connected components share no neighbourhood, so each component
    thins exactly as it would inside the full image -- in the same raster
    order, to the same fixed point. Doing it per component bbox turns a
    pass over the whole (mostly empty) crop into passes over the stems.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.shape, dtype=bool)
    labels, count = scipy.ndimage.label(mask, structure=_EIGHT_CONNECTED)
    if count == 0:
        return out
    for index, region in enumerate(scipy.ndimage.find_objects(labels), 1):
        component = labels[region] == index
        thinned = morphology.skeletonize(np.pad(component, 1))[1:-1, 1:-1]
        out[region] |= thinned
    return out


def _gather_neighbours(skel: np.ndarray):
    """Row-major coordinates of the skeleton pixels and their eight
    neighbours, read from a zero-padded copy so every gather is in range.

    Returns (ys, xs, p) with p[k] the neighbour in position k of the
    clockwise-from-north ordering p2..p9 used by find_skeleton_nodes:
    N, NE, E, SE, S, SW, W, NW.
    """
    padded = np.pad(skel, 1, mode='constant', constant_values=0)
    ys, xs = np.nonzero(skel)
    y, x = ys + 1, xs + 1
    p = (
        padded[y - 1, x], padded[y - 1, x + 1], padded[y, x + 1],
        padded[y + 1, x + 1], padded[y + 1, x], padded[y + 1, x - 1],
        padded[y, x - 1], padded[y - 1, x - 1],
    )
    return ys, xs, p


def get_nodes(skel: np.ndarray) -> Tuple[List[Tuple[int, int]], Any]:
    """Detect end nodes and split the skeleton, working only where there
    is skeleton to work on.

    find_segments pads by max_tree_height (1093 px at R13's 2.9 cm), so a
    4096 tile arrives as 6282x6282 -- 2.35x the pixels -- and typically
    holds ~5k live pixels in 39M, i.e. 0.01% occupancy. Every stage below
    is a dense array pass, so all of them paid for the padding and the
    emptiness. Cropping to the foreground bounding box first measured
    3.3x here (4158 -> 1253 ms) with the output arrays and end-node sets
    BIT-IDENTICAL, because every operation involved is local and the
    margin exceeds their reach.
    """
    box = _foreground_bbox(skel)
    if box is None:
        return [], skel
    y0, y1, x0, x1 = box
    if (y1 - y0, x1 - x0) != skel.shape:
        sub_nodes, sub_skel = get_nodes(skel[y0:y1, x0:x1].copy())
        out = np.zeros_like(skel)
        out[y0:y1, x0:x1] = sub_skel
        return [(int(a) + y0, int(b) + x0) for (a, b) in sub_nodes], out

    t = Timer()
    t.start()
    print("#######################################################")
    print("Splitting the skeleton into segments and detecting endnodes")

    skel, dn_count = remove_dense_skeleton_nodes(skel)

    print("Dense nodes removed: ", dn_count)
    t.stop()
    t.start()
    end_nodes, branch_points = find_skeleton_nodes(skel)
    bp_count = len(branch_points)
    while len(branch_points) > 0:
        skel = remove_branchpoints_from_skel(skel, branch_points)
        end_nodes, branch_points = find_skeleton_nodes(skel)
        bp_count = bp_count + len(branch_points)
    skel = _skeletonize_sparse(skel)
    print("Branch points removed: ", bp_count)
    print("Detected end nodes: ", len(end_nodes))
    t.stop()
    print("#######################################################")
    print("")
    return end_nodes, skel


# Remove "dense" (2x2 or larger) regions in the skeleton.
def remove_dense_skeleton_nodes(skel: np.ndarray) -> Tuple[ndarray, int]:
    # This is binary_erosion(np.pad(skel, 1), ones((2, 2)))[1:-1, 1:-1]:
    # with scipy's origin for an even structure (centre index 1) a pixel
    # is dense when it and its N, W and NW neighbours are all skeleton.
    # Evaluated at the skeleton pixels only instead of over the array.
    sk = np.asarray(skel, dtype=bool)
    ys, xs = np.nonzero(sk)
    inner = (ys > 0) & (xs > 0)
    yi, xi = ys[inner], xs[inner]
    dense = np.zeros(ys.size, dtype=bool)
    dense[inner] = sk[yi - 1, xi] & sk[yi, xi - 1] & sk[yi - 1, xi - 1]
    dy, dx = ys[dense], xs[dense]

    # `count` is only ever printed: the number of 4-connected dense
    # regions, labelled over their own bounding box rather than the tile.
    count = 0
    if dy.size:
        y0, x0 = int(dy.min()), int(dx.min())
        local = np.zeros((int(dy.max()) - y0 + 1, int(dx.max()) - x0 + 1),
                         dtype=bool)
        local[dy - y0, dx - x0] = True
        count = int(scipy.ndimage.label(local)[1])

    skel[dy, dx] = False
    return skel, count


def find_skeleton_nodes(
    skel: np.ndarray
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:

    print("Find skeletion nodes")

    # The transition count A(p1) of Zhang-Suen, evaluated at the skeleton
    # pixels only. The former version built the eight neighbour planes of
    # the whole array (ten full passes) to classify well under 1% of it.
    # Same test, same row-major order as np.argwhere gave.
    ys, xs, (p2, p3, p4, p5, p6, p7, p8, p9) = _gather_neighbours(
        np.asarray(skel, dtype=bool))

    # A(p1) calculation (transition count)
    transitions = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + \
                  ((p3 == 0) & (p4 == 1)) + \
                  ((p4 == 0) & (p5 == 1)) + \
                  ((p5 == 0) & (p6 == 1)) + \
                  ((p6 == 0) & (p7 == 1)) + \
                  ((p7 == 0) & (p8 == 1)) + \
                  ((p8 == 0) & (p9 == 1)) + \
                  ((p9 == 0) & (p2 == 1))

    # Endpoint: A(p1) == 1, Branchpoint: A(p1) >= 3
    endpoint_sel = transitions == 1
    branchpoint_sel = transitions >= 3

    endpoints = [(int(y), int(x)) for y, x in
                 zip(ys[endpoint_sel], xs[endpoint_sel])]
    branchpoints = [(int(y), int(x)) for y, x in
                    zip(ys[branchpoint_sel], xs[branchpoint_sel])]

    return endpoints, branchpoints


def remove_branchpoints_from_skel(skel, branchpoints):
    print("Remove branch points")
    skel_arr = np.asarray(skel, dtype=bool)
    branchpoints_arr = np.asarray(branchpoints)
    if branchpoints_arr.size == 0:
        return skel_arr

    # Clear the 3x3 block around each branch point directly; building a
    # full-size mask first was two passes over the array per call.
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            xs = branchpoints_arr[:, 0] + dx
            ys = branchpoints_arr[:, 1] + dy
            xs = np.clip(xs, 0, skel_arr.shape[0] - 1)
            ys = np.clip(ys, 0, skel_arr.shape[1] - 1)
            skel_arr[xs, ys] = False

    return skel_arr


def _edge_key(a: Tuple[int, int], b: Tuple[int, int]):
    return (a, b) if a <= b else (b, a)


def _build_part_from_path(path: List[Tuple[int, int]], min_length: int):
    if path is None or len(path) < 2:
        return None
    if len(path) - 1 < min_length:
        return None
    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    l_bound = (min(rows), min(cols))
    u_bound = (max(rows), max(cols))
    start = path[0]
    stop = path[-1]
    if start[0] > stop[0]:
        path = list(reversed(path))
        start, stop = stop, start
    return Part(start, stop, path, l_bound, u_bound)


def _trace_chain(start: Tuple[int, int], neighbor:
                 Tuple[int, int], skel: np.ndarray, node_set:
                 set, visited_edges: set) -> List[Tuple[int, int]]:

    path = [start, neighbor]
    visited_edges.add(_edge_key(start, neighbor))
    prev = start
    curr = neighbor

    while True:
        if curr in node_set and curr != start:
            break
        nbrs = get_neighbors(curr[0], curr[1], skel)
        nxts = [n for n in nbrs if n != prev]
        if len(nxts) == 0:
            break
        if len(nxts) > 1:
            # ambiguous interior -> stop chain here
            break
        nxt = nxts[0]
        ek = _edge_key(curr, nxt)
        if ek in visited_edges:
            break
        visited_edges.add(ek)
        path.append(nxt)
        prev, curr = curr, nxt
        if curr == start:
            break

    return path


def _trace_loop(seed: Tuple[int, int], skel:
                np.ndarray, visited_edges: set) -> List[Tuple[int, int]]:

    nbrs = get_neighbors(seed[0], seed[1], skel)
    if not nbrs:
        return []
    start = seed
    prev = start
    curr = nbrs[0]
    visited_edges.add(_edge_key(start, curr))
    path = [start, curr]

    while True:
        nbrs = get_neighbors(curr[0], curr[1], skel)
        nxts = [n for n in nbrs if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        ek = _edge_key(curr, nxt)
        if ek in visited_edges:
            break
        visited_edges.add(ek)
        path.append(nxt)
        prev, curr = curr, nxt
        if curr == start:
            break

    return path


def find_skeleton_segments(
        skel: np.ndarray,
        end_nodes: List[Tuple[int]],
        min_length: int,
        padding: int,
        config=None,
        offset: Tuple[int, int] = (0, 0),
        out_shape: Tuple[int, int] = None,
) -> (List[Part], np.ndarray):
    """Trace the skeleton into parts.

    `skel` may be a crop: `offset` is where its origin sits in the output
    frame, and `out_shape` the size of that frame. The parts come back in
    the output frame (an integer shift of every path pixel, so exact) and
    the returned skeleton is the output-frame array with the traced pixels
    set -- what refine_skeleton_segments slices its sub-windows from.
    """
    t = Timer()
    t.start()
    print("#######################################################")
    print("Find connected segments in the skeleton")
    print("Initial length of skeleton: ", np.count_nonzero(skel))
    print("Number of end nodes", len(end_nodes))
    print("Minimum length in pixel: ", min_length)

    skel_bool = np.asarray(skel, dtype=bool)
    oy, ox = int(offset[0]), int(offset[1])
    h, w = skel_bool.shape
    out_skel = np.zeros(out_shape or skel_bool.shape, dtype=bool)
    out_view = out_skel[oy:oy + h, ox:ox + w]     # crop frame, shares memory
    visited_edges = set()
    parts = []

    # Degree != 2 marks a node. Counted at the skeleton pixels only, in
    # the row-major order np.argwhere over a full mask would have given.
    ys, xs, planes = _gather_neighbours(skel_bool)
    deg = np.zeros(ys.size, dtype=np.uint8)
    for plane in planes:
        deg += plane
    is_node = deg != 2
    node_coords = [(int(y), int(x)) for y, x in
                   zip(ys[is_node], xs[is_node])]
    node_set = set(node_coords)

    def _keep(path):
        # Parts live in the output frame; the traced pixels mark the crop.
        part = _build_part_from_path(
            [(r + oy, c + ox) for r, c in path], min_length)
        if part is not None:
            parts.append(part)
            for rr, cc in path:
                out_view[rr, cc] = True

    for node in node_coords:
        nbrs = get_neighbors(node[0], node[1], skel_bool)
        for nb in nbrs:
            ek = _edge_key(node, nb)
            if ek in visited_edges:
                continue
            _keep(_trace_chain(node, nb, skel_bool, node_set, visited_edges))

    # handle loops or isolated remnants without degree!=2 nodes
    remaining = [tuple(map(int, p))
                 for p in np.argwhere(skel_bool & (~out_view))]
    for seed in remaining:
        if out_view[seed]:
            continue
        _keep(_trace_loop(seed, skel_bool, visited_edges))

    skeleton_parts = set(parts)
    print("Detected skeleton segments: ", len(skeleton_parts))
    t.stop()
    print("#######################################################")
    print("")
    return skeleton_parts, out_skel


# Parallel version of refine_skeleton_segments
# Find stem parts between nodes using the connectivity in the skeleton.
def refine_skeleton_segments(parts: List[Part], skel: np.ndarray,
                             measuring_point_spacing: int, min_length: int,
                             config=None) -> (List[Part], np.ndarray):
    t = Timer()
    t.start()
    split = 0
    out = 0
    refined_parts = []

    def return_callback(result):
        refined_part, s, o = result
        nonlocal split
        nonlocal out
        # nonlocal refined_parts
        split = split + s
        out = out + o
        if refined_part is not None:
            for refined in refined_part:
                refined_parts.append(refined)

    def error_callback(error):
        print(error, flush=True)

    print("#######################################################")
    print("#Refining and sorting out skeleton segments")
    print("Initial length of skeleton: ", np.count_nonzero(skel))
    print("Number of initial skeleton segments", len(parts))

    workers = min(_worker_count(config), max(len(parts), 1))
    if workers <= 1 or len(parts) <= 1:
        for part in parts:
            low_bounds = (part.l_bound[0] - 5, part.l_bound[1] - 5)
            up_bounds = (part.u_bound[0] + 5, part.u_bound[1] + 5)
            sub_skel = skel[
                low_bounds[0]:up_bounds[0] + 1,
                low_bounds[1]:up_bounds[1] + 1
            ]
            return_callback(refine_skeleton_segment(
                part, low_bounds, up_bounds, sub_skel,
                measuring_point_spacing, min_length
            ))
    else:
        with mp.Pool(workers) as pool:
            r = []
            for part in parts:
                low_bounds = (part.l_bound[0] - 5, part.l_bound[1] - 5)
                up_bounds = (part.u_bound[0] + 5, part.u_bound[1] + 5)
                sub_skel = skel[
                    low_bounds[0]:up_bounds[0] + 1,
                    low_bounds[1]:up_bounds[1] + 1
                ]
                r.append(pool.apply_async(refine_skeleton_segment, args=(
                    part, low_bounds, up_bounds, sub_skel,
                    measuring_point_spacing, min_length
                ), callback=return_callback, error_callback=error_callback))
            for r_ in r:
                r_.wait()

    print("Number of split segments:", split)
    print("Number of removed segments:", out)
    print("Number of refined segments:", len(refined_parts))

    t.stop()
    print("#######################################################")
    print("")
    return refined_parts


def refine_skeleton_segment(part: Part, low_bounds: Tuple[int, int],
                            up_bounds: Tuple[int, int],
                            skel: np.ndarray, measuring_point_spacing: int,
                            min_length: int) -> List[Part]:
    part.start = (part.start[0] - low_bounds[0], part.start[1] - low_bounds[1])
    part.stop = (part.stop[0] - low_bounds[0], part.stop[1] - low_bounds[1])
    part.path = [part.start, part.stop]
    refined_parts_ = []
    parts = [part]
    out_ = 0
    split_ = 0
    while len(parts) > 0:
        w = parts[0].start
        n = parts[0].start
        z = parts[0].stop
        p_last = [parts[0].start, parts[0].stop]
        parts[0].path = []
        parts[0].path.extend([w])
        # Pixels cleared from `skel` since the last measuring point, so a
        # split can put them back. This was a full-size boolean mask
        # reallocated at EVERY step along the path -- the part's whole
        # bounding window, refilled per pixel walked.
        cleared = []
        while w != z:
            x, y = w
            skel[(x, y)] = False
            cleared.append((x, y))
            ww = get_neighbors(x, y, skel)
            if ww:
                w = ww[0]
                p_recent = [n, w]
                angle = ang(p_recent, p_last)
                if w == z:
                    if angle > 10:
                        new_part = Part(n, parts[0].stop,
                                        [n, parts[0].stop],
                                        low_bounds, up_bounds)
                        parts.append(new_part)
                        parts[0].stop = n
                        for px in cleared:
                            skel[px] = True
                        cleared = []
                        split_ = split_ + 1
                    else:
                        parts[0].path.extend([w])
                        cleared = []
                else:
                    if math.dist(n, w) > measuring_point_spacing:
                        if n == parts[0].start:
                            if angle > 10:
                                new_part = Part(w, parts[0].stop,
                                                [w, parts[0].stop],
                                                low_bounds, up_bounds)
                                parts.append(new_part)
                                parts[0].stop = w
                                parts[0].path.extend([w])
                                split_ = split_ + 1
                                z = w
                            else:
                                parts[0].path.extend([w])
                                p_last = p_recent
                                n = w
                                cleared = []
                        else:
                            if angle > 30:
                                new_part = Part(n, parts[0].stop,
                                                [n, parts[0].stop],
                                                low_bounds, up_bounds)
                                parts.append(new_part)
                                parts[0].stop = n
                                for px in cleared:
                                    skel[px] = True
                                z = w
                                split_ = split_ + 1
                            else:
                                parts[0].path.extend([w])
                                p_last = p_recent
                                n = w
                                cleared = []
            else:
                parts[0].path.extend([(x, y)])
                parts[0].stop = (x, y)
                z = (x, y)
                w = z

        refined_part_ = Part(parts[0].start, parts[0].stop, parts[0].path,
                             low_bounds, up_bounds)
        parts.pop(0)

        if math.dist(refined_part_.start, refined_part_.stop) >= min_length:
            refined_part_.start = (refined_part_.start[0] + low_bounds[0],
                                   refined_part_.start[1] + low_bounds[1])
            refined_part_.stop = (refined_part_.stop[0] + low_bounds[0],
                                  refined_part_.stop[1] + low_bounds[1])
            for i in range(len(refined_part_.path)):
                refined_part_.path[i] = (
                    refined_part_.path[i][0] + low_bounds[0],
                    refined_part_.path[i][1] + low_bounds[1]
                )
            if refined_part_.start[0] > refined_part_.stop[0]:
                refined_part_ = Part(refined_part_.stop, refined_part_.start,
                                     refined_part_.path, low_bounds, up_bounds)
                refined_part_.path.reverse()
            refined_parts_.append(refined_part_)
        else:
            out_ = out_ + 1

    if len(refined_parts_) == 0:
        return None, split_, out_
    return refined_parts_, split_, out_


# Scanned in the same order as the previous offset array, so the returned
# neighbour order is unchanged.
_NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def get_neighbors(x: int, y: int, skel: np.ndarray) -> List[Tuple[int, int]]:
    """The 8-connected skeleton neighbours of (x, y).

    Deliberately plain Python. The previous implementation built an offset
    array, an added coordinate array, a bounds mask and two fancy-index results
    on EVERY call -- roughly five numpy allocations to inspect eight pixels.
    Profiled at 317,506 calls on one 4096 px tile, that made this the largest
    pure-Python cost in the vector stage: numpy's per-call dispatch overhead
    dwarfs the work when the array has eight elements.
    """
    h, w = skel.shape
    neighbors = []
    for dx, dy in _NEIGHBOR_OFFSETS:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < h and 0 <= ny < w and skel[nx, ny]:
            neighbors.append((nx, ny))
    return neighbors
