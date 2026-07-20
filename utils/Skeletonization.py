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

# refine_skeleton_segments slices a window extending 5 px beyond each
# part's bounding box. Padding the raster by exactly this margin keeps
# every window a plain in-bounds view (identical shape and content to the
# old full-padding windows, whose extra area was constant False anyway).
_REFINE_MARGIN = 5


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
    # The historical max_tree_height pad (~801 px per side at 5 cm GSD) was
    # constant False: no raster operation ever needed the pixels, they only
    # shifted coordinates by `padding`, which restore_geoinformation
    # subtracts again (docs/CODE_REVIEW_2.md C-1). Pad physically by just
    # the 5 px refine-window margin and keep the remaining
    # (padding - _REFINE_MARGIN) as a pure coordinate offset, applied when
    # parts are built -- emitted coordinates stay in the exact frame the
    # old code produced, so downstream (and the coordinate-hash-driven
    # set iteration orders) are bit-identical, while skeletonize and node
    # finding run on 2-6x less area.
    pred = np.pad(
        pred,
        ((_REFINE_MARGIN, _REFINE_MARGIN), (_REFINE_MARGIN, _REFINE_MARGIN)),
        'constant',
        constant_values=False
    )
    # Decision note: when padding < _REFINE_MARGIN (GSD coarser than
    # ~max_tree_height/4 m/px, i.e. >= 10 m/px at the default 40 m —
    # far beyond any real orthomosaic), coord_offset goes negative. The
    # emitted frame coordinates stay consistent (frame = window +
    # coord_offset maps real-tile pixel (r, c) to (r + padding,
    # c + padding) exactly as before), but the refine windows diverge
    # from the historical code, which produced negative slice starts
    # (Python wraparound = effectively broken windows) in that regime.
    # We deliberately keep the valid windows rather than clamping with
    # max(coord_offset, 0) to reproduce the historical bug.
    coord_offset = padding - _REFINE_MARGIN

    pred = _as_binary_mask(pred)

    skel = morphology.skeletonize(pred)

    t.stop()
    print("#######################################################")
    print("")

    end_nodes, skel = get_nodes(skel)
    segments, skel = find_skeleton_segments(
        skel, end_nodes, math.floor(min_length / 4),
        config=config, coord_offset=coord_offset
    )
    measuring_point_spacing = math.floor(
        min(config.min_length, config.measuring_point_spacing_m) / px_size)

    segments = refine_skeleton_segments(
        segments, skel,
        measuring_point_spacing,
        min_length, config=config, coord_offset=coord_offset
    )

    return segments


# get nodes
def get_nodes(skel: np.ndarray) -> Tuple[List[Tuple[int, int]], Any]:
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
    skel = morphology.skeletonize(skel)  # TODO is this code correct?
    print("Branch points removed: ", bp_count)
    print("Detected end nodes: ", len(end_nodes))
    t.stop()
    print("#######################################################")
    print("")
    return end_nodes, skel


# Remove "dense" (2x2 or larger) regions in the skeleton.
def remove_dense_skeleton_nodes(skel: np.ndarray) -> Tuple[ndarray, int]:
    dense_nodes = morphology.binary_erosion(
        np.pad(skel, 1),
        np.ones((2, 2))
    )[1:-1, 1:-1]
    # Only the number of dense regions is needed; the old center_of_mass
    # call computed one centroid per region (an O(image) weighted scan)
    # just to take len() of the result, which equals num_features.
    _, num_features = scipy.ndimage.measurements.label(dense_nodes)

    skel[np.where(dense_nodes.__eq__(True))] = False
    return skel, int(num_features)


def find_skeleton_nodes(
    skel: np.ndarray
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:

    print("Find skeletion nodes")

    # Pad the skeleton array (same as in the numpy version)
    skel = np.pad(skel, 1, mode='constant', constant_values=0)

    # Extract 8-neighbors using slicing"
    p2 = skel[:-2, 1:-1]
    p3 = skel[:-2, 2:]
    p4 = skel[1:-1, 2:]
    p5 = skel[2:, 2:]
    p6 = skel[2:, 1:-1]
    p7 = skel[2:, :-2]
    p8 = skel[1:-1, :-2]
    p9 = skel[:-2, :-2]
    p1 = skel[1:-1, 1:-1]

    # Binary skeleton mask
    mask = p1 == 1

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
    endpoint_mask = (transitions == 1) & mask
    branchpoint_mask = (transitions >= 3) & mask

    # Get coordinates (remove padding offset)
    endpoints = np.argwhere(endpoint_mask)
    branchpoints = np.argwhere(branchpoint_mask)

    # Convert to CPU tuples
    endpoints = [tuple(map(int, p)) for p in endpoints]
    branchpoints = [tuple(map(int, p)) for p in branchpoints]

    return endpoints, branchpoints


def remove_branchpoints_from_skel(skel, branchpoints):
    print("Remove branch points")
    skel_arr = np.asarray(skel, dtype=bool)
    branchpoints_arr = np.asarray(branchpoints)

    mask = np.zeros_like(skel_arr, dtype=bool)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            xs = branchpoints_arr[:, 0] + dx
            ys = branchpoints_arr[:, 1] + dy
            xs = np.clip(xs, 0, skel_arr.shape[0] - 1)
            ys = np.clip(ys, 0, skel_arr.shape[1] - 1)
            mask[xs, ys] = True

    skel_arr[mask] = False
    return skel_arr


def _neighbor_degree(skel: np.ndarray) -> np.ndarray:
    p = np.pad(skel.astype(np.uint8), 1, mode='constant', constant_values=0)
    deg = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] + p[1:-1, 2:] +
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    return deg


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


def _shift_part(part: Part, off: int) -> Part:
    """Translate a Part from array coordinates into the historical
    full-padding frame (all coordinates + off, both axes).

    Part hashes -- and with them the iteration order of the part sets that
    drive refine and stem connection -- derive from these coordinates, so
    the shift must happen before any set is materialized.
    """
    if off == 0:
        return part
    part.start = (part.start[0] + off, part.start[1] + off)
    part.stop = (part.stop[0] + off, part.stop[1] + off)
    part.path = [(r + off, c + off) for r, c in part.path]
    part.l_bound = (part.l_bound[0] + off, part.l_bound[1] + off)
    part.u_bound = (part.u_bound[0] + off, part.u_bound[1] + off)
    return part


def _neighbors_from_bytes(x: int, y: int, flat: bytes, h: int,
                          w: int) -> List[Tuple[int, int]]:
    """get_neighbors against a C-order bytes snapshot of the skeleton.

    The trace phase never mutates the skeleton, so find_skeleton_segments
    freezes it once via tobytes(); indexing bytes yields a plain int with
    none of numpy's per-element dispatch cost. Same offset scan order as
    get_neighbors, so the returned neighbour order is unchanged.
    """
    neighbors = []
    for dx, dy in _NEIGHBOR_OFFSETS:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < h and 0 <= ny < w and flat[nx * w + ny]:
            neighbors.append((nx, ny))
    return neighbors


def _trace_chain(start: Tuple[int, int], neighbor: Tuple[int, int],
                 flat: bytes, h: int, w: int, node_set: set,
                 visited_edges: set) -> List[Tuple[int, int]]:

    path = [start, neighbor]
    visited_edges.add(_edge_key(start, neighbor))
    prev = start
    curr = neighbor

    while True:
        if curr in node_set and curr != start:
            break
        # Inlined neighbour scan: nxt is the FIRST neighbour != prev in
        # offset order (== nxts[0] of the old list build); a second one
        # makes the interior ambiguous and stops the chain, exactly like
        # the old len(nxts) > 1 test.
        x, y = curr
        nxt = None
        ambiguous = False
        for dx, dy in _NEIGHBOR_OFFSETS:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < h and 0 <= ny < w and flat[nx * w + ny]:
                cand = (nx, ny)
                if cand == prev:
                    continue
                if nxt is None:
                    nxt = cand
                else:
                    ambiguous = True
                    break
        if nxt is None:
            break
        if ambiguous:
            # ambiguous interior -> stop chain here
            break
        ek = _edge_key(curr, nxt)
        if ek in visited_edges:
            break
        visited_edges.add(ek)
        path.append(nxt)
        prev, curr = curr, nxt
        if curr == start:
            break

    return path


def _trace_loop(seed: Tuple[int, int], flat: bytes, h: int, w: int,
                visited_edges: set) -> List[Tuple[int, int]]:

    nbrs = _neighbors_from_bytes(seed[0], seed[1], flat, h, w)
    if not nbrs:
        return []
    start = seed
    prev = start
    curr = nbrs[0]
    visited_edges.add(_edge_key(start, curr))
    path = [start, curr]

    while True:
        # First neighbour != prev in offset order (== nxts[0] before).
        x, y = curr
        nxt = None
        for dx, dy in _NEIGHBOR_OFFSETS:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < h and 0 <= ny < w and flat[nx * w + ny]:
                cand = (nx, ny)
                if cand != prev:
                    nxt = cand
                    break
        if nxt is None:
            break
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
        config=None,
        coord_offset: int = 0
) -> (List[Part], np.ndarray):
    t = Timer()
    t.start()
    print("#######################################################")
    print("Find connected segments in the skeleton")
    print("Initial length of skeleton: ", np.count_nonzero(skel))
    print("Number of end nodes", len(end_nodes))
    print("Minimum length in pixel: ", min_length)

    skel_bool = np.ascontiguousarray(np.asarray(skel, dtype=bool))
    out_skel = np.zeros_like(skel_bool, dtype=bool)
    visited_edges = set()
    parts = []

    deg = _neighbor_degree(skel_bool)
    node_mask = skel_bool & (deg != 2)
    node_coords = [tuple(map(int, p)) for p in np.argwhere(node_mask)]
    node_set = set(node_coords)

    # The trace phase reads the skeleton but never writes it: freeze it
    # once as C-order bytes so the walks index plain ints instead of
    # paying numpy scalar dispatch per pixel probe.
    height, width = skel_bool.shape
    flat = skel_bool.tobytes()

    for node in node_coords:
        nbrs = _neighbors_from_bytes(node[0], node[1], flat, height, width)
        for nb in nbrs:
            ek = _edge_key(node, nb)
            if ek in visited_edges:
                continue
            path = _trace_chain(node, nb, flat, height, width,
                                node_set, visited_edges)
            part = _build_part_from_path(path, min_length)
            if part is not None:
                for rr, cc in part.path:
                    out_skel[rr, cc] = True
                parts.append(_shift_part(part, coord_offset))

    # handle loops or isolated remnants without degree!=2 nodes
    remaining = [tuple(map(int, p))
                 for p in np.argwhere(skel_bool & (~out_skel))]
    for seed in remaining:
        if out_skel[seed]:
            continue
        path = _trace_loop(seed, flat, height, width, visited_edges)
        part = _build_part_from_path(path, min_length)
        if part is not None:
            for rr, cc in part.path:
                out_skel[rr, cc] = True
            parts.append(_shift_part(part, coord_offset))

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
                             config=None,
                             coord_offset: int = 0) -> (List[Part], np.ndarray):
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

    # Part bounds live in the historical full-padding frame; the skeleton
    # array is padded by only _REFINE_MARGIN. coord_offset maps between
    # the two: window rows/cols = frame coordinate - coord_offset. Window
    # origin (low_bounds) stays a frame coordinate, so
    # refine_skeleton_segment and every emitted coordinate are unchanged.
    workers = min(_worker_count(config), max(len(parts), 1))
    if workers <= 1 or len(parts) <= 1:
        for part in parts:
            low_bounds = (part.l_bound[0] - 5, part.l_bound[1] - 5)
            up_bounds = (part.u_bound[0] + 5, part.u_bound[1] + 5)
            sub_skel = skel[
                low_bounds[0] - coord_offset:
                up_bounds[0] - coord_offset + 1,
                low_bounds[1] - coord_offset:
                up_bounds[1] - coord_offset + 1
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
                    low_bounds[0] - coord_offset:
                    up_bounds[0] - coord_offset + 1,
                    low_bounds[1] - coord_offset:
                    up_bounds[1] - coord_offset + 1
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
        # Pixels consumed from skel since the last accepted measuring
        # point. Replaces a np.full(skel.shape, ...) snapshot per accepted
        # point plus whole-window np.where restores: the list is O(steps
        # walked), the arrays were O(window area). Restoring iterates the
        # same pixel set the boolean mask held, so skel ends up identical.
        # (The old code also re-allocated temp on branches that exit the
        # loop immediately, e.g. after the split-at-stop restore -- those
        # allocations were dead and have no list equivalent here.)
        visited = []
        while w != z:
            x, y = w
            skel[(x, y)] = False
            visited.append((x, y))
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
                        for px in visited:
                            skel[px] = True
                        split_ = split_ + 1
                    else:
                        parts[0].path.extend([w])
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
                                visited = []
                        else:
                            if angle > 30:
                                new_part = Part(n, parts[0].stop,
                                                [n, parts[0].stop],
                                                low_bounds, up_bounds)
                                parts.append(new_part)
                                parts[0].stop = n
                                for px in visited:
                                    skel[px] = True
                                z = w
                                split_ = split_ + 1
                            else:
                                parts[0].path.extend([w])
                                p_last = p_recent
                                n = w
                                visited = []
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
