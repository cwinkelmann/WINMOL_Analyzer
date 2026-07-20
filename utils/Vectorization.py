#!/usr/bin/env python

################################################################################
"""Imports"""

import math
import multiprocessing as mp
from typing import List, Sequence, Set

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from shapely.strtree import STRtree

from classes.Part import Part
from classes.Stem import Stem
from classes.Timer import Timer
from utils.Geometry import ang
from utils.IO import get_bounds_from_profile

# System epsilon
epsilon = np.finfo(float).eps


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


def _clone_stem(stem: Stem) -> Stem:
    return Stem(
        start=Point(stem.start.coords[0]),
        stop=Point(stem.stop.coords[0]),
        path=LineString(list(stem.path.coords)),
        vector=list(getattr(stem, 'vector', [])),
        segment_diameter_list=list(getattr(stem, 'segment_diameter_list', [])),
        segment_length_list=list(getattr(stem, 'segment_length_list', [])),
        segment_volume_list=list(getattr(stem, 'segment_volume_list', [])),
        crs=getattr(stem, 'crs', None),
    )


def _stem_end_lines(stem: Stem):
    if len(stem.path.coords) < 4:
        line_start = LineString([stem.path.coords[0], stem.path.coords[-1]])
        line_stop = LineString([stem.path.coords[0], stem.path.coords[-1]])
    else:
        i = len(stem.path.coords) - 2 if len(stem.path.coords) < 8 else 6
        line_start = LineString([stem.path.coords[1], stem.path.coords[i]])
        line_stop = LineString([stem.path.coords[-(i + 1)],
                                stem.path.coords[-2]])
    return line_start, line_stop


def _query_tree_indices(tree: STRtree, geom, fallback_geoms=None):
    try:
        matches = tree.query(geom)
    except Exception:
        return []
    if len(matches) == 0:
        return []
    first = matches[0]
    if isinstance(first, (int, np.integer)):
        return [int(i) for i in matches]
    if fallback_geoms is None:
        return []
    geom_to_idx = {id(g): i for i, g in enumerate(fallback_geoms)}
    return [geom_to_idx[id(g)] for g in matches if id(g) in geom_to_idx]


def _merge_diameter_lists(first: Stem, second: Stem) -> List[float]:
    """Per-node diameters for a merged path built as
    first.path.coords[:-1] + second.path.coords[1:] (the construction used by
    calc_connectivity_votes in every branch).

    Only meaningful when BOTH parents carry one diameter per path node — the
    tiled-merge case, where diameters were measured in-tile before merging.
    In-tile connect_stems runs BEFORE quantification (lists empty), so this
    returns [] there and behavior is unchanged. Without this, a merged stem
    kept the base's (shorter) diameter list and quantify_stem crashed with
    IndexError at merge time (docs/CODE_REVIEW_2.md A-1).
    """
    d_first = list(getattr(first, 'segment_diameter_list', []) or [])
    d_second = list(getattr(second, 'segment_diameter_list', []) or [])
    if (len(d_first) == len(first.path.coords)
            and len(d_second) == len(second.path.coords)):
        return d_first[:-1] + d_second[1:]
    return []


def _remove_duplicates_against_base(
    cycle_stems: Sequence[Stem], remaining: Set[int], base_idx: int) \
        -> tuple[Set[int], int]:
    if base_idx not in remaining:
        return remaining, 0
    base = cycle_stems[base_idx]
    buffer_geom = base.path.buffer(0.3)
    # Bounding-box prefilter (the counterpart of remove_duplicates'
    # STRtree, without building a tree per merge): a geometry contained
    # in the buffer necessarily has its bbox inside the buffer's bbox,
    # so skipping the others cannot change the removed set.
    minx, miny, maxx, maxy = buffer_geom.bounds
    to_remove = set()
    for idx in remaining:
        if idx == base_idx:
            continue
        try:
            path = cycle_stems[idx].path
            p_minx, p_miny, p_maxx, p_maxy = path.bounds
            if (p_minx < minx or p_miny < miny
                    or p_maxx > maxx or p_maxy > maxy):
                continue
            if buffer_geom.contains(path):
                to_remove.add(idx)
        except Exception:
            continue
    if to_remove:
        remaining = set(remaining)
        remaining.difference_update(to_remove)
    return remaining, len(to_remove)


################################################################################
"""Vector operations"""


# Spatial-index accelerated version of connect_stems
def connect_stems(stems: List[Stem], config) -> List[Stem]:
    max_distance = config.max_distance
    max_tree_height = config.max_tree_height
    tolerance_angle = config.tolerance_angle

    t = Timer()
    t.start()
    print("#######################################################")
    print("Gathering stem segments")

    cycle_nbr = 1
    c_count = 0
    out_count = 0
    duplicates_count = 0
    count_stem_parts = len(stems)
    global_change = True

    while global_change:
        global_change = False
        print("Cycle ", cycle_nbr)
        cycle_stems = list(stems)
        if not cycle_stems:
            break

        start_points = [s.start for s in cycle_stems]
        stop_points = [s.stop for s in cycle_stems]
        start_tree = STRtree(start_points)
        stop_tree = STRtree(stop_points)
        remaining = set(range(len(cycle_stems)))
        connected_stems = []

        while remaining:
            base_idx = next(iter(remaining))
            base_stem = cycle_stems[base_idx]

            while True:
                line_start, line_stop = _stem_end_lines(base_stem)
                start_buffer = base_stem.start.buffer(
                    max_distance, resolution=32)
                end_buffer = base_stem.stop.buffer(
                    max_distance, resolution=32)

                candidate_indices = set(
                    _query_tree_indices(stop_tree, start_buffer, stop_points))
                candidate_indices.update(
                    _query_tree_indices(start_tree, end_buffer, start_points))
                candidate_indices.intersection_update(remaining)
                candidate_indices.discard(base_idx)

                # exact endpoint filter
                filtered_indices = []
                for idx in candidate_indices:
                    candidate = cycle_stems[idx]
                    if (
                        start_buffer.contains(candidate.stop)
                        or end_buffer.contains(candidate.start)
                    ):
                        filtered_indices.append(idx)

                # Score first, build later: the vote depends only on
                # endpoint distances and end-line angles, all known
                # before any merged geometry exists. linemerge +
                # _clone_stem run once, for the argmin winner, instead
                # of for every candidate that passes the gates.
                best_vote = math.inf
                best_branch = None
                best_slave_idx = None

                for idx in filtered_indices:
                    scored = _score_connectivity(
                        base_stem,
                        line_start,
                        line_stop,
                        start_buffer,
                        end_buffer,
                        max_distance,
                        max_tree_height,
                        tolerance_angle,
                        cycle_stems[idx],
                    )
                    if scored is not None and scored[0] < best_vote:
                        best_vote, best_branch = scored
                        best_slave_idx = idx

                if best_slave_idx is not None:
                    base_stem = _build_connection(
                        base_stem, cycle_stems[best_slave_idx], best_branch)
                    cycle_stems[base_idx] = base_stem
                    remaining.discard(best_slave_idx)
                    global_change = True
                    c_count += 1
                    remaining, dup_removed = _remove_duplicates_against_base(
                        cycle_stems, remaining, base_idx)
                    duplicates_count += dup_removed
                    continue

                connected_stems.append(base_stem)
                remaining.discard(base_idx)
                break

        stems = connected_stems
        cycle_nbr += 1

    connected_stems = []
    for stem in stems:
        if stem.length > config.min_length:
            connected_stems.append(stem)
        else:
            out_count += 1
    connected_stems, dup_count_2 = remove_duplicates(connected_stems)
    duplicates_count += dup_count_2

    print("")
    print(count_stem_parts, "stem segments analyzed")
    print(c_count, "stem segments appended to other stems")
    print(duplicates_count, "duplicates are removed")
    print(out_count, "stem fragments with a length less than ",
          config.min_length, "m are filtered out")
    print("final number of stems", len(connected_stems))
    t.stop()
    print("#######################################################")
    print("")
    return connected_stems


class _VoteEnds:
    """Endpoint stand-in for the merged candidate inside calc_vote.

    calc_vote reads only candidate.start / candidate.stop, which are known
    before any merged geometry is built: branch 1 keeps stems0.start and
    takes stem.stop, branch 2 mirrored. Feeding these Points through the
    same calc_vote expression yields the identical float."""

    __slots__ = ("start", "stop")

    def __init__(self, start: Point, stop: Point):
        self.start = start
        self.stop = stop


def _score_connectivity(
        stems0: Stem,
        line_start: LineString,
        line_stop: LineString,
        start_buffer,
        end_buffer,
        max_distance,
        max_tree_height,
        tolerance_angle,
        stem: Stem
):
    """Gate checks + vote for appending `stem` to `stems0`, WITHOUT
    building any merged geometry.

    Returns (vote, branch) for the better of the two attachment branches
    (1: stem appended after stems0, 2: stem prepended) or None. Branch
    order and the strict-less tie rule reproduce the old argmin exactly.
    """
    if stem == stems0:
        # if the stems are identical: no change
        return None

    e_line_start, e_line_stop = _stem_end_lines(stem)

    ang_l_sp_el_st = abs(ang(line_stop.coords, e_line_start.coords))
    ang_el_sp_l_st = abs(ang(e_line_stop.coords, line_start.coords))

    best = None
    if end_buffer.contains(stem.start) and ang_l_sp_el_st < tolerance_angle:
        missing_part_ = LineString(
            [stems0.path.coords[-2],
             stem.path.coords[1]]
        )
        dist_f = 1 - (
            1 / (3 + max_distance - stems0.stop.distance(stem.start))
            ** 0.5
        )
        ang_l_sp_mp = abs(ang(line_stop.coords, missing_part_.coords))
        ang_mp_el_st = abs(ang(missing_part_.coords, e_line_start.coords))

        if (ang_l_sp_el_st < (tolerance_angle * dist_f) and ang_l_sp_mp < (
                tolerance_angle * dist_f) and ang_mp_el_st < (
                tolerance_angle * dist_f) and stems0.start.distance(
                stem.stop) < max_tree_height):
            vote = calc_vote(ang_l_sp_el_st, ang_l_sp_mp, ang_mp_el_st,
                             _VoteEnds(stems0.start, stem.stop),
                             stem, stems0, tolerance_angle)
            best = (vote, 1)

    if start_buffer.contains(stem.stop) and ang_el_sp_l_st < tolerance_angle:
        missing_part_ = LineString(
            [stem.path.coords[-2], stems0.path.coords[1]])
        dist_f = 1 - (
            1 / (3 + max_distance - stem.stop.distance(stems0.start))
            ** 0.5
        )
        ang_el_sp_mp = abs(ang(e_line_stop.coords, missing_part_.coords))
        ang_mp_l_st = abs(ang(missing_part_.coords, line_start.coords))

        if (ang_el_sp_l_st < (tolerance_angle * dist_f) and ang_el_sp_mp < (
                tolerance_angle * dist_f) and ang_mp_l_st < (
                tolerance_angle * dist_f) and stem.start.distance(
                stems0.stop) < max_tree_height):
            vote = calc_vote(ang_el_sp_l_st, ang_el_sp_mp, ang_mp_l_st,
                             _VoteEnds(stem.start, stems0.stop),
                             stems0, stem, tolerance_angle)
            # strict less: branch 1 wins ties, like the old first-argmin
            if best is None or vote < best[0]:
                best = (vote, 2)

    return best


def _merge_paths(first: Stem, second: Stem) -> LineString:
    """The merged path first[:-1] + missing part + second[1:], built with
    the exact expressions the old calc_connectivity_votes used."""
    missing_part_ = LineString(
        [first.path.coords[-2], second.path.coords[1]])
    first_long = len(first.path.coords) > 2
    second_long = len(second.path.coords) > 2
    if first_long and second_long:
        start = LineString(first.path.coords[:-1])
        end = LineString(second.path.coords[1:])
        return linemerge([start, missing_part_, end])
    if first_long:
        start = LineString(first.path.coords[:-1])
        return linemerge([start, missing_part_])
    if second_long:
        end = LineString(second.path.coords[1:])
        return linemerge([missing_part_, end])
    return missing_part_


def _build_connection(stems0: Stem, stem: Stem, branch: int) -> Stem:
    """Materialize the merged stem for the winning candidate/branch that
    _score_connectivity chose."""
    candidate = _clone_stem(stems0)
    if branch == 1:
        candidate.path = _merge_paths(stems0, stem)
        candidate.stop = stem.stop
        # merged path = stems0[:-1] + stem[1:]; merge the diameter lists
        # the same way and drop stale per-node measures (A-1)
        candidate.segment_diameter_list = _merge_diameter_lists(stems0, stem)
    else:
        candidate.path = _merge_paths(stem, stems0)
        candidate.start = stem.start
        # merged path = stem[:-1] + stems0[1:] in this branch (A-1)
        candidate.segment_diameter_list = _merge_diameter_lists(stem, stems0)
    candidate.segment_length_list = []
    candidate.segment_volume_list = []
    candidate.vector = []
    return candidate


def calc_connectivity_votes(
        stems0: Stem,
        line_start: LineString,
        line_stop: LineString,
        start_buffer,
        end_buffer,
        max_distance,
        max_tree_height,
        tolerance_angle,
        stem: Stem
) -> (bool, List[float], List[Stem], List[Stem]):
    """Compatibility wrapper over _score_connectivity/_build_connection
    with the historical (changed, vote, candidate, slave) contract."""
    scored = _score_connectivity(
        stems0, line_start, line_stop, start_buffer, end_buffer,
        max_distance, max_tree_height, tolerance_angle, stem)
    if scored is None:
        return False, math.inf, None, None
    vote, branch = scored
    return True, vote, _build_connection(stems0, stem, branch), stem


# calculate vote
def calc_vote(ang_l_sp_el_st, ang_l_sp_mp, ang_mp_el_st, candidate, stem,
              stems0, tolerance_angle):
    return (
        ((1 + ang_l_sp_el_st + ang_l_sp_mp + ang_mp_el_st) / tolerance_angle) *
        candidate.start.distance(candidate.stop) ** 2
        + stems0.stop.distance(stem.start) ** 2 *
        (1 + ang_l_sp_el_st + ang_l_sp_mp + ang_mp_el_st) / tolerance_angle
    )


# - Helper functions vector operations -
# Converts the List of [Part] containing Tuples[int] into List of [Stem]
# consisting of shapely geometries
def build_stem_parts(segments: List[Part]):

    t = Timer()
    t.start()
    print("#######################################################")
    print("Build stem segments")
    stems = []
    for i in range(len(segments)):
        if segments[i].start[1] >= segments[i].stop[1]:
            h = segments[i].start
            segments[i].start = segments[i].stop
            segments[i].stop = h
            segments[i].path.reverse()
        else:
            h = segments[i].start
            segments[i].start = segments[i].stop
            segments[i].stop = h
            segments[i].path.reverse()
    segments = set(segments)
    for seg in segments:
        stem = Stem(Point(seg.start), Point(seg.stop), LineString(seg.path), [],
                    [], [], [])
        stems.append(stem)

    print(len(stems), "stems segments build")

    t.stop()
    print("#######################################################")
    print("")
    return stems


def rebuild_endnodes_from_stems(stems: List[Stem]) -> List[Point]:
    t = Timer()
    t.start()
    print("#######################################################")
    print("Rebuild endnodes from stems")
    nodes = []
    for s in stems:
        nodes.append(s.start.coords)
        nodes.append(s.stop.coords)
    t.stop()
    print("#######################################################")
    print("")
    return nodes


# Removes duplicates from stem list
def remove_duplicates(stems: List[Stem], stems0=None) -> List[Stem]:
    stems = list(stems)
    stems.sort(key=lambda x: x.length, reverse=True)
    count = 0

    if type(stems0) is Stem:
        kept = []
        buffer_geom = stems0.path.buffer(0.3)
        for s in stems:
            try:
                if buffer_geom.contains(s.path):
                    count += 1
                    continue
            except Exception:
                pass
            kept.append(s)
        kept.append(stems0)
        return kept, count

    # Spatial prefilter. The original compared every stem against every other
    # remaining stem: on one 4096 px tile with 1567 stems that was ~1.2 MILLION
    # GEOS `contains` calls, the single largest entry in the vector-stage
    # profile. A stem can only be contained in a 0.3 m buffer whose bounding
    # box encloses it, so querying an STRtree first leaves only genuine
    # candidates -- O(n^2) becomes ~O(n log n) with an identical result.
    #
    # Stems are processed longest-first (sorted above) and `alive` marks those
    # neither kept nor already absorbed, which reproduces the original
    # pop-the-longest / drop-the-contained sequence exactly.
    paths = [s.path for s in stems]
    tree = STRtree(paths)
    alive = [True] * len(stems)
    kept = []
    for i, base in enumerate(stems):
        if not alive[i]:
            continue
        alive[i] = False
        kept.append(base)
        buffer_geom = base.path.buffer(0.3)
        for j in tree.query(buffer_geom):
            if not alive[j]:
                continue
            try:
                if buffer_geom.contains(paths[j]):
                    alive[j] = False
                    count += 1
            except Exception:
                pass
    return kept, count


# remove padding and restore geoinformation of the stems
def restore_geoinformation(stems: List[Stem], config, profile):
    t = Timer()
    t.start()

    print("#######################################################")
    print("Restoring geoinformation")

    px_size_x = abs(profile['transform'][0])
    px_size_y = abs(profile['transform'][4])
    bounds = get_bounds_from_profile(profile)
    padding = int(config.max_tree_height / max(px_size_x, px_size_y)) + 1

    for j in range(len(stems)):
        stems[j].start = (
            bounds.left + (stems[j].start[1] - padding) * px_size_x,
            bounds.top - (stems[j].start[0] - padding) * px_size_y)
        stems[j].stop = (
            bounds.left + (stems[j].stop[1] - padding) * px_size_x,
            bounds.top - (stems[j].stop[0] - padding) * px_size_y)

        for k in range(len(stems[j].path)):
            stems[j].path[k] = (
                bounds.left + (stems[j].path[k][1] - padding) * px_size_x,
                bounds.top - (stems[j].path[k][0] - padding) * px_size_y
            )

    t.stop()
    print("#######################################################")
    print("")
    return stems
