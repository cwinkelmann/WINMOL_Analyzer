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


def _merge_diameter_lists(first: Stem, second: Stem) -> List[float]:
    """Per-node diameters for a merged path built as
    ``first.path.coords[:-1] + second.path.coords[1:]`` (the construction every
    merge branch in calc_connectivity_votes uses).

    Only meaningful when BOTH parents carry one diameter per path node -- the
    tiled edge-merge case, where diameters were measured in-tile before
    merging. In-tile connect_stems runs BEFORE quantification (the lists are
    empty), so this returns ``[]`` there and behaviour is unchanged; the
    diameters are measured fresh afterwards. Without it a merged stem kept the
    base's SHORTER diameter list, and quantify_stem then indexed
    ``segment_diameter_list[i + 1]`` off the end (issue #41)."""
    d_first = list(getattr(first, 'segment_diameter_list', []) or [])
    d_second = list(getattr(second, 'segment_diameter_list', []) or [])
    if (len(d_first) == len(first.path.coords)
            and len(d_second) == len(second.path.coords)):
        return d_first[:-1] + d_second[1:]
    return []


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


def _remove_duplicates_against_base(
    cycle_stems: Sequence[Stem], remaining: Set[int], base_idx: int) \
        -> tuple[Set[int], int]:
    if base_idx not in remaining:
        return remaining, 0
    base = cycle_stems[base_idx]
    buffer_geom = base.path.buffer(0.3)
    to_remove = set()
    for idx in remaining:
        if idx == base_idx:
            continue
        try:
            if buffer_geom.contains(cycle_stems[idx].path):
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

                best_vote = math.inf
                best_candidate = None
                best_slave_idx = None

                for idx in filtered_indices:
                    changed, vote, candidate_stem, _ = calc_connectivity_votes(
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
                    if changed and vote < best_vote:
                        best_vote = vote
                        best_candidate = candidate_stem
                        best_slave_idx = idx

                if best_candidate is not None and best_slave_idx is not None:
                    base_stem = best_candidate
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
    # Calculate votes for the aggregation of stem parts to stems
    if stem == stems0:
        # if the stems are identical return no change and infinite vote
        return False, math.inf, None, None
    change = False
    votes = []
    candidates = []
    slaves = []

    if len(stem.path.coords) < 4:
        e_line_start = LineString([stem.path.coords[0], stem.path.coords[-1]])
        e_line_stop = LineString([stem.path.coords[0], stem.path.coords[-1]])
    else:
        if len(stem.path.coords) < 8:
            k = len(stem.path.coords) - 2
        else:
            k = 6
        e_line_start = LineString([stem.path.coords[1], stem.path.coords[k]])
        e_line_stop = LineString(
            [stem.path.coords[-(k + 1)], stem.path.coords[-2]])

    ang_l_sp_el_st = abs(ang(line_stop.coords, e_line_start.coords))
    ang_el_sp_l_st = abs(ang(e_line_stop.coords, line_start.coords))

    has_length_2 = len(stem.path.coords) == 2
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

            if len(stems0.path.coords) > 2 and len(stem.path.coords) > 2:
                start = LineString(stems0.path.coords[:-1])
                end = LineString(stem.path.coords[1:])
                new_path = linemerge([start, missing_part_, end])
            else:
                if len(stems0.path.coords) > 2 and has_length_2:
                    start = LineString(stems0.path.coords[:-1])
                    new_path = linemerge([start, missing_part_])
                else:
                    if (len(stems0.path.coords) == 2 and len(
                            stem.path.coords) > 2):
                        end = LineString(stem.path.coords[1:])
                        new_path = linemerge([missing_part_, end])
                    else:
                        if (len(stems0.path.coords) == 2 and has_length_2):
                            new_path = missing_part_

            change = True
            candidate = _clone_stem(stems0)
            candidate.path = new_path
            candidate.stop = stem.stop
            # merged path = stems0[:-1] + stem[1:]; merge the parents' per-node
            # diameters the same way and drop the now-stale measures, so a
            # later quantify_stem on this merged stem cannot IndexError (#41).
            candidate.segment_diameter_list = \
                _merge_diameter_lists(stems0, stem)
            candidate.segment_length_list = []
            candidate.segment_volume_list = []
            candidate.vector = []
            slave = stem
            vote = calc_vote(ang_l_sp_el_st, ang_l_sp_mp, ang_mp_el_st,
                             candidate, stem, stems0, tolerance_angle)
            candidates.append(candidate)
            votes.append(vote)
            slaves.append(slave)

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
                tolerance_angle * dist_f) and abs(
                ang(missing_part_.coords, line_start.coords)) < (
                tolerance_angle * dist_f) and stem.start.distance(
                stems0.stop) < max_tree_height):
            if len(stem.path.coords) > 2 and len(stems0.path.coords) > 2:
                start = LineString(stem.path.coords[:-1])
                end = LineString(stems0.path.coords[1:])
                new_path = linemerge([start, missing_part_, end])
            else:
                if len(stem.path.coords) > 2 and len(stems0.path.coords) == 2:
                    start = LineString(stem.path.coords[:-1])
                    new_path = linemerge([start, missing_part_])
                else:
                    if has_length_2 and len(stems0.path.coords) > 2:
                        end = LineString(stems0.path.coords[1:])
                        new_path = linemerge([missing_part_, end])
                    else:
                        if (has_length_2 and len(
                                stems0.path.coords) == 2):
                            new_path = missing_part_

            change = True
            candidate = _clone_stem(stems0)
            candidate.path = new_path
            candidate.start = stem.start
            # merged path = stem[:-1] + stems0[1:] in this branch; merge the
            # diameter lists the same way and drop the stale measures (#41).
            candidate.segment_diameter_list = \
                _merge_diameter_lists(stem, stems0)
            candidate.segment_length_list = []
            candidate.segment_volume_list = []
            candidate.vector = []
            slave = stem
            vote = calc_vote(ang_el_sp_l_st, ang_el_sp_mp, ang_mp_l_st,
                             candidate, stems0, stem, tolerance_angle)
            candidates.append(candidate)
            votes.append(vote)
            slaves.append(slave)

    if change:
        index_min = min(range(len(votes)), key=votes.__getitem__)
        return True, votes[index_min], candidates[index_min], slaves[index_min]
    else:
        return False, math.inf, None, None


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
