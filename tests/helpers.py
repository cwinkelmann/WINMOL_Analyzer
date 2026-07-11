"""Canonicalization and comparison utilities for the golden-master suite.

Fixtures are self-describing JSON (never pickle): a ``Part`` or ``Stem`` is
reduced to a canonical dict with deterministic ordering so that two pipeline
runs can be compared content-wise regardless of set-iteration or worker
completion order (see docs/CODE_REVIEW_2.md A-4).

Only pure-Python deps here (numpy/shapely + domain classes) — no TensorFlow,
so the fast stage tests never pay the TF import.
"""

import gzip
import json
import math
import os

import numpy as np
from shapely.geometry import LineString, Point

from classes.Part import Part
from classes.Stem import Stem

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# JSON IO (deterministic bytes: sorted keys, no whitespace, gzip mtime=0)
# ---------------------------------------------------------------------------

def dumps_canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def write_json_gz(path, obj):
    data = dumps_canonical(obj)
    with open(path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
            gz.write(data)


def read_json_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def write_if_changed(path, obj) -> bool:
    """Write a .json.gz fixture only when its canonical content changed.

    Keeps ``git status`` clean across regenerations that produce identical
    results. Returns True when the file was (re)written.
    """
    if os.path.exists(path):
        try:
            if read_json_gz(path) == json.loads(dumps_canonical(obj)):
                return False
        except Exception:
            pass
    write_json_gz(path, obj)
    return True


# ---------------------------------------------------------------------------
# Part <-> canonical dict
# ---------------------------------------------------------------------------

def _num(v):
    """Coerce numpy scalars to plain python; keep ints exact."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _pair(p):
    return [_num(p[0]), _num(p[1])]


def part_to_dict(part) -> dict:
    return {
        "start": _pair(part.start),
        "stop": _pair(part.stop),
        "path": [_pair(c) for c in part.path],
        "l_bound": _pair(part.l_bound),
        "u_bound": _pair(part.u_bound),
    }


def part_sort_key(d):
    return (
        d["start"][0], d["start"][1],
        d["stop"][0], d["stop"][1],
        len(d["path"]),
        d["path"][0][0] if d["path"] else 0,
        d["path"][-1][0] if d["path"] else 0,
    )


def parts_to_canonical(parts, sort=True) -> list:
    """sort=True for order-insensitive comparison; sort=False preserves
    PIPELINE order — fixtures must be stored in pipeline order because
    downstream stages (set() rebuilds, greedy connect_stems) are
    order-sensitive (review A-4), so replaying a sorted fixture would
    diverge from the recorded run."""
    records = (part_to_dict(p) for p in parts)
    return sorted(records, key=part_sort_key) if sort else list(records)


def parts_from_canonical(records) -> list:
    """Rebuild Part objects. start/stop/bounds MUST be tuples (Part.__hash__
    requires hashable fields; build_stem_parts calls set() on these)."""
    out = []
    for r in records:
        out.append(Part(
            start=tuple(r["start"]),
            stop=tuple(r["stop"]),
            path=[tuple(c) for c in r["path"]],
            l_bound=tuple(r["l_bound"]),
            u_bound=tuple(r["u_bound"]),
        ))
    return out


# ---------------------------------------------------------------------------
# Stem <-> canonical dict
# ---------------------------------------------------------------------------

def stem_to_dict(stem) -> dict:
    return {
        "start": [float(stem.start.x), float(stem.start.y)],
        "stop": [float(stem.stop.x), float(stem.stop.y)],
        "path": [[float(x), float(y)] for x, y in stem.path.coords],
        "diameters": [float(v) for v in stem.segment_diameter_list],
        "lengths": [float(v) for v in stem.segment_length_list],
        "volumes": [float(v) for v in stem.segment_volume_list],
        # per-node measurement vectors (LineStrings) set by get_diameters;
        # needed so reconstructed stems can produce the gpkg 'vectors' layer
        "vector": [[[float(x), float(y)] for x, y in ls.coords]
                   for ls in getattr(stem, "vector", []) or []],
    }


def stem_sort_key(d):
    return (
        round(d["start"][0], 9), round(d["start"][1], 9),
        round(d["stop"][0], 9), round(d["stop"][1], 9),
        len(d["path"]),
    )


def stems_to_canonical(stems, sort=True) -> list:
    """See parts_to_canonical: sort=False preserves pipeline order for
    fixtures; sort=True is for comparisons."""
    records = (stem_to_dict(s) for s in stems)
    return sorted(records, key=stem_sort_key) if sort else list(records)


def stems_from_canonical(records) -> list:
    """Rebuild Stem objects with the measurement lists as recorded (empty
    lists for pre-quantification stages)."""
    out = []
    for r in records:
        out.append(Stem(
            start=Point(r["start"]),
            stop=Point(r["stop"]),
            path=LineString(r["path"]),
            vector=[LineString(v) for v in r.get("vector", [])],
            segment_diameter_list=list(r["diameters"]),
            segment_length_list=list(r["lengths"]),
            segment_volume_list=list(r["volumes"]),
        ))
    return out


# ---------------------------------------------------------------------------
# Comparison assertions
# ---------------------------------------------------------------------------

def _assert_float_lists_close(actual, golden, rtol, atol, ctx):
    assert len(actual) == len(golden), (
        f"{ctx}: length {len(actual)} != golden {len(golden)}")
    if golden:
        diff = np.max(np.abs(np.asarray(actual) - np.asarray(golden)))
        assert np.allclose(actual, golden, rtol=rtol, atol=atol), (
            f"{ctx}: values differ beyond rtol={rtol} (max abs diff {diff})")


def assert_parts_match_golden(parts, golden_records):
    """Exact structural equality for pixel-space parts (integers).
    Order-insensitive: both sides are sorted before comparison."""
    actual = parts_to_canonical(parts)
    golden_records = sorted(golden_records, key=part_sort_key)
    assert len(actual) == len(golden_records), (
        f"part count {len(actual)} != golden {len(golden_records)}")
    for i, (a, g) in enumerate(zip(actual, golden_records)):
        assert a == g, (
            f"part[{i}] differs:\n actual start/stop {a['start']}->{a['stop']}"
            f" pathlen {len(a['path'])}\n golden start/stop {g['start']}->"
            f"{g['stop']} pathlen {len(g['path'])}")


def assert_world_parts_match_golden(parts, golden_records,
                                    rtol=1e-9, atol=1e-9):
    """World-coordinate parts: float coords compared with tolerance."""
    actual = parts_to_canonical(parts)
    golden_records = sorted(golden_records, key=part_sort_key)
    assert len(actual) == len(golden_records), (
        f"part count {len(actual)} != golden {len(golden_records)}")
    for i, (a, g) in enumerate(zip(actual, golden_records)):
        for key in ("start", "stop", "l_bound", "u_bound"):
            _assert_float_lists_close(a[key], g[key], rtol, atol,
                                      f"part[{i}].{key}")
        _assert_float_lists_close(
            [c for pt in a["path"] for c in pt],
            [c for pt in g["path"] for c in pt],
            rtol, atol, f"part[{i}].path")


def assert_stems_match_golden(stems, golden_records,
                              geom_rtol=1e-9, measure_rtol=1e-6):
    actual = stems_to_canonical(stems)
    golden_records = sorted(golden_records, key=stem_sort_key)
    assert len(actual) == len(golden_records), (
        f"stem count {len(actual)} != golden {len(golden_records)}")
    for i, (a, g) in enumerate(zip(actual, golden_records)):
        _assert_float_lists_close(a["start"], g["start"], geom_rtol, 1e-9,
                                  f"stem[{i}].start")
        _assert_float_lists_close(a["stop"], g["stop"], geom_rtol, 1e-9,
                                  f"stem[{i}].stop")
        _assert_float_lists_close(
            [c for pt in a["path"] for c in pt],
            [c for pt in g["path"] for c in pt],
            geom_rtol, 1e-9, f"stem[{i}].path")
        for key in ("diameters", "lengths", "volumes"):
            _assert_float_lists_close(a[key], g[key], measure_rtol, 1e-9,
                                      f"stem[{i}].{key}")
        _assert_float_lists_close(
            [c for ls in a.get("vector", []) for pt in ls for c in pt],
            [c for ls in g.get("vector", []) for pt in ls for c in pt],
            geom_rtol, 1e-9, f"stem[{i}].vector")


def raster_agreement(actual, golden) -> float:
    """Fraction of identical pixels between two equally-shaped arrays."""
    assert actual.shape == golden.shape, (
        f"raster shape {actual.shape} != golden {golden.shape}")
    return float(np.mean(actual == golden))


# ---------------------------------------------------------------------------
# GeoPackage canonicalization (always pyogrio: fiona read path is broken with
# the pinned geopandas 0.14 + installed fiona/pandas — CODE_REVIEW_2 A-8)
# ---------------------------------------------------------------------------

def gpkg_layers(path):
    import pyogrio
    return [row[0] for row in pyogrio.list_layers(path)]


def gpkg_stems_canonical(path) -> list:
    """Order-insensitive content view of a stems layer. stem_id is dropped:
    id assignment order is worker-completion dependent (A-4 territory)."""
    import pyogrio
    gdf = pyogrio.read_dataframe(path, layer="stems")
    records = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        records.append({
            "path": [[float(x), float(y)] for x, y in geom.coords],
            "length": float(row["length"]),
            "volume": float(row["volume"]),
        })
    records.sort(key=lambda r: (round(r["path"][0][0], 9),
                                round(r["path"][0][1], 9),
                                round(r["path"][-1][0], 9),
                                round(r["path"][-1][1], 9),
                                len(r["path"])))
    return records


def assert_gpkg_stems_match(path, golden_path,
                            geom_rtol=1e-9, measure_rtol=1e-6):
    actual = gpkg_stems_canonical(path)
    golden = gpkg_stems_canonical(golden_path)
    assert len(actual) == len(golden), (
        f"stems layer count {len(actual)} != golden {len(golden)}")
    for i, (a, g) in enumerate(zip(actual, golden)):
        _assert_float_lists_close(
            [c for pt in a["path"] for c in pt],
            [c for pt in g["path"] for c in pt],
            geom_rtol, 1e-9, f"gpkg stem[{i}].path")
        assert math.isclose(a["length"], g["length"],
                            rel_tol=measure_rtol, abs_tol=1e-9), (
            f"gpkg stem[{i}].length {a['length']} != {g['length']}")
        assert math.isclose(a["volume"], g["volume"],
                            rel_tol=measure_rtol, abs_tol=1e-9), (
            f"gpkg stem[{i}].volume {a['volume']} != {g['volume']}")


def gpkg_layer_count(path, layer) -> int:
    import pyogrio
    return len(pyogrio.read_dataframe(path, layer=layer))
