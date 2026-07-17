#!/usr/bin/env python3
"""Benchmark the two pipeline fixes on this stacked branch:

  * perf: GDAL-read tile resampling  (speed  -> per-tile `prep` time)
  * fix:  keep ortho-boundary stems  (recall -> total / boundary stem counts)

It runs the SAME orthomosaic through winmol_run.py on the base branch and on
this perf branch and prints a side-by-side comparison.

It uses `git worktree` to check out each branch into a throwaway directory, so
your working tree (and this script) are never touched, and it works even though
one of the branches is currently checked out.

Usage
-----
    python benchmark/benchmark_fixes.py \
        --ortho /path/to/ortho.tif \
        [--model standalone/model_onnx/Spruce_Deadwood.onnx] \
        [--python /Users/you/opt/anaconda3/envs/WINMOL_Analyzer/bin/python] \
        [--process Trees] [--cpu] [--out-dir benchmark/out]

Notes
-----
* Trees mode is required to exercise the vector merge (the edge fix); Stems mode
  only measures the resize speedup.
* Real device by default (CoreML/Metal on macOS). `--cpu` forces onnxruntime CPU
  for a device-independent, reproducible run. The resize win is in `prep` (CPU),
  so it shows regardless of device.
* Each branch run writes its stem map + GeoPackage under --out-dir for
  inspection / QGIS overlay.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_BRANCH = "feature/qgis-plugin-onnx"
PERF_BRANCH = "perf/resample-and-edge-fix"

# A conda env known to carry the compute deps (onnxruntime + geo stack).
_CONDA_GUESS = os.path.expanduser(
    "~/opt/anaconda3/envs/WINMOL_Analyzer/bin/python")


def _git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True)


def _parse_run(stdout: str) -> dict:
    """Pull metrics out of winmol_run's stdout."""
    m = re.findall(
        r"avg read ([\d.]+)s prep ([\d.]+)s infer ([\d.]+)s write ([\d.]+)s",
        stdout)
    read = prep = infer = write = None
    if m:
        read, prep, infer, write = (float(x) for x in m[-1])  # last = converged
    tpm = re.findall(r"([\d.]+) tiles/min", stdout)
    stems = re.search(r"Total stems written:\s+(\d+)", stdout)
    tiles = re.search(r"Written tile \d+/(\d+)", stdout)
    layout = re.search(r"src (\d+)x(\d+) -> out (\d+)x(\d+)", stdout)
    return {
        "avg_read_s": read, "avg_prep_s": prep,
        "avg_infer_s": infer, "avg_write_s": write,
        "tiles_per_min": float(tpm[-1]) if tpm else None,
        "stems_written": int(stems.group(1)) if stems else None,
        "n_tiles": int(tiles.group(1)) if tiles else None,
        "tile_layout": (f"{layout.group(1)}x{layout.group(2)} -> "
                        f"{layout.group(3)}x{layout.group(4)}"
                        if layout else None),
    }


def _boundary_stems(gpkg: str, stem_map: str, edge_m: float):
    """(total, on_boundary) stem counts. 'on_boundary' = stems intersecting the
    outer `edge_m` ring of the ortho — exactly what the edge fix recovers."""
    try:
        import geopandas as gpd
        import rasterio
        from shapely.geometry import box
        gdf = gpd.read_file(gpkg, layer="stems")
        with rasterio.open(stem_map) as s:
            b = s.bounds
        inner = box(b.left, b.bottom, b.right, b.top).buffer(-abs(edge_m))
        on_edge = int((~gdf.geometry.within(inner)).sum())
        return len(gdf), on_edge
    except Exception as exc:
        print(f"    [warn] boundary analysis failed: {exc}")
        return None, None


def run_branch(name, branch, ortho, model, out_dir, python_exe, process,
               force_cpu, edge_m):
    wt = os.path.join(out_dir, f"_wt_{name}")
    if os.path.exists(wt):
        _git("worktree", "remove", "--force", wt)
    print(f"\n=== {name}  ({branch}) ===")
    r = _git("worktree", "add", "--detach", wt, branch)
    if r.returncode != 0:
        print(f"    [error] worktree add failed: {r.stderr.strip()}")
        return None
    try:
        stem_out = os.path.join(out_dir, f"{name}_stem_map.tiff")
        trees_out = os.path.join(out_dir, f"{name}_detected")
        cmd = [python_exe, "-u", os.path.join(wt, "winmol_run.py"),
               os.path.abspath(model), os.path.abspath(ortho),
               stem_out, trees_out, process]
        env = dict(os.environ)
        if force_cpu:
            env["WINMOL_ONNX_FORCE_CPU"] = "1"
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        print(f"    running: {' '.join(cmd)}")
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=wt, env=env, capture_output=True,
                              text=True)
        wall = time.monotonic() - t0
        if proc.returncode != 0:
            print(f"    [error] exit {proc.returncode}\n"
                  f"{proc.stdout[-1500:]}\n{proc.stderr[-800:]}")
            return None
        metrics = _parse_run(proc.stdout)
        metrics["wall_s"] = wall
        gpkg = f"{trees_out}_detected_stems.gpkg"
        if not os.path.exists(gpkg):
            alt = f"{trees_out}.gpkg"
            gpkg = alt if os.path.exists(alt) else gpkg
        metrics["gpkg"] = gpkg
        if os.path.exists(gpkg):
            total, on_edge = _boundary_stems(gpkg, stem_out, edge_m)
            metrics["stems_total"] = total
            metrics["stems_on_boundary"] = on_edge
        print(f"    done in {wall:.1f}s | prep {metrics['avg_prep_s']}s/tile "
              f"| infer {metrics['avg_infer_s']}s/tile "
              f"| stems {metrics.get('stems_total')}")
        return metrics
    finally:
        _git("worktree", "remove", "--force", wt)


def _fmt(v, suffix=""):
    return "n/a" if v is None else f"{v}{suffix}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    default_python = (_CONDA_GUESS if os.path.exists(_CONDA_GUESS)
                      else sys.executable)
    ap.add_argument("--ortho", required=True, help="orthomosaic GeoTIFF")
    ap.add_argument("--model",
                    default="standalone/model_onnx/Spruce_Deadwood.onnx")
    ap.add_argument("--python", default=default_python,
                    help="interpreter with the compute deps")
    ap.add_argument("--process", default="Trees",
                    choices=["Stems", "Trees", "Nodes"])
    ap.add_argument("--cpu", action="store_true",
                    help="force onnxruntime CPU (reproducible, device-neutral)")
    ap.add_argument("--edge-m", type=float, default=12.0,
                    help="ortho-boundary ring width for the recall metric")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "benchmark", "out"))
    args = ap.parse_args()

    if not os.path.exists(args.ortho):
        sys.exit(f"ortho not found: {args.ortho}")
    os.makedirs(args.out_dir, exist_ok=True)

    st = _git("status", "--porcelain")
    if st.stdout.strip():
        print("[warn] working tree not clean; worktrees use committed state:\n"
              + st.stdout)

    print(f"ortho : {args.ortho}\nmodel : {args.model}\n"
          f"python: {args.python}\nmode  : {args.process}"
          f"{'  (CPU forced)' if args.cpu else '  (real device)'}")

    runs = {}
    for name, branch in (("base", BASE_BRANCH), ("fixed", PERF_BRANCH)):
        runs[name] = run_branch(
            name, branch, args.ortho, args.model, args.out_dir,
            args.python, args.process, args.cpu, args.edge_m)

    base, fixed = runs.get("base"), runs.get("fixed")
    print("\n" + "=" * 72)
    print("BENCHMARK: base (feature/qgis-plugin-onnx) vs fixed "
          "(perf/resample-and-edge-fix)")
    print("=" * 72)
    rows = [
        ("wall time (s)", "wall_s"),
        ("throughput (tiles/min)", "tiles_per_min"),
        ("prep  s/tile  [resize fix]", "avg_prep_s"),
        ("infer s/tile", "avg_infer_s"),
        ("read  s/tile", "avg_read_s"),
        ("stems total   [edge fix]", "stems_total"),
        ("stems on boundary [edge fix]", "stems_on_boundary"),
        ("tile layout", "tile_layout"),
    ]
    print(f"{'metric':32} {'base':>16} {'fixed':>16}   delta")
    print("-" * 72)
    for label, key in rows:
        b = base.get(key) if base else None
        f = fixed.get(key) if fixed else None
        delta = ""
        if isinstance(b, (int, float)) and isinstance(f, (int, float)) and b:
            if key in ("avg_prep_s", "wall_s", "avg_infer_s", "avg_read_s"):
                delta = f"{b / f:.1f}x faster" if f else ""
            elif key in ("stems_total", "stems_on_boundary"):
                delta = f"+{f - b} stems"
            elif key == "tiles_per_min":
                delta = f"{f / b:.1f}x" if b else ""
        print(f"{label:32} {_fmt(b):>16} {_fmt(f):>16}   {delta}")
    print("-" * 72)

    out_json = os.path.join(args.out_dir, "benchmark_result.json")
    with open(out_json, "w") as fh:
        json.dump(runs, fh, indent=2, default=str)
    print(f"\nfull metrics + output GeoPackages under: {args.out_dir}")
    print(f"json: {out_json}")


if __name__ == "__main__":
    main()
