#!/usr/bin/env python
"""Benchmark two WINMOL container images against each other on one orthomosaic.

Intended pairing:

    legacy  Dockerfile.blackwell — TensorFlow, .hdf5, upstream StefanReder main
    gpu     docker/gpu/Dockerfile — onnxruntime-gpu, .onnx, this fork

Same GPU, same input, N runs each, median reported. This is the containerised
form of benchmark/bench_orig_vs_changed.py, which compares git refs with local
interpreters; use THIS one when the two sides are images.

The two images have different interfaces, so each side needs a profile:

    legacy   models baked in; input/output under the workdir's standalone/
             dirs; ENTRYPOINT is `python -u`, so the script name is passed
    gpu      models mounted at /models; /input and /output; ENTRYPOINT is
             already `python -u winmol_batch.py`

Example:

    python benchmark/bench_containers.py \\
      --old-image winmol_analyser_docker --old-profile legacy \\
      --new-image ghcr.io/cwinkelmann/winmol-analyzer-gpu:latest \\
      --new-profile gpu \\
      --ortho /data/20220212_Barnekow_4.tiff \\
      --models /data/models \\
      --outdir benchmark/out/containers --repeats 3

Both sides run the SAME weights (General/GenDS): the .onnx was converted from
the .hdf5 at 0.0000 % binary-mask disagreement, so output differences are the
pipeline, not the model.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from bench_orig_vs_changed import stem_stats  # reuse the gpkg reader
except Exception:                                  # pragma: no cover
    stem_stats = None

LEGACY_WORKDIR = "/workspace/WINMOL-Analyzer"


def docker_cmd(profile, image, ortho_dir, out_dir, models_dir, model_name,
               gpus=True):
    """The full `docker run` argv for one side."""
    cmd = ["docker", "run", "--rm"]
    if gpus:
        cmd += ["--gpus", "all"]

    if profile == "legacy":
        # Models are baked in; the CLI reads ./standalone/{input,output}
        # relative to the image's workdir.
        cmd += [
            "-v", f"{ortho_dir}:{LEGACY_WORKDIR}/standalone/input:ro",
            "-v", f"{out_dir}:{LEGACY_WORKDIR}/standalone/output",
            image,
            "winmol_batch.py", model_name,
        ]
    elif profile == "gpu":
        cmd += [
            "-v", f"{models_dir}:/models:ro",
            "-v", f"{ortho_dir}:/input:ro",
            "-v", f"{out_dir}:/output",
            image,
            model_name, "--input", "/input", "--output", "/output",
        ]
    else:
        raise ValueError(f"unknown profile {profile!r}")
    return cmd


def find_gpkg(out_dir):
    hits = []
    for root, _dirs, files in os.walk(out_dir):
        hits += [os.path.join(root, f) for f in files if f.endswith(".gpkg")]
    # The merged//largest result is the one to measure.
    return max(hits, key=os.path.getsize) if hits else None


def run_once(tag, profile, image, ortho, models_dir, outdir, index,
             model_name, gpus):
    run_dir = os.path.join(outdir, f"{tag}_{index}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    cmd = docker_cmd(profile, image, os.path.dirname(os.path.abspath(ortho)),
                     run_dir, models_dir, model_name, gpus=gpus)
    log_path = run_dir + ".log"
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    wall = round(time.time() - t0, 1)

    log = open(log_path, errors="replace").read()
    rec = {
        "config": tag, "run": index, "exit": proc.returncode,
        "wall_s": wall, "image": image, "profile": profile,
        # Whether the GPU was actually used, not merely requested.
        "providers": next((ln.strip() for ln in log.splitlines()
                           if "ExecutionProvider" in ln), None),
    }
    gpkg = find_gpkg(run_dir)
    rec["stems"] = stem_stats(sys.executable, gpkg) if (gpkg and stem_stats) \
        else None
    if proc.returncode != 0:
        rec["failure_tail"] = [ln for ln in log.splitlines() if ln.strip()][-8:]
    print(f"  [{tag} #{index}] exit={proc.returncode} wall={wall}s "
          f"stems={(rec['stems'] or {}).get('count')}", flush=True)
    return rec


def summarise(results):
    out = {}
    for tag in sorted({r["config"] for r in results}):
        ok = [r for r in results if r["config"] == tag and r["exit"] == 0]
        if not ok:
            out[tag] = {"runs": 0, "note": "all runs failed"}
            continue
        walls = [r["wall_s"] for r in ok]
        counts = [r["stems"]["count"] for r in ok
                  if r.get("stems") and "count" in r["stems"]]
        hashes = {r["stems"]["geom_md5"] for r in ok
                  if r.get("stems") and r["stems"].get("geom_md5")}
        out[tag] = {
            "runs": len(ok),
            "wall_s_median": statistics.median(walls),
            "wall_s_all": walls,
            "stems_median": statistics.median(counts) if counts else None,
            "stems_all": counts,
            # Needs >=2 runs, else "identical" is vacuously true.
            "identical_geometry_across_runs": (len(hashes) == 1
                                               if len(ok) >= 2 else None),
        }
    return out


def print_report(summary):
    print("\n=== MEDIAN PER IMAGE ===")
    hdr = f"{'image':12} {'runs':>5} {'wall_s':>9} {'stems':>7}  deterministic"
    print(hdr)
    print("-" * len(hdr))
    for tag, s in summary.items():
        if not s.get("runs"):
            print(f"{tag:12}  {s.get('note')}")
            continue
        ident = s["identical_geometry_across_runs"]
        det = ("n/a (1 run)" if ident is None else
               "YES" if ident else "NO")
        print(f"{tag:12} {s['runs']:>5} {s['wall_s_median']:>9} "
              f"{str(s['stems_median']):>7}  {det}")
    walls = {t: s.get("wall_s_median") for t, s in summary.items()
             if s.get("runs")}
    if len(walls) == 2:
        (ta, wa), (tb, wb) = walls.items()
        if wa and wb:
            fast, slow = (tb, ta) if wb < wa else (ta, tb)
            print(f"\n{fast} is {max(wa, wb) / min(wa, wb):.1f}x faster "
                  f"than {slow}")
    print("\nstems_all:", {t: s.get("stems_all") for t, s in summary.items()})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-image", required=True)
    ap.add_argument("--new-image", required=True)
    ap.add_argument("--old-profile", default="legacy",
                    choices=["legacy", "gpu"])
    ap.add_argument("--new-profile", default="gpu", choices=["legacy", "gpu"])
    ap.add_argument("--ortho", required=True,
                    help="orthomosaic; its DIRECTORY is mounted as the input")
    ap.add_argument("--models", default="",
                    help="host dir with the .onnx models (gpu profile only)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--old-model", default="general",
                    help="model name for the OLD image (lowercase upstream)")
    ap.add_argument("--new-model", default="General")
    ap.add_argument("--no-gpus", action="store_true",
                    help="omit --gpus all (for a CPU sanity run)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    sides = [
        ("legacy", a.old_profile, a.old_image, a.old_model),
        ("new", a.new_profile, a.new_image, a.new_model),
    ]
    results = []
    for tag, profile, image, model_name in sides:
        print(f"=== {tag}: {image} ({profile})", flush=True)
        for i in range(1, a.repeats + 1):
            results.append(run_once(tag, profile, image, a.ortho, a.models,
                                    a.outdir, i, model_name,
                                    gpus=not a.no_gpus))
            json.dump(results,
                      open(os.path.join(a.outdir, "results.json"), "w"),
                      indent=2)
    summary = summarise(results)
    json.dump(summary, open(os.path.join(a.outdir, "summary.json"), "w"),
              indent=2)
    print_report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
