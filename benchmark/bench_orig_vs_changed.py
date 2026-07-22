#!/usr/bin/env python
"""Original pipeline vs the full change stack, on one orthomosaic, without QGIS.

    A  ORIGINAL  <ref>=origin/main, TensorFlow/Keras .hdf5, PYTHONHASHSEED UNSET
                 (exactly as shipped, including its nondeterminism)
    B  CHANGED   <ref>=this branch, ONNX/onnxruntime .onnx

Both sides run the SAME weights: General.onnx was converted from
model_UNet_GenDS_512_2023-02-27_211141.hdf5 with 0.0000 % binary-mask
disagreement (docs/onnx-conversion-parity.md), so differences in OUTPUT are
attributable to pipeline changes -- the ortho-boundary edge fix, the
determinism guard, the GDAL-read resample -- and not to swapping models.

Each configuration runs N times and the REPORTED FIGURE IS THE MEDIAN. Repeats
are not optional padding: the original has no PYTHONHASHSEED guard, so its stem
count varies run to run, and a single run each would misattribute that variance
to the code changes.

The script creates its own git worktrees, so it needs nothing but a checkout.

    python benchmark/bench_orig_vs_changed.py \
        --ortho /data/20220212_Barnekow_4.tiff \
        --orig-model /models/model_UNet_GenDS_512_2023-02-27_211141.hdf5 \
        --new-model  /models/General.onnx \
        --outdir benchmark/out/barnekow

## Running this on a CUDA machine

Timings from Apple Silicon are NOT a fair performance benchmark: there the
vector phase dominates (~73 % of a run) and inference is comparatively cheap,
so the profile shape differs from a datacentre GPU. On CUDA:

  * ORIGINAL side needs TensorFlow with CUDA + `tf-keras`
    (`TF_USE_LEGACY_KERAS=1` is set automatically -- the shipped .hdf5 are
    Keras 2 artifacts and will not load under Keras 3).
  * CHANGED side needs `onnxruntime-gpu`. Force the provider explicitly with
    `--onnx-providers CUDAExecutionProvider` so a silent CPU fallback cannot be
    mistaken for a slow GPU.
  * Use `--python` to point at the interpreter that has those installed, or run
    the two sides with different interpreters via `--orig-python`/`--new-python`
    (TF and onnxruntime-gpu often disagree about CUDA/cuDNN versions, so
    separate environments are usually the path of least resistance).

Re-print a summary from an existing run without recomputing:

    python benchmark/bench_orig_vs_changed.py --report benchmark/out/barnekow
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHASE_RE = re.compile(
    r"(prediction|vector|merge|skeleton|quantif\w*)\D{0,40}?([\d.]+)\s*s", re.I)
# "Selected providers"/"Device" come from the current banner; "Visible GPUs"
# is now printed on the NVIDIA path only, and the ORIGINAL side of the
# comparison still prints the old TensorFlow/CUDA wording -- keep both so a
# cross-branch run reports a device line for either.
DEVICE_RE = re.compile(
    r"(Visible GPUs|Hardware detected|Inference runtime|Device"
    r"|provider\w*)\s*[:=]\s*(.+)", re.I)


def _stats_code():
    return (
        "import geopandas as gpd, hashlib, json, sys\n"
        "g = gpd.read_file(sys.argv[1], layer='stems', engine='pyogrio')\n"
        "wkb = sorted(x.wkb_hex for x in g.geometry)\n"
        "print(json.dumps({\n"
        "  'count': len(g),\n"
        "  'length_m': round(float(g.length.sum()), 2),\n"
        "  'volume_m3': (round(float(g['volume'].sum()), 3)\n"
        "                if 'volume' in g.columns else None),\n"
        "  'geom_md5': hashlib.md5(''.join(wkb).encode()).hexdigest(),\n"
        "}))\n")


def stem_stats(python, gpkg):
    if not os.path.exists(gpkg):
        return None
    out = subprocess.run([python, "-c", _stats_code(), gpkg],
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": (out.stderr or out.stdout)[-300:]}


def parse_log(text):
    phases = {}
    for name, secs in PHASE_RE.findall(text):
        key = name.lower()[:6]
        try:
            phases[key] = max(phases.get(key, 0.0), float(secs))
        except ValueError:
            pass
    device = [m.group(0).strip()[:120] for m in DEVICE_RE.finditer(text)][:3]
    return phases, device


def ensure_worktree(ref, path):
    if os.path.isdir(os.path.join(path, ".git")) or os.path.exists(path):
        return path
    subprocess.run(["git", "worktree", "add", "-q", "--detach", path, ref],
                   cwd=REPO, check=True)
    return path


def run_once(cfg, ortho, outdir, index):
    prefix = os.path.join(outdir, f"{cfg['tag']}_{index}")
    log_path = prefix + ".log"
    env = dict(os.environ)
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env.pop("PYTHONHASHSEED", None)          # ORIGINAL behaviour by default
    # The batch-size autotune defaults to "auto": it would tune on the first
    # run (a one-off stall of ~60 s) and reuse a cached batch afterwards, so
    # run 1 and run 2 of the same variant would not be comparable. Pin it off
    # -- benchmarks measure the pipeline, not the tuner.
    env["WINMOL_BATCH_AUTOTUNE"] = "off"
    env.update(cfg["env"])

    cmd = [cfg["python"], "-u", "winmol_run.py", cfg["model"], ortho,
           prefix + "_stem_map.tif", prefix, "Trees"]
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=cfg["workdir"], env=env,
                              stdout=lf, stderr=subprocess.STDOUT)
    wall = round(time.time() - t0, 1)
    phases, device = parse_log(open(log_path, errors="replace").read())
    rec = {"config": cfg["tag"], "run": index, "exit": proc.returncode,
           "wall_s": wall, "phases": phases, "device": device,
           "stems": stem_stats(cfg["python"], prefix + ".gpkg")}
    if proc.returncode != 0:
        lines = [ln for ln in open(log_path, errors="replace")
                 .read().splitlines() if ln.strip()]
        rec["failure_tail"] = lines[-8:]
    print(f"  [{cfg['tag']} #{index}] exit={proc.returncode} wall={wall}s "
          f"stems={(rec['stems'] or {}).get('count')}", flush=True)
    return rec


def summarise(results):
    """Median per configuration, plus the run-to-run spread that matters."""
    out = {}
    for tag in sorted({r["config"] for r in results}):
        runs = [r for r in results if r["config"] == tag and r["exit"] == 0]
        if not runs:
            out[tag] = {"runs": 0, "note": "all runs failed"}
            continue
        counts = [r["stems"]["count"] for r in runs if r.get("stems")
                  and "count" in r["stems"]]
        walls = [r["wall_s"] for r in runs]
        vols = [r["stems"]["volume_m3"] for r in runs
                if r.get("stems") and r["stems"].get("volume_m3") is not None]
        hashes = {r["stems"]["geom_md5"] for r in runs
                  if r.get("stems") and r["stems"].get("geom_md5")}
        out[tag] = {
            "runs": len(runs),
            "wall_s_median": statistics.median(walls),
            "wall_s_all": walls,
            "stems_median": statistics.median(counts) if counts else None,
            "stems_all": counts,
            "stems_spread": (max(counts) - min(counts)) if counts else None,
            "volume_m3_median": statistics.median(vols) if vols else None,
            # A determinism claim needs at least two runs to compare; with one
            # run there is exactly one hash and "identical" would be vacuous.
            "identical_geometry_across_runs": (len(hashes) == 1
                                               if len(runs) >= 2 else None),
            "distinct_geometries": len(hashes),
        }
    return out


def print_report(summary):
    print("\n=== MEDIAN OF %d RUNS PER CONFIGURATION ===" %
          max((v.get("runs", 0) for v in summary.values()), default=0))
    hdr = f"{'config':10} {'wall_s':>9} {'stems':>7} {'spread':>7} " \
          f"{'volume_m3':>11}  deterministic"
    print(hdr)
    print("-" * len(hdr))
    for tag, s in summary.items():
        if not s.get("runs"):
            print(f"{tag:10}  {s.get('note')}")
            continue
        ident = s["identical_geometry_across_runs"]
        det = ("n/a (1 run)" if ident is None else
               "YES" if ident else f"NO ({s['distinct_geometries']} distinct)")
        print(f"{tag:10} {s['wall_s_median']:>9} {str(s['stems_median']):>7} "
              f"{str(s['stems_spread']):>7} "
              f"{str(s['volume_m3_median']):>11}  {det}")
    print("\nstems_all:", {t: s.get("stems_all") for t, s in summary.items()})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report",
                    help="re-print the summary for an existing outdir, then "
                         "exit")
    ap.add_argument("--ortho")
    ap.add_argument("--outdir")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--orig-ref", default="origin/main")
    ap.add_argument("--new-ref", default="HEAD")
    ap.add_argument("--orig-model", help=".hdf5 Keras model")
    ap.add_argument("--new-model", help=".onnx model")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--orig-python", help="override for the TF side")
    ap.add_argument("--new-python", help="override for the onnxruntime side")
    ap.add_argument("--onnx-providers",
                    help="e.g. CUDAExecutionProvider — set explicitly on GPU "
                         "boxes so a silent CPU fallback is not mistaken for a "
                         "slow GPU")
    a = ap.parse_args()

    if a.report:
        results = json.load(open(os.path.join(a.report, "results.json")))
        print_report(summarise(results))
        return 0

    for required in ("ortho", "outdir", "orig_model", "new_model"):
        if not getattr(a, required):
            ap.error(f"--{required.replace('_', '-')} is required")

    os.makedirs(a.outdir, exist_ok=True)
    wt_orig = ensure_worktree(a.orig_ref,
                              os.path.join(a.outdir, "_wt_original"))
    wt_new = ensure_worktree(a.new_ref, os.path.join(a.outdir, "_wt_changed"))

    new_env = {}
    if a.onnx_providers:
        new_env["WINMOL_ONNX_PROVIDERS"] = a.onnx_providers

    configs = [
        {"tag": "original", "workdir": wt_orig, "model": a.orig_model,
         "python": a.orig_python or a.python,
         "env": {"TF_USE_LEGACY_KERAS": "1"}},
        {"tag": "changed", "workdir": wt_new, "model": a.new_model,
         "python": a.new_python or a.python, "env": new_env},
    ]

    results = []
    for cfg in configs:
        print(f"=== {cfg['tag']}: {os.path.basename(cfg['model'])} "
              f"({cfg['python']})", flush=True)
        for i in range(1, a.repeats + 1):
            results.append(run_once(cfg, a.ortho, a.outdir, i))
            with open(os.path.join(a.outdir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

    summary = summarise(results)
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
