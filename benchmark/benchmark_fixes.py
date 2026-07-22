#!/usr/bin/env python3
"""WINMOL pipeline benchmark harness.

Two kinds of comparison, same machinery:

1. RUNTIME (TensorFlow vs ONNX) -- the default when you pass --tf-model /
   --onnx-model. The current code dispatches on the model extension
   (.hdf5 -> Keras/TensorFlow, .onnx -> onnxruntime) through the SAME pipeline,
   so this isolates the inference engine. The machine decides the hardware:
   tf-metal + CoreML on macOS, TF-CUDA + onnxruntime-CUDA on a CUDA box. Run the
   identical command on each machine.

2. BRANCH A/B (--branch-ab) -- run one model on the base branch vs this perf
   branch (quantifies the resample + edge fixes). Uses git worktrees so your
   checkout is never touched.

Examples
--------
  # TF vs ONNX on whatever machine you run it on (Mac -> tf-metal vs CoreML,
  # CUDA box -> TF-CUDA vs onnxruntime-CUDA):
  python benchmark/benchmark_fixes.py --ortho ortho.tif \
      --tf-model  /path/model_UNet_SpecDS_Spruce_Deadwood_512_*.hdf5 \
      --onnx-model standalone/model_onnx/Spruce_Deadwood.onnx

  # resize+edge fixes, base vs this branch:
  python benchmark/benchmark_fixes.py --ortho ortho.tif --branch-ab \
      --model standalone/model_onnx/Spruce_Deadwood.onnx

Notes
-----
* Trees mode (default) exercises the vector merge; Stems mode measures only the
  prediction/resample.
* `--cpu` forces onnxruntime CPU (device-neutral). The TF path ignores it.
* `--tf-legacy-keras` sets TF_USE_LEGACY_KERAS=1 for TF variants (use if a
  .hdf5 won't load under Keras 3 on the target machine).
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

_CONDA_GUESS = os.path.expanduser(
    "~/opt/anaconda3/envs/WINMOL_Analyzer/bin/python")


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True)


# winmol_run.report_runtime_env() prints, indented under "Environment:":
#   Available providers: CUDAExecutionProvider, CPUExecutionProvider
#   Selected providers: CPUExecutionProvider (forced by WINMOL_ONNX_FORCE_CPU)
#   Device: CPU
# Only the "Selected"/"Device" lines say what ran -- "Available" lists every
# provider the build ships, so matching a bare provider name anywhere in the
# log reports CoreML for a CPU-pinned run.
SELECTED_PROVIDERS_RE = re.compile(r"^\s*Selected providers:\s*(.+)$", re.M)
DEVICE_RE = re.compile(r"^\s*Device:\s*(.+)$", re.M)


def _detect_backend(stdout: str) -> str:
    """Which inference engine + device actually ran (from the log)."""
    selected = SELECTED_PROVIDERS_RE.search(stdout)
    if selected:
        chosen = selected.group(1).split("(")[0]
        if "CUDAExecutionProvider" in chosen:
            return "onnxruntime / CUDA"
        if "CoreMLExecutionProvider" in chosen:
            return "onnxruntime / CoreML"
        return "onnxruntime / CPU"
    device = DEVICE_RE.search(stdout)
    if device:
        label = device.group(1).strip()
        if "CUDA" in label:
            return "onnxruntime / CUDA"
        if "CoreML" in label or "Metal" in label:
            return "onnxruntime / CoreML"
        return "onnxruntime / CPU"
    if "OnnxSegmenter" in stdout:
        m = re.search(r"providers?\W+\[?([A-Za-z, ]*Provider)", stdout)
        return f"onnxruntime ({m.group(1)})" if m else "onnxruntime / CPU"
    if "Metal device set to" in stdout:
        return "tensorflow / Metal"
    if re.search(r"Created TensorFlow device.*GPU", stdout):
        return "tensorflow / CUDA"
    if "Tensorflow version" in stdout:
        return "tensorflow / CPU"
    return "?"


def _parse_run(stdout: str) -> dict:
    m = re.findall(
        r"avg read ([\d.]+)s prep ([\d.]+)s infer ([\d.]+)s write ([\d.]+)s",
        stdout)
    read = prep = infer = write = None
    if m:
        read, prep, infer, write = (float(x) for x in m[-1])
    tpm = re.findall(r"([\d.]+) tiles/min", stdout)
    stems = re.search(r"Total stems written:\s+(\d+)", stdout)
    tiles = re.search(r"Written tile \d+/(\d+)", stdout)
    layout = re.search(r"src (\d+)x(\d+) -> out (\d+)x(\d+)", stdout)
    # New banner: "Legacy Keras model path: TensorFlow 2.16.2".
    # Old banner: "Tensorflow version: 2.16.2".
    tf_ver = re.search(
        r"(?:Tensorflow version:|TensorFlow)\s*([\d.]+)", stdout)
    return {
        "backend": _detect_backend(stdout),
        "avg_read_s": read, "avg_prep_s": prep,
        "avg_infer_s": infer, "avg_write_s": write,
        "tiles_per_min": float(tpm[-1]) if tpm else None,
        "stems_written": int(stems.group(1)) if stems else None,
        "n_tiles": int(tiles.group(1)) if tiles else None,
        "tile_layout": (f"{layout.group(1)}x{layout.group(2)}->"
                        f"{layout.group(3)}x{layout.group(4)}"
                        if layout else None),
        "tf_version": tf_ver.group(1) if tf_ver else None,
    }


def _boundary_stems(gpkg: str, stem_map: str, edge_m: float):
    try:
        import geopandas as gpd
        import rasterio
        from shapely.geometry import box
        gdf = gpd.read_file(gpkg, layer="stems")
        with rasterio.open(stem_map) as s:
            b = s.bounds
        inner = box(b.left, b.bottom, b.right, b.top).buffer(-abs(edge_m))
        return len(gdf), int((~gdf.geometry.within(inner)).sum())
    except Exception as exc:
        print(f"    [warn] boundary analysis failed: {exc}")
        return None, None


def _parse_onnx_profile(path):
    """onnxruntime chrome-trace -> per-execution-provider op time (i.e. how much
    ran on CoreML/ANE vs CPU) + the heaviest ops."""
    try:
        with open(path) as f:
            events = json.load(f)
    except Exception as e:
        return {"error": str(e)}
    prov, ops, total = {}, {}, 0.0
    for e in events:
        if e.get("cat") != "Node" or not str(e.get("name", "")).endswith(
                "kernel_time"):
            continue
        dur = float(e.get("dur", 0))
        a = e.get("args", {})
        p = a.get("provider", "?").replace("ExecutionProvider", "")
        prov[p] = prov.get(p, 0.0) + dur
        op = a.get("op_name", "?")
        ops[op] = ops.get(op, 0.0) + dur
        total += dur
    top = sorted(ops.items(), key=lambda x: -x[1])[:6]
    return {
        "provider_pct": {p: round(100 * d / total, 1)
                         for p, d in prov.items()} if total else {},
        "top_ops_ms": [(o, round(d / 1000, 1)) for o, d in top],
        "total_node_ms": round(total / 1000, 1),
    }


def _parse_powermetrics(path):
    """GPU + ANE utilisation from a powermetrics capture. On Apple silicon
    CoreML often runs on the ANE, which shows as ANE power but NOT GPU % --
    that's why the process looked 'idle' in Activity Monitor's GPU view."""
    try:
        text = open(path, errors="replace").read()
    except Exception as e:
        return {"error": str(e)}

    def nums(pat):
        return [float(x) for x in re.findall(pat, text)]

    def st(a):
        return ({"avg": round(sum(a) / len(a), 1), "max": round(max(a), 1),
                 "n": len(a)} if a else None)
    gpu = nums(r"GPU\s*(?:HW\s*)?active residency:\s*([\d.]+)%")
    return {
        "gpu_active_pct": st(gpu),
        "gpu_power_mW": st(nums(r"GPU Power:\s*([\d.]+)\s*mW")),
        "ane_power_mW": st(nums(r"ANE Power:\s*([\d.]+)\s*mW")),
        "gpu_busy_fraction": (round(sum(1 for x in gpu if x > 5) / len(gpu), 2)
                              if gpu else None),
    }


def run_variant(v, ortho, out_dir, python_exe, process, force_cpu, edge_m,
                onnx_profile=False, powermetrics=False, pm_interval=500):
    label, branch, model = v["label"], v.get("branch"), v["model"]
    wt = None
    cwd = REPO
    print(f"\n=== {label}"
          + (f"  (branch {branch})" if branch else "  (current checkout)")
          + f"  model {os.path.basename(model)} ===")
    if branch:
        wt = os.path.join(out_dir, f"_wt_{label}")
        if os.path.exists(wt):
            _git("worktree", "remove", "--force", wt)
        r = _git("worktree", "add", "--detach", wt, branch)
        if r.returncode != 0:
            print(f"    [error] worktree add failed: {r.stderr.strip()}")
            return None
        cwd = wt
    try:
        stem_out = os.path.join(out_dir, f"{label}_stem_map.tiff")
        trees_out = os.path.join(out_dir, f"{label}_detected")
        cmd = [python_exe, "-u", os.path.join(cwd, "winmol_run.py"),
               os.path.abspath(model), os.path.abspath(ortho),
               stem_out, trees_out, process]
        env = dict(os.environ)
        env.update(v.get("env", {}))
        if force_cpu:
            env["WINMOL_ONNX_FORCE_CPU"] = "1"
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        # Off, so a cold vs warm autotune cache cannot contaminate a timing
        # run (the first run would pay ~60 s of tuning, later ones nothing).
        env.setdefault("WINMOL_BATCH_AUTOTUNE", "off")
        onnx_prefix = None
        if onnx_profile and str(model).lower().endswith(".onnx"):
            onnx_prefix = os.path.join(out_dir, f"{label}_onnxprof")
            env["WINMOL_ONNX_PROFILE"] = "1"
            env["WINMOL_ONNX_PROFILE_PREFIX"] = onnx_prefix
        pm_proc = None
        pm_file = os.path.join(out_dir, f"{label}_powermetrics.txt")
        if powermetrics:
            try:
                pm_proc = subprocess.Popen(
                    ["sudo", "-n", "powermetrics", "-s",
                     "gpu_power,ane_power,cpu_power", "-i", str(pm_interval)],
                    stdout=open(pm_file, "w"), stderr=subprocess.DEVNULL)
                time.sleep(1.0)
                if pm_proc.poll() is not None:
                    print("    [profile] powermetrics could not start; run "
                          "`sudo -v` first to cache credentials.")
                    pm_proc = None
            except FileNotFoundError:
                pm_proc = None
        print(f"    running: {' '.join(cmd)}")
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                              text=True)
        wall = time.monotonic() - t0
        if pm_proc:
            pm_proc.terminate()
            try:
                pm_proc.wait(timeout=5)
            except Exception:
                pm_proc.kill()
        if proc.returncode != 0:
            print(f"    [error] exit {proc.returncode}\n"
                  f"{proc.stdout[-2000:]}\n{proc.stderr[-800:]}")
            return None
        metrics = _parse_run(proc.stdout)
        metrics["wall_s"] = round(wall, 1)
        metrics["model"] = os.path.basename(model)
        if powermetrics and os.path.exists(pm_file):
            metrics["gpu_profile"] = _parse_powermetrics(pm_file)
        if onnx_prefix:
            import glob
            hits = sorted(glob.glob(onnx_prefix + "*"))
            if hits:
                metrics["onnx_profile"] = _parse_onnx_profile(hits[-1])
        gpkg = f"{trees_out}_detected_stems.gpkg"
        if not os.path.exists(gpkg) and os.path.exists(f"{trees_out}.gpkg"):
            gpkg = f"{trees_out}.gpkg"
        if os.path.exists(gpkg):
            total, on_edge = _boundary_stems(gpkg, stem_out, edge_m)
            metrics["stems_total"] = total
            metrics["stems_on_boundary"] = on_edge
        print(f"    {metrics['backend']} | {wall:.1f}s | "
              f"prep {metrics['avg_prep_s']}s infer {metrics['avg_infer_s']}s "
              f"| stems {metrics.get('stems_total')}")
        return metrics
    finally:
        if wt:
            _git("worktree", "remove", "--force", wt)


def build_variants(args):
    if args.tf_model or args.onnx_model:
        variants = []
        if args.tf_model:
            env = {"TF_USE_LEGACY_KERAS": "1"} if args.tf_legacy_keras else {}
            variants.append({"label": "tensorflow", "model": args.tf_model,
                             "branch": None, "env": env})
        if args.onnx_model:
            variants.append({"label": "onnx", "model": args.onnx_model,
                             "branch": None, "env": {}})
        return variants
    if args.branch_ab:
        return [
            {"label": "base", "model": args.model, "branch": BASE_BRANCH},
            {"label": "fixed", "model": args.model, "branch": PERF_BRANCH},
        ]
    return [{"label": "current", "model": args.model, "branch": None}]


def _fmt(v):
    return "n/a" if v is None else str(v)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ortho", required=True, help="orthomosaic GeoTIFF")
    ap.add_argument("--tf-model", help="a .hdf5 Keras model (TensorFlow path)")
    ap.add_argument("--onnx-model", help="a .onnx model (onnxruntime path)")
    ap.add_argument("--branch-ab", action="store_true",
                    help="compare base vs this perf branch (needs --model)")
    ap.add_argument("--model",
                    default="standalone/model_onnx/Spruce_Deadwood.onnx",
                    help="model for --branch-ab / default single run")
    default_python = (_CONDA_GUESS if os.path.exists(_CONDA_GUESS)
                      else sys.executable)
    ap.add_argument("--python", default=default_python,
                    help="interpreter with the compute deps")
    ap.add_argument("--process", default="Trees",
                    choices=["Stems", "Trees", "Nodes"])
    ap.add_argument("--cpu", action="store_true",
                    help="force onnxruntime CPU (device-neutral)")
    ap.add_argument("--tf-legacy-keras", action="store_true",
                    help="set TF_USE_LEGACY_KERAS=1 for TF variants")
    ap.add_argument("--onnx-profile", action="store_true",
                    help="onnxruntime op-level profile (CoreML/ANE vs CPU op "
                         "time); no sudo, onnx variants only")
    ap.add_argument("--powermetrics", action="store_true",
                    help="sample GPU+ANE utilisation via powermetrics during "
                         "each run (needs sudo; caches creds once)")
    ap.add_argument("--pm-interval", type=int, default=500,
                    help="powermetrics sample interval (ms)")
    ap.add_argument("--edge-m", type=float, default=12.0)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "benchmark", "out"))
    args = ap.parse_args()

    if not os.path.exists(args.ortho):
        sys.exit(f"ortho not found: {args.ortho}")
    os.makedirs(args.out_dir, exist_ok=True)
    variants = build_variants(args)

    print(f"ortho : {args.ortho}\npython: {args.python}\n"
          f"mode  : {args.process}"
          f"{'  (onnx CPU forced)' if args.cpu else ''}")
    st = _git("status", "--porcelain")
    if st.stdout.strip() and any(v.get("branch") for v in variants):
        print("[warn] working tree not clean; branch worktrees use committed "
              "state only")
    if args.powermetrics:
        print("[profile] powermetrics needs sudo — caching credentials "
              "(one prompt)…")
        subprocess.run(["sudo", "-v"])

    results = []
    for v in variants:
        results.append((v["label"],
                        run_variant(v, args.ortho, args.out_dir, args.python,
                                    args.process, args.cpu, args.edge_m,
                                    onnx_profile=args.onnx_profile,
                                    powermetrics=args.powermetrics,
                                    pm_interval=args.pm_interval)))

    print("\n" + "=" * 66)
    print("BENCHMARK RESULT")
    print("=" * 66)
    labels = [lbl for lbl, _ in results]
    data = {lbl: (m or {}) for lbl, m in results}
    # legend: label -> model + backend (kept out of the table so columns stay
    # narrow even with long .hdf5 filenames)
    for lbl in labels:
        d = data[lbl]
        print(f"  {lbl:11} {d.get('backend', '?'):22} "
              f"{d.get('model', '?')}")
    print()
    col = 20
    rows = [
        ("backend / device", "backend"),
        ("wall time (s)", "wall_s"),
        ("throughput tiles/min", "tiles_per_min"),
        ("infer s/tile", "avg_infer_s"),
        ("prep s/tile", "avg_prep_s"),
        ("read s/tile", "avg_read_s"),
        ("stems total", "stems_total"),
        ("stems on boundary", "stems_on_boundary"),
        ("tile layout", "tile_layout"),
    ]

    def _cell(v):
        s = "n/a" if v is None else str(v)
        return (s[:col - 2] + "…") if len(s) > col - 1 else s

    header = f"{'metric':22}" + "".join(f"{_cell(lbl):>{col}}"
                                        for lbl in labels)
    print(header)
    print("-" * len(header))
    for label, key in rows:
        cells = "".join(f"{_cell(data[lbl].get(key)):>{col}}"
                        for lbl in labels)
        print(f"{label:22}{cells}")
    print("-" * len(header))

    # headline ratio when exactly two variants
    if len(results) == 2 and all(m for _, m in results):
        a, b = results[0][1], results[1][1]
        for key, name in (("avg_infer_s", "infer/tile"),
                          ("wall_s", "wall time"),
                          ("tiles_per_min", "throughput")):
            x, y = a.get(key), b.get(key)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                    and x and y:
                if key == "tiles_per_min":
                    print(f"{name:14}: {labels[1]} is "
                          f"{y / x:.2f}x {labels[0]}")
                else:
                    faster = labels[1] if y < x else labels[0]
                    print(f"{name:14}: {max(x, y) / min(x, y):.2f}x "
                          f"faster on {faster}")

    # profiling detail
    if any(m and (m.get("onnx_profile") or m.get("gpu_profile"))
           for _, m in results):
        print("\nPROFILE")
        for lbl, m in results:
            if not m:
                continue
            op = m.get("onnx_profile")
            if op and not op.get("error"):
                print(f"  {lbl}: onnx op-time by provider {op['provider_pct']} "
                      f"% (total {op['total_node_ms']} ms) | heaviest "
                      f"{op['top_ops_ms']}")
            gp = m.get("gpu_profile")
            if gp and not gp.get("error"):
                g = gp.get("gpu_active_pct")
                print(f"  {lbl}: system GPU active {g}% | GPU busy fraction "
                      f"{gp.get('gpu_busy_fraction')} | ANE power "
                      f"{gp.get('ane_power_mW')} mW  (CoreML on the ANE shows "
                      f"here, not in GPU %)")

    out_json = os.path.join(args.out_dir, "benchmark_result.json")
    with open(out_json, "w") as fh:
        json.dump({lbl: m for lbl, m in results}, fh, indent=2, default=str)
    print(f"\noutputs + json under: {args.out_dir}")


if __name__ == "__main__":
    main()
