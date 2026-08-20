#!/usr/bin/env python3
"""Batch runner for WINMOL Analyzer.

- Reads available models from config.json (repo root).
- Runs winmol_run.py for all *.tif / *.tiff in the input folder.
- Optionally merges tiled outputs with utils.IO.merge_and_filter_tiled_results.
"""

import argparse
import concurrent.futures
import os
import queue
import subprocess
import sys
from typing import List, Optional

from plugin_utils import container
from plugin_utils.config_overrides import set_default
from plugin_utils.gpu_probe import run_nvidia_smi_query
from plugin_utils.model_registry import (
    ModelDownloadError,
    Registry,
    detect_device,
    ensure_model,
    load_registry,
)

DEFAULT_INPUT_FOLDER = "./standalone/input"
DEFAULT_OUTPUT_FOLDER = "./standalone/output"
DEFAULT_MODEL_DIR = "./standalone/model"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def resolve_model_path(
    name: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> str:
    """Resolve a family or model id from config.json to a verified local
    file, downloading (and checksum-verifying) it into ``model_dir`` if
    it is missing or stale. Raises KeyError for an unknown name and
    ModelDownloadError if the fetch/verification fails."""
    registry = load_registry(config_path)
    # sys.prefix is the environment this batch run will itself predict
    # in, so its sentinel is the right answer to "can we use the card?".
    # Without it a CPU-only env on an NVIDIA box fetches the fp16 variant.
    entry = registry.resolve(name, device=detect_device(sys.prefix))
    return ensure_model(entry, model_dir)


def _format_model_list(registry: Registry) -> str:
    """Human-readable listing of families and models for --list-models."""
    lines = ["Families (device-aware; pick one to get the best variant "
             "for this machine):"]
    for fid in sorted(registry.families):
        fam = registry.families[fid]
        lines.append(f"  {fid:<20s} {fam.label}")
    lines.append("")
    lines.append("Models (id, label, precision, size):")
    for mid in sorted(registry.entries):
        e = registry.entries[mid]
        size = f"{e.size_mb:.2f} MB" if e.size_mb is not None else "size ?"
        lines.append(f"  {mid:<24s} {e.label:<45s} "
                     f"precision={e.precision:<5s} {size}")
    return "\n".join(lines)


def list_orthomosaics(input_folder: str) -> List[str]:
    if not os.path.isdir(input_folder):
        return []
    return sorted(
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".tif", ".tiff"))
    )


def detect_gpu_count() -> int:
    """Number of visible NVIDIA GPUs, or 0 if none / nvidia-smi unavailable."""
    lines = run_nvidia_smi_query("index", timeout=20)
    return len(lines) if lines else 0


def _with_cpu_budget(overrides_json: str, cpu_budget: int) -> str:
    """Add a max_cpu_workers budget to WINMOL_CONFIG_OVERRIDES_JSON.

    An explicit user-set max_cpu_workers wins. Unparsable JSON is passed
    through untouched -- the child reports the real error.
    """
    return set_default(overrides_json, "max_cpu_workers", int(cpu_budget))


def run_winmol(input_image: str, model_path: str, output_folder: str,
               gpu_id: Optional[int] = None,
               cpu_budget: Optional[int] = None,
               process_type: str = 'Nodes') -> None:
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    output_stem_map = os.path.join(output_folder, f"{base_name}_stem_map.tif")
    output_prefix = os.path.join(output_folder, base_name)

    os.makedirs(output_folder, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        "winmol_run.py",
        model_path,
        input_image,
        output_stem_map,
        output_prefix,
        process_type,
    ]

    env = dict(os.environ)
    # winmol_run.py's determinism guard re-execs itself when
    # PYTHONHASHSEED is unset; pinning it here spares every child that
    # extra interpreter start. An explicit user value is respected.
    env.setdefault("PYTHONHASHSEED", "0")
    if gpu_id is not None:
        # Pin this ortho to one GPU. The child then plans for a SINGLE GPU --
        # the well-tested path -- instead of every concurrent job trying to
        # spread itself across all of them and contending.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if cpu_budget is not None:
        # Each child plans against the FULL machine; with N jobs the
        # CPU-bound vector phases would oversubscribe the cores N-fold.
        # Give every job an equal share via the standard override knob.
        env["WINMOL_CONFIG_OVERRIDES_JSON"] = _with_cpu_budget(
            env.get("WINMOL_CONFIG_OVERRIDES_JSON", ""), cpu_budget)

    tag = f"[gpu {gpu_id}] " if gpu_id is not None else ""
    print(f"{tag}Processing {input_image} "
          f"with model {os.path.basename(model_path)}", flush=True)
    subprocess.run(command, check=True, env=env)
    print(f"{tag}Done: {base_name}", flush=True)


def merge_results(
    work_dir: str,
    output_gpkg: Optional[str] = None,
    edge_buffer_m: float = 1.0,
) -> str:
    """Merge tiled results into a single GeoPackage."""
    from utils import IO  # local import: only needed when merge is requested

    return IO.merge_and_filter_tiled_results(
        work_dir=work_dir,
        output_gpkg=output_gpkg,
        edge_buffer_m=edge_buffer_m,
    )


def process_orthos(orthos, model_path, output_folder, jobs=1,
                   process_type='Nodes'):
    """Run every orthomosaic, optionally several at once.

    Returns a list of (path, reason) for those that failed -- the batch
    always attempts all of them. Previously a single failure raised out of
    the loop and abandoned the rest, so one bad file in an overnight batch
    of twenty cost the other nineteen.

    Why parallelise across ORTHOS rather than harder within one: a single
    orthomosaic cannot use many GPUs (the planner caps GPU workers by tile
    count, and under 1000 tiles that is two), and its vector phase is
    CPU-bound anyway, so extra GPUs do not help it. Whole orthomosaics are
    independent -- embarrassingly parallel, no coordination, and each child
    takes the single-GPU path. It also overlaps one job's CPU-bound vector
    phase with another's GPU-bound prediction, which is worth something even
    on ONE GPU.
    """
    failures = []
    jobs = max(1, int(jobs))

    if jobs == 1 or len(orthos) == 1:
        for ortho in orthos:
            try:
                run_winmol(ortho, model_path, output_folder,
                           process_type=process_type)
            except subprocess.CalledProcessError as e:
                print(f"  FAILED: {ortho}: {e}", flush=True)
                failures.append((ortho, str(e)))
        return failures

    gpus = detect_gpu_count()
    # Worker slot -> GPU. With more slots than GPUs they share, which is
    # deliberate: prediction and vectorisation alternate, so a GPU is idle
    # for much of each job.
    slots: "queue.Queue" = queue.Queue()
    for i in range(jobs):
        slots.put(i % gpus if gpus else None)

    # container.cpu_count(), not os.cpu_count(): inside `docker run
    # --cpus=4` the latter reports the HOST's cores and every job would
    # be budgeted for cores the container cannot use.
    cpu_budget = max(1, container.cpu_count() // jobs)

    print(f"Processing {len(orthos)} orthomosaics, {jobs} at a time"
          + (f" across {gpus} GPU(s)" if gpus else " (no GPU detected)")
          + f" | {cpu_budget} CPU workers per job",
          flush=True)

    def _one(ortho):
        slot = slots.get()
        try:
            run_winmol(ortho, model_path, output_folder, gpu_id=slot,
                       cpu_budget=cpu_budget,
                       process_type=process_type)
            return ortho, None
        except subprocess.CalledProcessError as e:
            return ortho, str(e)
        finally:
            slots.put(slot)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for ortho, err in pool.map(_one, orthos):
            if err:
                print(f"  FAILED: {ortho}: {err}", flush=True)
                failures.append((ortho, err))
    return failures


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch process orthomosaics in a folder using WINMOL Analyzer. "
            "Models are resolved from config.json: a family name (Spruce, "
            "Beech, Spruce_Deadwood, General) picks the best precision for "
            "this machine, or pass an explicit model id. See --list-models."
        )
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model family or id from config.json (see --list-models)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the families and models available in config.json, "
             "then exit.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory holding (or to download) model files "
             f"(default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FOLDER,
        help=f"Input folder (default: {DEFAULT_INPUT_FOLDER})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder (default: {DEFAULT_OUTPUT_FOLDER})",
    )

    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help=(
            "Process this many orthomosaics concurrently (default: 1). Each "
            "job is pinned to one GPU via CUDA_VISIBLE_DEVICES, so a machine "
            "with N GPUs can run N orthomosaics at once. Values above the "
            "GPU count still help: prediction and vectorisation alternate, "
            "so one job's CPU-bound vector phase overlaps another's GPU "
            "work."
        ),
    )
    parser.add_argument(
        "--process-type",
        default="Nodes",
        choices=["Stems", "Trees", "Nodes"],
        help=("How far to run the pipeline: Stems writes only the binary "
              "stem-map raster; Trees/Nodes also vectorise into a "
              "GeoPackage (default: Nodes)."),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "After processing, run utils.IO.merge_and_filter_tiled_results "
            "on the output folder (useful for tiled processing workflows)."
        ),
    )
    parser.add_argument(
        "--merge-output",
        default=None,
        help=(
            "Optional output .gpkg path for merged results. If omitted, the IO "
            "function chooses a default name in the work directory."
        ),
    )
    parser.add_argument(
        "--edge-buffer-m",
        type=float,
        default=0.0,
        help=(
            "Edge buffer in meters used for tile-edge filtering at the "
            "batch merge (default: 0). Per-ortho seam dedup already "
            "applied; leave 0 unless you know why."
        ),
    )

    args = parser.parse_args(argv)

    if args.list_models:
        print(_format_model_list(load_registry(DEFAULT_CONFIG_PATH)))
        return 0

    if not args.model:
        parser.error("the following arguments are required: model")

    try:
        model_path = resolve_model_path(args.model, args.model_dir)
    except (KeyError, ModelDownloadError) as e:
        print(f"ERROR: {e}")
        return 2

    orthos = list_orthomosaics(args.input)
    if not orthos:
        print(f"No orthomosaics found in {args.input}.")
        return 0

    failures = process_orthos(orthos, model_path, args.output, args.jobs,
                              process_type=args.process_type)
    if failures:
        print(f"\n{len(failures)} of {len(orthos)} orthomosaics FAILED:")
        for path, reason in failures:
            print(f"  {os.path.basename(path)}: {reason}")

    if args.merge:
        try:
            merged = merge_results(
                work_dir=args.output,
                output_gpkg=args.merge_output,
                edge_buffer_m=args.edge_buffer_m,
            )
            print(f"Merged tiled results -> {merged}")
        except Exception as e:
            print(f"Merge failed: {e}")
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
