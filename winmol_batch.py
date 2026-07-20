#!/usr/bin/env python3
"""Batch runner for WINMOL Analyzer.

- Reads available models from config.json (repo root).
- Runs winmol_run.py for all *.tif / *.tiff in the input folder.
- Optionally merges tiled outputs with utils.IO.merge_and_filter_tiled_results.
"""

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
from typing import Dict, List, Optional


DEFAULT_INPUT_FOLDER = "./standalone/input"
DEFAULT_OUTPUT_FOLDER = "./standalone/output"
# config.json now names .onnx files, which live in standalone/model_onnx (where
# scripts/convert_models_to_onnx.py writes them and every test looks).
# standalone/model holds the legacy Keras .hdf5 originals, so the old default
# pointed at a directory that no longer contains what config.json describes.
DEFAULT_MODEL_DIR = "./standalone/model_onnx"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def url_to_filename(url: str) -> str:
    """Turn a config.json URL into the local model filename."""
    return url.split("/")[-1].split("?")[0]


def load_model_paths(
    config_path: str = DEFAULT_CONFIG_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> Dict[str, str]:
    """Load model names from config.json and map them to local model paths."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "config.json not found at: "
            f"{config_path} (expected repo root; next to winmol_batch.py)"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"Invalid/empty config.json: {config_path}")

    model_paths: Dict[str, str] = {}
    for model_name, url in cfg.items():
        if not isinstance(model_name, str) or not isinstance(url, str):
            continue
        model_paths[model_name] = os.path.join(model_dir, url_to_filename(url))

    if not model_paths:
        raise ValueError(
            "No model entries found in config.json. Expected {name: url}."
        )

    return model_paths


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
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:
        pass
    return 0


def run_winmol(input_image: str, model_path: str, output_folder: str,
               gpu_id: Optional[int] = None) -> None:
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
        "Nodes",
    ]

    env = dict(os.environ)
    if gpu_id is not None:
        # Pin this ortho to one GPU. The child then plans for a SINGLE GPU —
        # the well-tested path — instead of every concurrent job trying to
        # spread itself across all of them and contending.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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


def process_orthos(orthos, model_path, output_folder, jobs=1):
    """Run every orthomosaic, optionally several at once.

    Returns a list of (path, reason) for those that failed — the batch always
    attempts all of them. Previously a single failure raised out of the loop
    and abandoned the rest, so one bad file in an overnight batch of twenty
    cost the other nineteen.

    Why parallelise across ORTHOS rather than harder within one: a single
    orthomosaic cannot use many GPUs (the planner caps GPU workers by tile
    count, and under 1000 tiles that is two), and its vector phase is CPU-bound
    anyway, so extra GPUs do not help it. Whole orthomosaics are independent —
    embarrassingly parallel, no coordination, and each child takes the
    single-GPU path. It also overlaps one job's CPU-bound vector phase with
    another's GPU-bound prediction, which is worth something even on ONE GPU.
    """
    failures = []
    jobs = max(1, int(jobs))

    if jobs == 1 or len(orthos) == 1:
        for ortho in orthos:
            try:
                run_winmol(ortho, model_path, output_folder)
            except subprocess.CalledProcessError as e:
                print(f"  FAILED: {ortho}: {e}", flush=True)
                failures.append((ortho, str(e)))
        return failures

    gpus = detect_gpu_count()
    # Worker slot -> GPU. With more slots than GPUs they share, which is
    # deliberate: prediction and vectorisation alternate, so a GPU is idle for
    # much of each job.
    slots = queue.Queue()
    for i in range(jobs):
        slots.put(i % gpus if gpus else None)

    print(f"Processing {len(orthos)} orthomosaics, {jobs} at a time"
          + (f" across {gpus} GPU(s)" if gpus else " (no GPU detected)"),
          flush=True)

    def _one(ortho):
        slot = slots.get()
        try:
            run_winmol(ortho, model_path, output_folder, gpu_id=slot)
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


def _resolve_model_name(model_paths):
    """argparse `type` that accepts a model name in any case.

    The model names originally lived in a hardcoded lowercase dict
    (`spruce` / `beech` / `general`). When they moved to config.json keys they
    became capitalised, which silently broke every existing invocation --
    `winmol_batch.py general` had worked since 2025 and started erroring.
    Matching case-insensitively keeps those callers (and the Dockerfiles in the
    repo root) working, while the canonical capitalised names are what gets
    used internally.
    """
    lookup = {name.lower(): name for name in model_paths}

    def _resolve(value):
        try:
            return lookup[value.lower()]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"invalid model {value!r}; choose from "
                + ", ".join(sorted(model_paths)))

    return _resolve


def main(argv: List[str]) -> int:
    # Only the KEYS are needed here (for `choices`), and those come from
    # config.json, not from the model directory — so the default dir is fine
    # for building the parser even when --model-dir overrides it below.
    model_paths = load_model_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Batch process orthomosaics in a folder using WINMOL Analyzer. "
            "Available models are loaded from config.json."
        )
    )
    parser.add_argument(
        "model",
        type=_resolve_model_name(model_paths),
        help=("Model to use (from config.json): "
              + ", ".join(sorted(model_paths)) + ". Case-insensitive."),
        metavar="{" + ",".join(sorted(model_paths)) + "}",
    )
    parser.add_argument(
        "--input",
        default=os.environ.get("WINMOL_INPUT_DIR", DEFAULT_INPUT_FOLDER),
        help=("Input folder (default: $WINMOL_INPUT_DIR or "
              f"{DEFAULT_INPUT_FOLDER})"),
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("WINMOL_OUTPUT_DIR", DEFAULT_OUTPUT_FOLDER),
        help=("Output folder (default: $WINMOL_OUTPUT_DIR or "
              f"{DEFAULT_OUTPUT_FOLDER})"),
    )

    parser.add_argument(
        "--model-dir",
        default=os.environ.get("WINMOL_MODEL_DIR", DEFAULT_MODEL_DIR),
        help=(
            "Directory holding the model files named in config.json "
            f"(default: $WINMOL_MODEL_DIR or {DEFAULT_MODEL_DIR}). Set this "
            "when the models live elsewhere — e.g. a directory mounted into a "
            "container."
        ),
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help=(
            "Process this many orthomosaics concurrently (default: 1). Each "
            "job is pinned to one GPU via CUDA_VISIBLE_DEVICES, so a machine "
            "with N GPUs can run N orthomosaics at once. Values above the GPU "
            "count still help: prediction and vectorisation alternate, so one "
            "job's CPU-bound vector phase overlaps another's GPU work."
        ),
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
        default=1.0,
        help="Edge buffer in meters used for tile-edge filtering (default: 1)",
    )

    args = parser.parse_args(argv)

    # Re-resolve against the chosen directory now that --model-dir is known.
    model_paths = load_model_paths(model_dir=args.model_dir)
    model_path = model_paths[args.model]
    if not os.path.exists(model_path):
        print(
            f"ERROR: Model file not found: {model_path}\n"
            f"Looked in: {args.model_dir}\n"
            "Point --model-dir (or $WINMOL_MODEL_DIR) at the directory holding "
            "the models named in config.json, or fetch them with:\n"
            "  gh release download models-onnx-v1 "
            "--repo cwinkelmann/WINMOL_Analyzer --dir <dir>"
        )
        return 2

    orthos = list_orthomosaics(args.input)
    if not orthos:
        print(f"No orthomosaics found in {args.input}.")
        return 0

    failures = process_orthos(orthos, model_path, args.output, args.jobs)
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
