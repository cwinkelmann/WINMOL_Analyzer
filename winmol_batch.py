#!/usr/bin/env python3
"""Batch runner for WINMOL Analyzer.

- Reads available models from config.json (repo root) — the model
  registry (schema v2 with families/variants/checksums, or a legacy flat
  {name: url} map). See --list-models.
- Downloads the resolved model on demand (checksum-verified, atomic)
  unless --no-download forbids network access.
- Runs winmol_run.py for all *.tif / *.tiff in the input folder.
- Optionally merges tiled outputs with utils.IO.merge_and_filter_tiled_results.
"""

import argparse
import concurrent.futures
import os
import queue
import subprocess
import sys
from typing import Dict, List, Optional

from plugin_utils import model_registry


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
    """Map every registry model id to its expected local file path.

    Backed by plugin_utils.model_registry. Schema-v2 registries use each
    entry's mandatory ``file`` name (always the URL basename), unifying
    the plugin's and the CLI's on-disk naming. Legacy flat {name: url}
    configs keep the historical URL-basename mapping (url_to_filename)
    byte-identically. Never downloads and never checks existence — it
    only computes expected paths. Raises FileNotFoundError / ValueError
    exactly as before.
    """
    registry = model_registry.load_registry(config_path)

    if registry.schema >= 2:
        return {mid: model_registry.local_path(entry, model_dir)
                for mid, entry in registry.entries.items()}

    model_paths: Dict[str, str] = {}
    for mid, entry in registry.entries.items():
        if not entry.url:
            continue    # legacy: entries with non-string urls are skipped
        model_paths[mid] = os.path.join(model_dir,
                                        url_to_filename(entry.url))

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


def _resolve_model_name(names):
    """argparse `type` that accepts a model or family name in any case.

    The model names originally lived in a hardcoded lowercase dict
    (`spruce` / `beech` / `general`). When they moved to config.json keys they
    became capitalised, which silently broke every existing invocation --
    `winmol_batch.py general` had worked since 2025 and started erroring.
    Matching case-insensitively keeps those callers (and the Dockerfiles in the
    repo root) working, while the canonical capitalised names are what gets
    used internally.

    Exact ids win before the case-insensitive fallback, and entry ids win
    over family ids in that fallback (family "unet_pt" and entry
    "UNet_PT" collide case-insensitively) — mirroring Registry.resolve.
    """
    exact = set(names)
    # build family->entry overwrite order: later wins, so list entries
    # last to give them case-insensitive precedence.
    lookup = {name.lower(): name for name in reversed(list(names))}

    def _resolve(value):
        if value in exact:
            return value
        try:
            return lookup[value.lower()]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"invalid model {value!r}; choose from "
                + ", ".join(sorted(names)))

    return _resolve


def _stderr_progress(done, total, entry):
    """Carriage-return download progress line on stderr."""
    mb = done / 1e6
    if total:
        pct = int(done * 100 / total)
        line = (f"\r  {entry.file}  {pct:3d}%  "
                f"({mb:.1f}/{total / 1e6:.1f} MB)")
    else:
        line = f"\r  {entry.file}  {mb:.1f} MB"
    print(line, end="", file=sys.stderr, flush=True)


def _manual_hint(entry, model_dir):
    """The manual-download command matching this entry's hosting source."""
    if "WINMOL_segmentor_pt" in entry.url:
        return ("  gh release download models-v1 "
                "--repo cwinkelmann/WINMOL_segmentor_pt "
                f"-p {entry.file} --dir {model_dir}")
    if "zenodo.org" in entry.url:
        dest = os.path.join(model_dir, entry.file)
        return f"  curl -L -o {dest} '{entry.url}'"
    return f"  curl -L -o {os.path.join(model_dir, entry.file)} '{entry.url}'"


def _print_models(registry, model_dir):
    """--list-models: one line per model id, then the family ids."""
    print(f"{'ID':<26} {'BACKEND':<7} {'SIZE':>9} {'F1':>6} "
          f"{'STATE':<10} LABEL")
    for e in registry.entries.values():
        installed = os.path.exists(
            model_registry.local_path(e, model_dir))
        size = f"{e.size_mb:.1f} MB" if e.size_mb else "-"
        f1 = f"{e.f1:.3f}" if e.f1 else "-"
        state = "installed" if installed else "-"
        print(f"{e.id:<26} {e.backend:<7} {size:>9} {f1:>6} "
              f"{state:<10} {e.label}")
    if registry.families:
        print("\nFamily ids (auto-pick a device variant, see --variant):")
        for fam in registry.families.values():
            print(f"  {fam.id:<24} {fam.label}")


def main(argv: List[str]) -> int:
    # Only the NAMES are needed here (for the vocabulary), and those come
    # from config.json, not from the model directory — so the default dir
    # is fine for building the parser even when --model-dir overrides it
    # below. Schema v2 adds the family ids (e.g. unet_pt) to the model
    # ids as accepted names.
    registry = model_registry.load_registry(DEFAULT_CONFIG_PATH)
    names = list(registry.entries) + list(registry.families)

    parser = argparse.ArgumentParser(
        description=(
            "Batch process orthomosaics in a folder using WINMOL Analyzer. "
            "Available models are loaded from the registry in config.json "
            "(see --list-models)."
        )
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=_resolve_model_name(names),
        help=("Model id or family id (from config.json): "
              + ", ".join(sorted(names)) + ". Case-insensitive."),
        metavar="MODEL",
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
    parser.add_argument(
        "--variant",
        choices=["auto", "default", "fp32", "int8", "fp16"],
        default="auto",
        help=(
            "Precision variant when MODEL is a family id (e.g. unet_pt): "
            "'auto' substitutes the device variant only when it is "
            "certified lossless; 'default' takes the device variant "
            "(int8 on CPU, fp16 on GPU) regardless — what the GUI opens "
            "on; fp32/int8/fp16 force one. Explicit model ids (e.g. "
            "Spruce_Deadwood, UNet_PT_int8) are never rewritten."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Never touch the network: fail with exit code 2 when the "
            "resolved model file is absent or fails checksum verification."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["quiet", "normal", "debug"],
        default=None,
        help=(
            "Verbosity of every child run (default: $WINMOL_LOG_LEVEL or "
            "'normal'). 'debug' restores the per-tile MERGE/VECTOR "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help=(
            "List the registry (model ids, backend, size, F1, installed "
            "state, family ids) and exit."
        ),
    )

    args = parser.parse_args(argv)

    if args.log_level:
        # Children copy os.environ, so this reaches winmol_run.py and, from
        # there, the spawned vector-tile workers.
        os.environ["WINMOL_LOG_LEVEL"] = args.log_level

    if args.list_models:
        _print_models(registry, args.model_dir)
        return 0
    if not args.model:
        parser.error("the model argument is required (or --list-models)")

    if registry.schema >= 2:
        try:
            entry = registry.resolve(
                args.model, device=model_registry.detect_device(),
                variant=args.variant)
        except (KeyError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 2
        progressed = []

        def _progress(done, total, e):
            progressed.append(True)
            _stderr_progress(done, total, e)

        try:
            model_path = model_registry.ensure_model(
                entry, args.model_dir, progress=_progress,
                allow_download=not args.no_download)
        except model_registry.ModelDownloadError as exc:
            print(
                f"ERROR: {exc}\n"
                f"Looked in: {args.model_dir}\n"
                "Point --model-dir (or $WINMOL_MODEL_DIR) at the "
                f"directory holding {entry.file}, or fetch it manually:\n"
                + _manual_hint(entry, args.model_dir)
            )
            return 2
        finally:
            if progressed:
                print(file=sys.stderr)   # newline after the \r progress
    else:
        # Legacy flat config: resolve against the chosen directory now
        # that --model-dir is known; no downloads (historic behavior).
        model_paths = load_model_paths(model_dir=args.model_dir)
        model_path = model_paths[args.model]
        if not os.path.exists(model_path):
            print(
                f"ERROR: Model file not found: {model_path}\n"
                f"Looked in: {args.model_dir}\n"
                "Point --model-dir (or $WINMOL_MODEL_DIR) at the directory "
                "holding the models named in config.json, or fetch them "
                "with:\n"
                "  gh release download models-v1 "
                "--repo cwinkelmann/WINMOL_segmentor_pt --dir <dir>"
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
