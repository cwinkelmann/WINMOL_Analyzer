#!/usr/bin/env python
from __future__ import annotations

import os
import sys

# connect_stems joins stems in the set-iteration order of string-hashed Part
# objects, which Python salts per process — so re-exec once with a pinned
# PYTHONHASHSEED before anything hashes into a set ('-u' re-added to keep the
# plugin's log stream unbuffered).
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)

import json
import shutil
import subprocess
import tempfile

from classes.Config import Config
from classes.ExecutionPlan import build_execution_plan
from classes.HardwareInfo import HardwareInfo
from classes.Timer import Timer
from utils import IO
from utils import Skeletonization as Skel
from utils import Vectorization as Vec
from utils import Quantification as Quant
from utils.Tiling import build_tile_grid, meters_to_pixels

VALID_PROCESS_TYPES = {'Stems', 'Trees', 'Nodes'}

#: Providers that mean "a GPU/accelerator is available" — CUDA on NVIDIA,
#: CoreML on Apple Silicon (Metal/ANE).
_ACCELERATOR_PROVIDERS = ("CUDAExecutionProvider", "CoreMLExecutionProvider")


def _cpu_stream_forces_onnx_cpu(prediction_backend, selected_providers):
    """In cpu_stream mode, should the ONNX runtime be pinned to the CPU
    provider?

    cpu_stream is the single-device streaming path the planner picks when
    there is no CUDA GPU. On Apple Silicon it is chosen for lack of CUDA,
    but CoreML is still a real accelerator (18x faster than CPU here) and
    must NOT be disabled. So force the CPU provider only when the user
    explicitly asked for the ``cpu`` backend, or the machine offers no
    accelerator at all (CUDA or CoreML)."""
    if str(prediction_backend).lower() == "cpu":
        return True
    return not any(p in _ACCELERATOR_PROVIDERS for p in selected_providers)


class ImageProcessing:
    def __init__(self, model_path, uav_path, stem_path,
                 trees_path, process_type):
        print("Initialization")
        self.model_path = model_path
        self.uav_path = uav_path
        self.stem_path = stem_path
        self.trees_path = trees_path
        self.process_type = process_type
        self.config = Config()
        self.apply_env_config_overrides()

    def apply_env_config_overrides(self):
        raw = os.environ.get("WINMOL_CONFIG_OVERRIDES_JSON", "").strip()
        if not raw:
            return

        try:
            overrides = json.loads(raw)
        except Exception as exc:
            print(f"Invalid WINMOL_CONFIG_OVERRIDES_JSON: {exc}")
            sys.exit(2)

        if not isinstance(overrides, dict):
            print(
                "Invalid WINMOL_CONFIG_OVERRIDES_JSON: "
                "top-level JSON must be an object."
            )
            sys.exit(2)

        print("Applying config overrides from WINMOL_CONFIG_OVERRIDES_JSON:")
        for key, value in overrides.items():
            if not hasattr(self.config, key):
                print(f"  [skip] unknown config key: {key}")
                continue
            setattr(self.config, key, value)
            print(f"  {key:30} {value}")

    def detect_hardware(self):
        hardware = HardwareInfo.detect()
        print(
            f"Hardware detected: CPUs={hardware.cpu_count}, "
            f"RAM={hardware.total_ram_gb} GB, "
            f"GPUs={hardware.gpu_count}"
        )
        if hardware.gpu_names:
            print("Visible GPUs:", hardware.gpu_names)
        return hardware

    def build_plan(self, hardware=None):
        if hardware is None:
            hardware = self.detect_hardware()
        raster_info = IO.get_raster_info(self.uav_path)
        plan = build_execution_plan(
            self.config, hardware, raster_info, self.process_type)

        print('Execution plan:')
        print(f"  process_type     = {plan.process_type}")
        print(f"  prediction_mode  = {plan.prediction_mode}")
        print(f"  vector_mode      = {plan.vector_mode}")
        print(f"  tile_inner_px    = {plan.tile_inner_px}")
        print(f"  tile_overlap_m   = {plan.tile_overlap_m}")
        print(f"  halo_px          = {plan.halo_px}")
        print(f"  gpu_workers      = {plan.gpu_workers}")
        print(f"  cpu_workers      = {plan.cpu_workers}")
        print(f"  vector_tile_workers = {plan.vector_tile_workers}")
        print(f"  vector_inner_workers = {plan.vector_inner_workers}")
        print(f"  prediction_batch = {plan.prediction_batch_size}")
        print(f"  queue_batches    = {plan.producer_queue_batches}")
        print(f"  producer_workers = {plan.producer_workers}")
        print(f"  progress_interval_s = {plan.progress_interval_s}")
        print(f"  est_pred_tiles   = {plan.estimated_prediction_tiles}")
        # Say so when the planner overrode a configured value. These caps
        # used to be silent, which is how a configured
        # prediction_producer_workers_gpu=6 ran as 3, and the vector pool
        # ran on 2 of 12 cores, without anyone noticing they were capped.
        for note in getattr(plan, 'capped', None) or []:
            print(f"  WARNING: {note}")
        self._apply_plan_to_config(plan, hardware)
        return plan

    def _apply_plan_to_config(self, plan, hardware):
        # Carry the detected hardware onto the config so the prediction
        # phase can key its autotune cache on it without re-probing
        # nvidia-smi.
        self.config.hardware = hardware
        self.config.cpu_workers = (
            plan.vector_inner_workers
            if plan.vector_mode == 'tiled'
            else plan.cpu_workers
        )
        self.config.gpu_workers = plan.gpu_workers
        self.config.vector_mode = plan.vector_mode
        self.config.vector_tile_workers = plan.vector_tile_workers
        self.config.prediction_batch_size = plan.prediction_batch_size
        self.config.producer_queue_batches = plan.producer_queue_batches
        self.config.prediction_producer_workers = plan.producer_workers
        self.config.progress_interval_s = plan.progress_interval_s

    def run_prediction_phase(self, plan):
        if plan.prediction_mode == 'multi_gpu_stream' and plan.gpu_workers > 1:
            from utils.PredictWorkers import run_multi_gpu_prediction

            print("\nPerforming multi-GPU streamed prediction...")
            profile = run_multi_gpu_prediction(
                self.model_path,
                self.uav_path,
                self.stem_path,
                tile_jobs=None,
                gpu_ids=list(range(plan.gpu_workers)),
                config=self.config,
            )
            return (None, profile, self.stem_path)

        from utils import Prediction as Pred

        if plan.prediction_mode == 'cpu_stream':
            from utils.onnx_runtime import selected_providers
            if _cpu_stream_forces_onnx_cpu(
                    getattr(self.config, 'prediction_backend', 'auto'),
                    selected_providers()):
                os.environ["WINMOL_ONNX_FORCE_CPU"] = "1"

        print("\nLoading Model...")
        model = IO.load_model_from_path(self.model_path, self.config)
        from utils.onnx_runtime import last_active_report
        report = last_active_report()
        if report:
            print(
                f"Execution providers (active): "
                f"{report['active_providers']} "
                f"(device: {report['accelerator_label']})")
        print("\nPerforming Prediction with Resampling in stream mode...")
        profile = Pred.predict_stream_to_raster(
            self.uav_path,
            self.stem_path,
            model,
            self.config,
        )
        return (None, profile, self.stem_path)

    def trees_processing(self, pred, profile):
        print("\nFinding Stem Segments...")
        segments = Skel.find_segments(pred, self.config, profile)
        print("\nRestoring Geoinformation...")
        segments = Vec.restore_geoinformation(segments, self.config, profile)
        print("\nBuilding Stem Parts...")
        stems = Vec.build_stem_parts(segments)
        print("\nConnecting Stem Parts...")
        stems = Vec.connect_stems(stems, self.config)
        print("\nRebuilding End Nodes...")
        Vec.rebuild_endnodes_from_stems(stems)
        print("\nQuantifying Stems...")
        stems = Quant.quantify_stems(stems, pred, profile, config=self.config)
        return stems

    def run_vector_phase(self, plan, pred_path=None, pred=None, profile=None):
        if self.process_type == 'Stems':
            return None

        print("\nRunning tiled vector processing...")
        work_dir = tempfile.mkdtemp(
            prefix='winmol_tiles_',
            dir=os.path.dirname(self.trees_path) or None)
        try:
            raster_info = IO.get_raster_info(pred_path or self.stem_path)
            halo_px = meters_to_pixels(
                plan.tile_overlap_m,
                raster_info['pixel_size_x'],
                raster_info['pixel_size_y'],
            )
            jobs = build_tile_grid(
                raster_info['width'],
                raster_info['height'],
                plan.tile_inner_px,
                halo_px,
            )
            tile_paths = []
            skipped_tiles = 0
            for job in jobs:
                pred_tile, tile_profile = IO.load_raster_window_with_profile(
                    pred_path or self.stem_path, job.halo_window)
                pred_arr = pred_tile if hasattr(pred_tile, 'size') else None
                if (
                    pred_arr is None
                    or pred_arr.size == 0
                    or not (pred_arr >= 1).any()
                ):
                    skipped_tiles += 1
                    continue
                tile_path = os.path.join(
                    work_dir, f"{job.tile_id}_roi_stem_map.tif")
                IO.write_tile_raster(pred_tile, tile_profile, tile_path)
                tile_paths.append(tile_path)
            print(
                f"Prepared {len(tile_paths)}/{len(jobs)} vector tiles "
                f"with foreground | skipped_empty {skipped_tiles}"
            )
            if not tile_paths:
                print("No foreground tiles found for vector stage.")
                return None
            from utils.VectorTilePipeline import process_prediction_tiles

            process_prediction_tiles(
                tile_paths,
                self.config,
                self.process_type,
                work_dir,
                plan.cpu_workers,
            )
            merged = self.run_merge_phase(plan, work_dir)
            if plan.keep_temp:
                print(f"Keeping tile work directory: {work_dir}")
            else:
                shutil.rmtree(work_dir, ignore_errors=True)
            return merged
        except Exception:
            if not plan.keep_temp:
                shutil.rmtree(work_dir, ignore_errors=True)
            raise

    def run_merge_phase(self, plan, work_dir):
        out_path = self.trees_path if self.trees_path.lower().endswith(
            '.gpkg') else f"{self.trees_path}.gpkg"
        return IO.merge_and_filter_tiled_results(
            work_dir=work_dir,
            output_gpkg=out_path,
            edge_buffer_m=plan.tile_overlap_m,
            config=self.config,
            # Pass the full stem-map extent so stems on the ortho's true
            # outer boundary (corners) aren't trimmed by the interior-seam
            # dedup.
            stem_map_path=self.stem_path,
        )

    def run_stem_pipeline(self, plan):
        self.run_prediction_phase(plan)

    def run_tree_pipeline(self, plan):
        pred, profile, pred_path = self.run_prediction_phase(plan)
        return self.run_vector_phase(
            plan, pred_path=pred_path, pred=pred, profile=profile)

    def check_DL_env(self):
        def get_nvidia_driver_version():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version",
                     "--format=csv,noheader"], stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    print(
                        f"NVIDIA GPU Driver Version: {result.stdout.strip()}")
                else:
                    print("Failed to retrieve NVIDIA driver version.")
            except FileNotFoundError:
                print("No NVIDIA GPU available or drivers not installed.")

        get_nvidia_driver_version()
        try:
            import onnxruntime as ort
            from utils.onnx_runtime import selected_providers
            print("ONNX Runtime version:", ort.__version__)
            print("Available execution providers:",
                  ort.get_available_providers())
            print("Selected execution providers:", selected_providers())
        except Exception as e:
            print("ONNX Runtime error: ", e)

    def display_starting_text(self, plan=None):
        print("Check CUDA environment")
        self.check_DL_env()
        print("Command-line arguments:")
        print("Model Path:", self.model_path)
        print("Image Path:", self.uav_path)
        print("Semantic Stem Map Path:", self.stem_path)
        print("Process type:", self.process_type)
        if self.trees_path:
            print("Detected Wind-thrown Trees Path:", self.trees_path)
        self.config.display()

    def main(self):
        hardware = self.detect_hardware()
        plan = self.build_plan(hardware)
        if self.process_type == 'Stems':
            self.run_stem_pipeline(plan)
        else:
            self.run_tree_pipeline(plan)


if __name__ == '__main__':
    print(f"Determinism: PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}",
          flush=True)
    if len(sys.argv) != 6:
        print("""Usage:
            python3 -u winmol_run.py <model_path> <input_tiff> <stem_map_tiff>
                                     <output_prefix> <Stems|Trees|Nodes>""")
        print(f"Received {len(sys.argv) - 1} arguments: {sys.argv[1:]}")
        sys.exit(2)

    tt = Timer()
    tt.start()
    print("Start timer")
    model_path = str(sys.argv[1])
    uav_path = str(sys.argv[2])
    stem_path = str(sys.argv[3])
    trees_path = str(sys.argv[4])
    process_type = str(sys.argv[5])

    valid_process_types = {"Stems", "Trees", "Nodes"}
    if process_type not in valid_process_types:
        print(f"Invalid process type: {process_type}")
        print(f"Allowed values: {sorted(valid_process_types)}")
        sys.exit(2)

    image_processor = ImageProcessing(
        model_path, uav_path, stem_path, trees_path, process_type)
    hardware = image_processor.detect_hardware()
    plan = image_processor.build_plan(hardware)
    image_processor.display_starting_text(plan)
    if process_type == 'Stems':
        image_processor.run_stem_pipeline(plan)
    else:
        image_processor.run_tree_pipeline(plan)

    print("Stop timer")
    tt.stop()
