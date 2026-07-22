#!/usr/bin/env python
from __future__ import annotations

import os
import sys

# Determinism: the vector stage's connect_stems joins stems in an order that
# depends on set-iteration of string-hashed Part objects (docs/CODE_REVIEW_2.md
# A-4). Python salts string hashing per process, so WITHOUT a fixed seed the
# SAME orthomosaic yields a slightly different number of stems on every run.
# Pin the seed (as tests/generate_fixtures.py and tests/conftest.py already do)
# by re-executing once with PYTHONHASHSEED=0 before anything hashes into a set.
# '-u' is re-added so the plugin still gets unbuffered, line-streamed logs.
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
from utils import Log
from utils import Skeletonization as Skel
from utils import Vectorization as Vec
from utils import Quantification as Quant
from utils.Tiling import build_tile_grid, meters_to_pixels

VALID_PROCESS_TYPES = {'Stems', 'Trees', 'Nodes'}


def _import_tensorflow():
    """Return the tensorflow module, or None when it isn't installed.

    Models are ONNX (onnxruntime manages its own devices); TensorFlow is only
    present in dev environments that still load legacy .hdf5 models. All TF
    configuration below is therefore best-effort and skipped when TF is absent.
    """
    try:
        import tensorflow as tf
        return tf
    except Exception:
        return None


LEGACY_KERAS_SUFFIXES = ('.hdf5', '.h5', '.keras')


def _is_legacy_keras_model(model_path) -> bool:
    """True for the standalone/legacy Keras models TensorFlow still loads."""
    return str(model_path or '').lower().endswith(LEGACY_KERAS_SUFFIXES)


def _nvidia_driver_version():
    """The installed NVIDIA driver version, or None when there is no NVIDIA
    GPU. Absence is the normal case on macOS and CPU boxes, not an error."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _configure_tensorflow_runtime(model_path=None):
    """TensorFlow device setup — only relevant for legacy Keras models.

    ONNX models are run by onnxruntime, which manages its own devices, so this
    is a silent no-op on the normal path.
    """
    if not _is_legacy_keras_model(model_path):
        return None
    tf = _import_tensorflow()
    if tf is None:
        return None
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} TensorFlow GPU(s).")
        except RuntimeError as e:
            print(f"Memory growth setup failed: {e}")
    return tf


def _force_cpu_only(model_path=None):
    """Pin inference to the CPU for ``prediction_backend='cpu'``.

    onnxruntime reads WINMOL_ONNX_FORCE_CPU when the session is created, so
    this MUST run before IO.load_model_from_path(). HardwareInfo.detect() has
    already run by then, so the reported accelerator still describes the
    machine rather than this override.
    """
    os.environ['WINMOL_ONNX_FORCE_CPU'] = '1'
    print('Pinned inference to the CPU (WINMOL_ONNX_FORCE_CPU=1).')
    if not _is_legacy_keras_model(model_path):
        return None
    tf = _import_tensorflow()
    if tf is None:
        return None
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("Configured TensorFlow for CPU-only prediction.")
    except RuntimeError as exc:
        print(f"CPU-only TensorFlow setup failed: {exc}")
    return tf


class ImageProcessing:
    def __init__(self, model_path, uav_path, stem_path,
                 trees_path, process_type):
        print("Initializing WINMOL Analyzer")
        self.model_path = model_path
        self.uav_path = uav_path
        self.stem_path = stem_path
        self.trees_path = trees_path
        self.process_type = process_type
        self.config = Config()
        self.apply_env_config_overrides()
        # Resolve verbosity once the overrides are in, and export it so the
        # spawned vector-tile workers inherit the same level.
        Log.configure_from_config(self.config)

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
        label = getattr(hardware, 'accelerator_label', None) or 'CPU'
        if getattr(hardware, 'accelerator', 'cpu') == 'cpu':
            label = 'CPU (no GPU acceleration available)'
        print(
            f"Hardware detected: CPU cores={hardware.cpu_count}, "
            f"RAM={hardware.total_ram_gb} GB, "
            f"accelerator={label}"
        )
        # Only the NVIDIA path has per-device names worth listing.
        if getattr(hardware, 'accelerator', 'cpu') == 'cuda' \
                and hardware.gpu_names:
            print(f"Visible GPUs: {hardware.gpu_names}")
        # The RTX-4080 case: the GPU is right there, but the installed
        # onnxruntime has no CUDA provider, so the run silently crawled on the
        # CPU. Name the hardware and the remedy instead of staying quiet.
        unusable = list(getattr(hardware, 'unusable_gpu_names', []) or [])
        if unusable:
            print(
                f"WARNING: nvidia-smi reports {unusable} but this onnxruntime "
                "build cannot use them (CPUExecutionProvider only), so "
                "inference will run on the CPU. Install onnxruntime-gpu to "
                "use them — see docs/GPU.md."
            )
        return hardware

    def build_plan(self, hardware=None):
        if hardware is None:
            hardware = self.detect_hardware()
        raster_info = IO.get_raster_info(self.uav_path)
        plan = build_execution_plan(
            self.config, hardware, raster_info, self.process_type)

        print("Execution plan:")
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
        self._apply_plan_to_config(plan, hardware)
        return plan

    def _apply_plan_to_config(self, plan, hardware=None):
        # Carry the detected hardware onto the config so the prediction phase
        # can key its autotune cache on it without re-probing nvidia-smi.
        if hardware is not None:
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

    def _correct_accelerator_after_load(self, model):
        """Reconcile the banner with the session that actually got built.

        "Hardware detected: ..." is printed before the model is loaded, so it
        can only state an EXPECTATION. Once onnxruntime has bound its
        providers we know the truth; if it differs, say so and downgrade the
        recorded hardware so nothing downstream keeps sizing a GPU run for a
        CPU session.

        When it MATCHES, say that too. The banner's "(expected; not yet
        verified against a session)" is honest but leaves the log hedging
        forever; one confirmation line here settles it using the observation
        this hook already has, without a second provider check anywhere.
        """
        active_kind = getattr(model, 'accelerator', None)
        if not active_kind:
            return
        hardware = getattr(self.config, 'hardware', None)
        expected = getattr(hardware, 'accelerator', None) if hardware else None
        if expected is None:
            return
        if active_kind == expected:
            label = getattr(model, 'accelerator_label', active_kind)
            print(f"Device confirmed: inference is running on {label} "
                  "(verified against the loaded session).")
            return
        label = getattr(model, 'accelerator_label', active_kind)
        expected_label = getattr(hardware, 'accelerator_label', expected)
        print(
            f"Correction: inference is running on {label} "
            f"(expected {expected_label})."
        )
        hardware.accelerator = active_kind
        hardware.accelerator_label = label
        if active_kind == 'cpu':
            if getattr(hardware, 'gpu_names', None):
                hardware.unusable_gpu_names = list(hardware.gpu_names)
            hardware.gpu_names = []
            hardware.gpu_memory_gb = []
            hardware.gpu_count = 0

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
            # Before the model is loaded: the provider list is fixed when the
            # onnxruntime session is created.
            _force_cpu_only(self.model_path)

        print("\nLoading Model...")
        model = IO.load_model_from_path(self.model_path)
        self._correct_accelerator_after_load(model)
        print("\nPerforming prediction with resampling (stream mode)...")
        profile = Pred.predict_stream_to_raster(
            self.uav_path,
            self.stem_path,
            model,
            self.config,
        )
        return (None, profile, self.stem_path)

    def trees_processing(self, pred, profile):
        print("\nFinding stem segments...")
        segments = Skel.find_segments(pred, self.config, profile)
        print("\nRestoring geoinformation...")
        segments = Vec.restore_geoinformation(segments, self.config, profile)
        print("\nBuilding stem parts...")
        stems = Vec.build_stem_parts(segments)
        print("\nConnecting stem parts...")
        stems = Vec.connect_stems(stems, self.config)
        print("\nRebuilding end nodes...")
        Vec.rebuild_endnodes_from_stems(stems)
        print("\nQuantifying stems...")
        stems = Quant.quantify_stems(stems, pred, profile, config=self.config)
        # Un-tiled path: connect_stems ran once over the whole raster, so
        # this count is the run's answer. The tiled path's answer is the
        # merge stage's "Total stems written" instead.
        print("")
        print("STEM SUMMARY (final result for this run)")
        print(f"Total stems:           {len(stems)}")
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
            vector_tile_px = int(plan.tile_inner_px) + 2 * int(halo_px)
            print(
                f"Prepared {len(tile_paths)}/{len(jobs)} vector tiles "
                f"with foreground | skipped_empty {skipped_tiles}"
            )
            # A vector tile is a completely different unit from a
            # prediction tile — inner 4096 px plus halo against ~727 px —
            # and the log used to call both of them "tile". Standalone,
            # unparsed line: run_progress.py keys off the "Prepared n/m"
            # line above, which is untouched.
            print(
                f"VECTOR PHASE | {len(tile_paths)} vector tiles | "
                f"~{vector_tile_px}x{vector_tile_px} px each (inner "
                f"{int(plan.tile_inner_px)} + halo {int(halo_px)} per "
                f"side, clipped at the raster edge)",
                flush=True,
            )
            if not tile_paths:
                print("No foreground tiles found for the vector stage.")
                return None
            from utils.VectorTilePipeline import process_prediction_tiles

            process_prediction_tiles(
                tile_paths,
                self.config,
                self.process_type,
                work_dir,
                plan.cpu_workers,
                tile_px=vector_tile_px,
            )
            merged = self.run_merge_phase(plan, work_dir)
            if plan.keep_temp:
                print(f"Keeping tile work directory: {work_dir}")
            else:
                shutil.rmtree(work_dir, ignore_errors=True)
            return merged
        except Exception:
            # Never delete completed tile results on failure: they may
            # represent hours of work and allow inspection/resume
            # (docs/CODE_REVIEW_2.md A-15).
            print(f"Vector phase failed; keeping tile work directory "
                  f"for inspection: {work_dir}")
            raise

    def run_merge_phase(self, plan, work_dir):
        out_path = self.trees_path if self.trees_path.lower().endswith(
            '.gpkg') else f"{self.trees_path}.gpkg"
        return IO.merge_and_filter_tiled_results(
            work_dir=work_dir,
            output_gpkg=out_path,
            edge_buffer_m=plan.tile_overlap_m,
            config=self.config,
            # Pass the full stem-map extent so stems on the ortho's true outer
            # boundary (corners) aren't trimmed by the interior-seam dedup.
            stem_map_path=self.stem_path,
        )

    def run_stem_pipeline(self, plan):
        self.run_prediction_phase(plan)

    def run_tree_pipeline(self, plan):
        pred, profile, pred_path = self.run_prediction_phase(plan)
        return self.run_vector_phase(
            plan, pred_path=pred_path, pred=pred, profile=profile)

    def report_runtime_env(self):
        """Report the runtime that actually performs inference.

        Models are ONNX and run through onnxruntime; TensorFlow/CUDA versions
        say nothing about that, and TensorFlow being absent is the normal,
        expected state of the plugin environment (requirements/cpu.txt).
        """
        print("Environment:")
        try:
            from utils import onnx_runtime
            report = onnx_runtime.runtime_report()
        except Exception as exc:
            print(f"  Inference runtime: onnxruntime unavailable ({exc})")
            report = None

        if report is not None:
            print("  Inference runtime: onnxruntime "
                  f"{report['onnxruntime_version']}")
            print("  Available providers: "
                  + ", ".join(report['available_providers']))
            selected = "  Selected providers: " + ", ".join(
                report['selected_providers'])
            if report['override']:
                selected += f" (forced by {report['override']})"
            print(selected)
            # Prefer the OBSERVED session over the requested list. A requested
            # provider that failed to bind would otherwise be reported as the
            # device, which is exactly the lie this guards against.
            #
            # Only trust the observation if it describes THIS provider
            # request: a report left behind by a session built under a
            # different configuration (batch runs load a model per image) says
            # nothing about the run being reported now.
            active = onnx_runtime.last_active_report()
            if active is not None and (
                    list(active.get('requested_providers') or [])
                    != list(report['selected_providers'])):
                active = None
            if active is not None:
                print("  Active providers: "
                      + ", ".join(active['active_providers']))
                print(f"  Device: {active['accelerator_label']} (verified)")
            else:
                print(f"  Device: {report['accelerator_label']} "
                      "(expected; not yet verified against a session)")

        driver = _nvidia_driver_version()
        if driver:
            print(f"  NVIDIA driver: {driver}")

        # TensorFlow matters only for the legacy .hdf5/.keras models the
        # standalone path can still load. Never report its absence as a fault.
        if _is_legacy_keras_model(self.model_path):
            tf = _import_tensorflow()
            if tf is not None:
                print("  Legacy Keras model path: TensorFlow "
                      f"{tf.__version__}")
            else:
                print("  Legacy Keras model requested but TensorFlow is not "
                      "installed.")

    def display_starting_text(self, plan=None):
        if plan is not None and plan.prediction_mode == 'multi_gpu_stream':
            print(
                "Skipping parent runtime initialization "
                "for worker-local multi-GPU mode."
            )
        else:
            self.report_runtime_env()
        print("Command-line arguments:")
        print(f"Model path: {self.model_path}")
        print(f"Image path: {self.uav_path}")
        print(f"Semantic stem map path: {self.stem_path}")
        print(f"Process type: {self.process_type}")
        if self.trees_path:
            print(f"Detected wind-thrown trees path: {self.trees_path}")
        self.config.display()

    def main(self):
        hardware = self.detect_hardware()
        plan = self.build_plan(hardware)
        if self.process_type == 'Stems':
            self.run_stem_pipeline(plan)
        else:
            self.run_tree_pipeline(plan)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("""Usage:
            python3 -u winmol_run.py <model_path> <input_tiff> <stem_map_tiff>
                                     <output_prefix> <Stems|Trees|Nodes>""")
        print(f"Received {len(sys.argv) - 1} arguments: {sys.argv[1:]}")
        sys.exit(2)

    tt = Timer()
    tt.start()
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
    if plan.prediction_mode == 'multi_gpu_stream' and plan.gpu_workers > 1:
        print(
            "Skipping parent runtime configuration "
            "for worker-local multi-GPU mode."
        )
    else:
        _configure_tensorflow_runtime(model_path)
    image_processor.display_starting_text(plan)
    if process_type == 'Stems':
        image_processor.run_stem_pipeline(plan)
    else:
        image_processor.run_tree_pipeline(plan)

    print(f"Total runtime: {tt.stop():.1f} s")
