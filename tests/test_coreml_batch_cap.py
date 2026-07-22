"""CoreML must not be handed a memory-derived batch size.

The planner sizes the prediction micro-batch from GPU memory, and on Apple
Silicon HardwareInfo reports half of unified RAM as "GPU memory" — so a 24 GB
Mac lands in the >=12 GB tier and gets batch 4, a 64 GB Mac batch 8. On CUDA
that is right; on CoreML it is backwards. Measured on this tree (M2,
onnxruntime 1.27.0, standalone/model_onnx/Spruce.onnx, fresh process per row):

    batch 1 -> 0.228 s/image
    batch 2 -> 0.285 s/image
    batch 6 -> 0.435 s/image
    batch 8 -> 0.498 s/image

Per-image cost RISES with batch size, so the planner's default was ~2x slower
per tile than batch 1 — this is the user's "6-7 s/tile on macOS". Batch size
does not change the prediction (max abs diff 0.0 between b1, b4 and b8), so the
cap is a pure throughput fix.
"""
import pytest

from classes.Config import Config
from classes.ExecutionPlan import build_execution_plan
from classes.HardwareInfo import HardwareInfo
from utils import Prediction as Pred


class Raster:
    width = 20000
    height = 20000
    count = 3
    pixel_size_x = 0.02
    pixel_size_y = 0.02
    dtype = "uint8"
    estimated_input_gb = 2.0


def mac(total_ram_gb=64.0):
    """An Apple Silicon box as HardwareInfo.detect() would describe it."""
    return HardwareInfo(
        cpu_count=10,
        total_ram_gb=total_ram_gb,
        gpu_count=1,
        gpu_names=["Apple Silicon GPU (Metal/CoreML)"],
        gpu_memory_gb=[total_ram_gb * 0.5],
        accelerator="coreml",
        accelerator_label="Apple Silicon GPU (Metal/CoreML)",
    )


def nvidia(mem_gb=24.0):
    return HardwareInfo(
        cpu_count=16,
        total_ram_gb=64.0,
        gpu_count=1,
        gpu_names=["NVIDIA GeForce RTX 4080"],
        gpu_memory_gb=[mem_gb],
        accelerator="cuda",
        accelerator_label="NVIDIA GPU (CUDA)",
    )


def plan_for(hardware, process_type="Trees", **overrides):
    config = Config()
    for key, value in overrides.items():
        setattr(config, key, value)
    return build_execution_plan(config, hardware, Raster(), process_type)


def test_coreml_batch_is_capped():
    plan = plan_for(mac())
    assert plan.prediction_batch_size <= Config.prediction_batch_max_coreml


def test_cuda_batch_is_not_capped_by_the_coreml_rule():
    """The cap must be CoreML-only — CUDA genuinely benefits from batching."""
    plan = plan_for(nvidia())
    assert plan.prediction_batch_size > Config.prediction_batch_max_coreml


def test_coreml_cap_can_be_disabled():
    plan = plan_for(mac(), prediction_batch_max_coreml=None)
    assert plan.prediction_batch_size > 2


def test_coreml_cap_never_drops_below_one():
    plan = plan_for(mac(), prediction_batch_max_coreml=0)
    assert plan.prediction_batch_size == 1


# --- the autotune must not sweep past the cap either ---------------------

class Cfg:
    prediction_batch_max_gpu = 12
    prediction_batch_max_coreml = 2
    hardware = None


def test_autotune_candidates_stop_at_the_coreml_cap():
    cfg = Cfg()
    cfg.hardware = mac()
    assert Pred._prediction_batch_candidates(cfg, 1) == [1, 2]


def test_autotune_candidates_unrestricted_on_cuda():
    cfg = Cfg()
    cfg.hardware = nvidia()
    assert max(Pred._prediction_batch_candidates(cfg, 1)) == 12


@pytest.mark.parametrize("initial", [4, 8])
def test_coreml_cap_never_empties_the_candidate_list(initial):
    """A cap below the starting batch must still leave it measurable."""
    cfg = Cfg()
    cfg.hardware = mac()
    assert Pred._prediction_batch_candidates(cfg, initial) == [initial]
