"""producer_workers is now 'reader threads per prediction worker':
R = clamp((hw_cpu - 1) // n_gpu, 1, 16) for multi-GPU; single-GPU caps
at 3 instead of 16 (measured on the T14: extra reader threads contend
with the single consumer for the GIL). CPU-only is always 1.

Not derived from cpu_workers: that is capped at 32 by max_cpu_workers,
which on an 8-GPU box would give 4 readers per GPU -- ~136 tiles/s
against a card that consumes 250."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from classes.Config import Config  # noqa: E402
from classes.ExecutionPlan import (  # noqa: E402
    _reader_threads, _workers_per_gpu, build_execution_plan)


@pytest.mark.parametrize("hw_cpu,n_gpu,cpu_only,expected", [
    (224, 8, False, 16),    # carrot: 223 // 8 = 27 -> cap 16
    (12, 1, False, 3),      # T14: single-GPU cap, not 11
    (224, 1, False, 3),     # single-GPU cap dominates even with cores to spare
    (4, 1, False, 3),
    (2, 1, False, 1),       # floor
    (1, 1, False, 1),       # floor, never 0
    (64, 2, False, 16),     # 63 // 2 = 31 -> cap
    (224, 8, True, 1),      # CPU-only ignores cores
    (12, 1, True, 1),
])
def test_reader_threads_rule(hw_cpu, n_gpu, cpu_only, expected):
    assert _reader_threads(hw_cpu, n_gpu, cpu_only) == expected


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("WINMOL_PREDICTION_READERS", "5")
    assert _reader_threads(224, 8, False) == 5
    assert _reader_threads(224, 8, True) == 5


def test_env_override_ignored_when_not_an_int(monkeypatch):
    monkeypatch.setenv("WINMOL_PREDICTION_READERS", "lots")
    assert _reader_threads(12, 1, False) == 3


# --- G2: producer_queue_batches / reader_chunk by scenario -----------------
#
# Per-worker in-flight tiles = (producer_queue_batches + R + 1) * chunk. A
# single-worker machine (single-GPU or CPU-only) gets no benefit from a
# deep queue -- the pool already decouples the read from inference -- so
# both knobs are capped there to keep RSS at or below main's single
# process. Multi-GPU is unchanged.

class _Hardware:
    def __init__(self, cpu_count, gpu_count, gpu_memory_gb=None,
                 total_ram_gb=64.0):
        self.cpu_count = cpu_count
        self.gpu_count = gpu_count
        self.gpu_memory_gb = gpu_memory_gb or []
        self.gpu_names = []
        self.total_ram_gb = total_ram_gb


_RASTER = {
    'width': 8192, 'height': 8192, 'bands': 3, 'dtype': 'uint8',
    'pixel_size_x': 0.05, 'pixel_size_y': 0.05, 'estimated_input_gb': 2.0,
}


def _plan(cpu_count, gpu_count, gpu_memory_gb=None, **cfg_overrides):
    config = Config()
    for key, value in cfg_overrides.items():
        setattr(config, key, value)
    return build_execution_plan(
        config, _Hardware(cpu_count, gpu_count, gpu_memory_gb),
        _RASTER, 'Trees')


# All three requested at the same depth (8): multi-GPU passes it through
# unchanged, single-GPU and CPU-only clamp it down to a hard depth of 4 --
# a single worker gets no benefit from a deep queue, the reader pool
# already decouples the read from inference, so depth only spends RAM.
def test_multi_gpu_keeps_deep_queue_and_full_chunk():
    plan = _plan(32, 8, [80.0], producer_queue_batches=8)
    assert plan.producer_queue_batches == 8
    assert plan.reader_chunk == 12


def test_single_gpu_caps_queue_depth_and_reader_chunk():
    # 6.0 GB keeps workers_per_gpu at 1 (below the 8.0 GB threshold in
    # _workers_per_gpu) so this isolates the queue/chunk cap from the
    # two-worker behaviour, which gets its own tests below.
    plan = _plan(12, 1, [6.0], producer_queue_batches=8)
    assert plan.producer_queue_batches == 4
    assert plan.reader_chunk == 8


def test_cpu_only_caps_queue_depth_and_reader_chunk():
    plan = _plan(12, 0, producer_queue_batches=8)
    assert plan.producer_queue_batches <= 4
    assert plan.reader_chunk == 4


# --- G3: workers_per_gpu -----------------------------------------------

@pytest.mark.parametrize("scen,gpu_mem_gb,expected", [
    ('gpu', 16.0, 2),        # T14: 4080 SUPER
    ('gpu', 8.0, 2),         # threshold inclusive
    ('gpu', 6.0, 1),         # too little VRAM for two contexts
    ('multi_gpu_dgx', 80.0, 1),  # carrot: unchanged until measured
    ('cpu_only', 0.0, 1),
])
def test_workers_per_gpu_rule(scen, gpu_mem_gb, expected):
    assert _workers_per_gpu(scen, gpu_mem_gb) == expected


def test_workers_per_gpu_env_override_wins(monkeypatch):
    monkeypatch.setenv('WINMOL_WORKERS_PER_GPU', '3')
    assert _workers_per_gpu('gpu', 16.0) == 3
    assert _workers_per_gpu('multi_gpu_dgx', 80.0) == 3
    monkeypatch.setenv('WINMOL_WORKERS_PER_GPU', 'two')
    assert _workers_per_gpu('gpu', 16.0) == 2      # ignored, rule applies


def test_single_gpu_two_workers_shrink_chunk_batch_and_readers():
    plan = _plan(12, 1, [16.0])
    assert plan.workers_per_gpu == 2
    assert plan.reader_chunk == 4
    assert plan.prediction_batch_size <= 2
    # 11 cores / 2 workers = 5 -> per-worker cap 3
    assert plan.producer_workers == 3


def test_single_gpu_small_card_keeps_one_worker():
    plan = _plan(12, 1, [6.0])
    assert plan.workers_per_gpu == 1
    assert plan.reader_chunk == 8


def test_single_gpu_few_cores_split_readers_across_workers():
    # 5 cores: (5-1)//2 = 2 readers per worker, not 3
    plan = _plan(5, 1, [16.0])
    assert plan.workers_per_gpu == 2
    assert plan.producer_workers == 2


def test_multi_gpu_and_cpu_only_keep_one_worker_per_device():
    assert _plan(32, 8, [80.0]).workers_per_gpu == 1
    assert _plan(12, 0).workers_per_gpu == 1
