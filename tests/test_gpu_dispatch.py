"""GPU EDT dispatch: backend selection, fork-safety guard, batched path.

CUDA cannot run on this machine, so the CuPy bundle is replaced by a
numpy/scipy fake with the same call surface. That exercises the entire
GPU code path (dispatch -> batched gather -> per-stem assembly) minus the
device itself; on-device parity is verified by benchmark/gpu_edt_parity.py
on a CUDA box. The scipy CPU fallback path is pinned against the golden
stage_quantified_edt fixture, which this suite must keep matching.
"""

import copy
import multiprocessing as mp
from multiprocessing.context import SpawnContext
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.ndimage as scipy_ndi
from shapely.geometry import LineString

import helpers
from utils import GpuDispatch
from utils import Quantification as Quant
from utils import VectorTilePipeline as VTP


# ---------------------------------------------------------------------------
# Fake CuPy bundle (numpy/scipy behind the CuPy call surface)
# ---------------------------------------------------------------------------

class _FakeMemoryPool:
    freed = 0

    def free_all_blocks(self):
        _FakeMemoryPool.freed += 1


class _FakeCp:
    @staticmethod
    def asarray(x):
        return np.asarray(x)

    @staticmethod
    def asnumpy(x):
        return np.asarray(x)

    @staticmethod
    def get_default_memory_pool():
        return _FakeMemoryPool()


class _FakeNdi:
    @staticmethod
    def distance_transform_edt(mask, sampling=None, float64_distances=None):
        # The real GPU call MUST pass float64_distances=True (parity with
        # SciPy per CuPy docs); the fake enforces that contract.
        assert float64_distances is True
        return scipy_ndi.distance_transform_edt(
            np.asarray(mask), sampling=sampling)


def _fake_bundle():
    return SimpleNamespace(cp=_FakeCp, ndi=_FakeNdi)


@pytest.fixture()
def fake_cupy(monkeypatch):
    monkeypatch.setattr(GpuDispatch, '_CUPY', _fake_bundle())
    monkeypatch.setattr(GpuDispatch, '_CUPY_CHECKED', True)


@pytest.fixture()
def no_cupy(monkeypatch):
    monkeypatch.setattr(GpuDispatch, '_CUPY', None)
    monkeypatch.setattr(GpuDispatch, '_CUPY_CHECKED', True)


def _cfg(**kw):
    base = dict(diameter_method='edt', edt_backend='auto')
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def test_resolve_cpu_when_cupy_missing(no_cupy):
    backend, reason = GpuDispatch.resolve_edt_backend(_cfg())
    assert backend == 'cpu'
    assert 'unavailable' in reason


def test_resolve_pref_cpu_wins_over_available_gpu(fake_cupy):
    backend, _ = GpuDispatch.resolve_edt_backend(_cfg(edt_backend='cpu'))
    assert backend == 'cpu'


def test_resolve_gpu_pref_falls_back_without_cupy(no_cupy):
    backend, reason = GpuDispatch.resolve_edt_backend(
        _cfg(edt_backend='gpu'))
    assert backend == 'cpu'
    assert 'falling back' in reason


def test_resolve_unknown_backend_value_is_cpu(fake_cupy):
    backend, _ = GpuDispatch.resolve_edt_backend(_cfg(edt_backend='tpu'))
    assert backend == 'cpu'


def test_resolve_gpu_in_main_process(fake_cupy):
    assert mp.current_process().name == 'MainProcess'
    backend, _ = GpuDispatch.resolve_edt_backend(_cfg())
    assert backend == 'gpu'


def test_resolve_cpu_in_forked_child(fake_cupy, monkeypatch):
    monkeypatch.setattr(
        GpuDispatch.mp, 'current_process',
        lambda: SimpleNamespace(name='ForkPoolWorker-1'))
    monkeypatch.setattr(GpuDispatch, '_SPAWN_WORKER', False)
    backend, reason = GpuDispatch.resolve_edt_backend(_cfg())
    assert backend == 'cpu'
    assert 'forked worker' in reason


def test_resolve_gpu_in_marked_spawn_worker(fake_cupy, monkeypatch):
    monkeypatch.setattr(
        GpuDispatch.mp, 'current_process',
        lambda: SimpleNamespace(name='SpawnPoolWorker-1'))
    monkeypatch.setattr(GpuDispatch, '_SPAWN_WORKER', False)
    monkeypatch.setattr(GpuDispatch, '_GPU_LOCK', None)
    lock = object()
    GpuDispatch.mark_spawn_worker(lock)
    backend, _ = GpuDispatch.resolve_edt_backend(_cfg())
    assert backend == 'gpu'
    assert GpuDispatch._GPU_LOCK is lock


def test_gpu_edt_wanted_matrix(fake_cupy):
    assert GpuDispatch.gpu_edt_wanted(_cfg()) is True
    assert GpuDispatch.gpu_edt_wanted(
        _cfg(diameter_method='contour')) is False
    assert GpuDispatch.gpu_edt_wanted(_cfg(edt_backend='cpu')) is False


def test_gpu_edt_wanted_false_without_cupy(no_cupy):
    assert GpuDispatch.gpu_edt_wanted(_cfg()) is False


# ---------------------------------------------------------------------------
# Tile-pool context decision (VectorTilePipeline)
# ---------------------------------------------------------------------------

def test_tile_pool_plan_spawn_when_gpu_edt(monkeypatch):
    monkeypatch.setattr(GpuDispatch, 'gpu_edt_wanted', lambda config: True)
    ctx, use_gpu = VTP._tile_pool_plan(_cfg())
    assert use_gpu is True
    assert isinstance(ctx, SpawnContext)


def test_tile_pool_plan_default_fork_pool(monkeypatch):
    monkeypatch.setattr(GpuDispatch, 'gpu_edt_wanted', lambda config: False)
    ctx, use_gpu = VTP._tile_pool_plan(_cfg(diameter_method='contour'))
    assert use_gpu is False
    assert ctx is mp


def _spawn_probe(_):
    return (GpuDispatch._SPAWN_WORKER, GpuDispatch._GPU_LOCK is not None)


def test_spawn_pool_initializer_marks_workers():
    """The exact pool wiring VectorTilePipeline uses when GPU EDT is
    active: spawn context, mark_spawn_worker initializer, shared Lock."""
    ctx = mp.get_context('spawn')
    with ctx.Pool(
        2,
        initializer=GpuDispatch.mark_spawn_worker,
        initargs=(ctx.Lock(),),
    ) as pool:
        out = pool.map(_spawn_probe, range(4))
    assert all(flag and has_lock for flag, has_lock in out)
    # the parent process itself must stay unmarked
    assert GpuDispatch._SPAWN_WORKER is False


# ---------------------------------------------------------------------------
# Batched gather: node -> (row, col) parity with the scalar path
# ---------------------------------------------------------------------------

def test_batched_rowcol_matches_scalar(stem_map):
    _, profile = stem_map
    t = profile['transform']
    rng = np.random.default_rng(42)
    stems = []
    for _ in range(20):
        n = int(rng.integers(2, 12))
        xs = rng.uniform(t.c - 30.0, t.c + 120.0, n)
        ys = rng.uniform(t.f - 120.0, t.f + 30.0, n)
        stems.append(SimpleNamespace(
            path=LineString(list(zip(xs, ys)))))
    rows, cols, counts = Quant._gather_node_rowcols(stems, profile)
    assert sum(counts) == rows.shape[0] == cols.shape[0]
    pos = 0
    for stem, count in zip(stems, counts):
        for i, (x, y) in enumerate(stem.path.coords):
            row, col = Quant._xy_to_rowcol(x, y, profile)
            assert row == rows[pos + i]
            assert col == cols[pos + i]
        pos += count


def test_edt_gather_values_and_dtype(fake_cupy):
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    sampling = (0.5, 0.5)
    expected = scipy_ndi.distance_transform_edt(mask, sampling=sampling)
    rows = np.array([0, 3, 4])
    cols = np.array([0, 3, 4])
    radii = GpuDispatch.edt_gather(mask, sampling, rows, cols)
    assert radii.dtype == np.float64
    np.testing.assert_array_equal(radii, expected[rows, cols])


def test_edt_gather_raises_without_cupy(no_cupy):
    with pytest.raises(RuntimeError):
        GpuDispatch.edt_gather(
            np.ones((2, 2), dtype=bool), (1.0, 1.0),
            np.array([0]), np.array([0]))


# ---------------------------------------------------------------------------
# End-to-end: GPU path (fake device) against the golden EDT fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def connect_stage_stems(golden):
    return helpers.stems_from_canonical(golden("stage_connect_stems"))


def _edt_config(pipeline_config):
    config = copy.copy(pipeline_config)
    config.diameter_method = "edt"
    config.edt_backend = "auto"
    return config


def test_gpu_batched_matches_golden_edt(connect_stage_stems, stem_map,
                                        pipeline_config, golden,
                                        fake_cupy, monkeypatch):
    pred, profile = stem_map
    calls = {'n': 0}
    real_gather = GpuDispatch.edt_gather

    def spy(*args, **kwargs):
        calls['n'] += 1
        return real_gather(*args, **kwargs)

    monkeypatch.setattr(GpuDispatch, 'edt_gather', spy)
    stems = Quant.quantify_stems(
        connect_stage_stems, pred, profile, _edt_config(pipeline_config))
    # exactly ONE batched gather for the whole tile, no per-node calls
    assert calls['n'] == 1
    helpers.assert_stems_match_golden(stems, golden("stage_quantified_edt"))


def test_gpu_failure_falls_back_to_cpu(connect_stage_stems, stem_map,
                                       pipeline_config, golden,
                                       fake_cupy, monkeypatch, capsys):
    pred, profile = stem_map

    def boom(*args, **kwargs):
        raise RuntimeError('synthetic CUDA OOM')

    monkeypatch.setattr(GpuDispatch, 'edt_gather', boom)
    stems = Quant.quantify_stems(
        connect_stage_stems, pred, profile, _edt_config(pipeline_config))
    helpers.assert_stems_match_golden(stems, golden("stage_quantified_edt"))
    assert 'falling back to scipy CPU EDT' in capsys.readouterr().out
