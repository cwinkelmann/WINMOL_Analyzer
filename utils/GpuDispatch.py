"""GPU dispatch for the EDT diameter path (diameter_method='edt' only).

CuPy is optional: this module lazily try-imports it and everything falls
back to the existing scipy CPU path automatically when it is missing, when
no CUDA device is present, or when running in a process where CUDA is
unsafe.

Fork-safety (the reason for the process guard): on Linux the vector tile
pool is created with fork() AFTER the parent process initialized CUDA
(TensorFlow / onnxruntime do so during prediction). CUDA cannot be
re-initialized in a forked child, so GPU EDT is only allowed in:

  * the main process (serial vector path, tile_workers == 1), or
  * spawn-context pool workers explicitly marked via ``mark_spawn_worker``
    (``VectorTilePipeline`` switches its tile pool to the spawn context
    when GPU EDT is active and installs that marker as the pool
    initializer).

Every other child process resolves to the scipy CPU fallback.

GPU contention: concurrent spawn workers share one cross-process lock
(installed by ``mark_spawn_worker``) so at most one EDT workspace lives on
the GPU at a time, and the CuPy memory pool is released after every call
because TensorFlow's memory-growth pool owns the rest of the card.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from types import SimpleNamespace

import numpy as np

# Set by mark_spawn_worker() inside spawn-context pool workers. A fork()ed
# child inherits the parent's value (False in workers, since the parent
# never marks itself), so it correctly reads as "not a spawn worker".
_SPAWN_WORKER = False

# Cross-process lock serializing GPU sections between spawn workers.
# None in the main process, where the thread lock below suffices.
_GPU_LOCK = None
_LOCAL_LOCK = threading.Lock()

# Cached CuPy bundle: SimpleNamespace(cp=..., ndi=...) or None.
_CUPY = None
_CUPY_CHECKED = False


def reset_cupy_cache():
    """Forget the cached CuPy probe (test hook)."""
    global _CUPY, _CUPY_CHECKED
    _CUPY = None
    _CUPY_CHECKED = False


def _import_cupy():
    """Return SimpleNamespace(cp, ndi) or None. Cached after first probe."""
    global _CUPY, _CUPY_CHECKED
    if _CUPY_CHECKED:
        return _CUPY
    _CUPY_CHECKED = True
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            _CUPY = None
        else:
            _CUPY = SimpleNamespace(cp=cp, ndi=cndi)
    except Exception:
        _CUPY = None
    return _CUPY


def cupy_available() -> bool:
    return _import_cupy() is not None


def mark_spawn_worker(lock=None):
    """Pool initializer for spawn-context tile workers.

    Marks the process as CUDA-safe (fresh process, no inherited CUDA
    context) and installs the shared cross-process GPU lock.
    """
    global _SPAWN_WORKER, _GPU_LOCK
    _SPAWN_WORKER = True
    if lock is not None:
        _GPU_LOCK = lock


def in_gpu_capable_process() -> bool:
    """True when CUDA may be initialized in this process.

    Main process: always allowed (single in-process CUDA context shared
    with TF/ORT is fine). Child processes: only when created by the
    spawn-context pool that marked them via ``mark_spawn_worker``.
    """
    if mp.current_process().name == "MainProcess":
        return True
    return bool(_SPAWN_WORKER)


def resolve_edt_backend(config):
    """Return ('gpu'|'cpu', reason). Never raises.

    Only consulted when diameter_method == 'edt'; the results-changing
    contour/edt switch itself is NOT touched by this module.
    """
    pref = str(getattr(config, 'edt_backend', 'auto') or 'auto').lower()
    if pref not in ('auto', 'cpu', 'gpu'):
        return 'cpu', f"unknown edt_backend={pref!r}; using scipy CPU"
    if pref == 'cpu':
        return 'cpu', "edt_backend='cpu'"
    if not cupy_available():
        if pref == 'gpu':
            return 'cpu', ("edt_backend='gpu' requested but CuPy/CUDA "
                           "unavailable; falling back to scipy CPU")
        return 'cpu', 'CuPy/CUDA unavailable'
    if not in_gpu_capable_process():
        return 'cpu', ('forked worker: CUDA unsafe after the parent '
                       'initialized it; using scipy CPU')
    return 'gpu', f"edt_backend={pref!r}, CuPy/CUDA available"


def gpu_edt_wanted(config) -> bool:
    """Parent-side pool decision: would tile workers want GPU EDT?

    Evaluated in the parent BEFORE the vector pool is created, so it must
    not depend on the process guard (the parent is the main process).
    """
    method = str(getattr(config, 'diameter_method', 'contour') or
                 'contour').lower()
    if method != 'edt':
        return False
    pref = str(getattr(config, 'edt_backend', 'auto') or 'auto').lower()
    if pref == 'cpu':
        return False
    return cupy_available()


class _gpu_section:
    """Serialize GPU work across processes (spawn workers) or threads."""

    def __enter__(self):
        self._lock = _GPU_LOCK if _GPU_LOCK is not None else _LOCAL_LOCK
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


def _free_gpu_memory(bundle):
    try:
        bundle.cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def edt_gather(mask, sampling, rows, cols):
    """One GPU EDT + one batched gather + one D2H copy.

    mask: 2D bool array; sampling: (py, px) meters per pixel;
    rows/cols: in-bounds int arrays of node pixel indices.
    Returns float64 radii (meters), one per node.

    float64_distances=True is mandatory: CuPy documents it as matching
    SciPy, which keeps the GPU backend comparable to the CPU edt branch.
    """
    bundle = _import_cupy()
    if bundle is None:
        raise RuntimeError('CuPy/CUDA unavailable')
    cp = bundle.cp
    with _gpu_section():
        try:
            edt_gpu = bundle.ndi.distance_transform_edt(
                cp.asarray(mask),
                sampling=sampling,
                float64_distances=True,
            )
            radii = cp.asnumpy(edt_gpu[cp.asarray(rows), cp.asarray(cols)])
        finally:
            # TF's memory-growth pool owns the rest of the card; never
            # retain EDT workspace between tiles.
            _free_gpu_memory(bundle)
    return np.asarray(radii, dtype=np.float64)
