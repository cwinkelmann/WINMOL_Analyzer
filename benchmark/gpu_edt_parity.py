#!/usr/bin/env python
"""GPU EDT parity + timing check. Run on a Linux CUDA box.

tests/test_gpu_dispatch.py exercises the whole GPU dispatch path against a
numpy/scipy fake; the one thing it cannot verify on a non-CUDA machine is
the device itself. This script closes that gap on real hardware:

1. raw parity: cupyx.scipy.ndimage.distance_transform_edt
   (float64_distances=True, documented to match SciPy) vs
   scipy.ndimage.distance_transform_edt on the golden stem map;
2. pipeline parity: quantify_stems with edt_backend='gpu' vs 'cpu'
   (diameters and frustum volumes, exact comparison);
3. timing: scipy EDT vs GPU EDT incl. H2D/D2H transfers.

Usage (from the repo root, cupy-cuda12x installed — see
requirements/gpu-edt.txt or Dockerfile.blackwell):

    python benchmark/gpu_edt_parity.py \
        [--stem-map tests/fixtures/stem_map.tif] [--repeat 5]

Exit codes: 0 = parity holds, 1 = mismatch, 2 = no usable CuPy/CUDA.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_config(overrides):
    from classes.Config import Config
    config = Config()
    snap = os.path.join(REPO, 'tests', 'fixtures', 'config_snapshot.json')
    if os.path.exists(snap):
        with open(snap) as f:
            for key, value in json.load(f).items():
                setattr(config, key, value)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem-map',
                    default=os.path.join(REPO, 'tests', 'fixtures',
                                         'stem_map.tif'))
    ap.add_argument('--repeat', type=int, default=5)
    args = ap.parse_args()

    from utils import GpuDispatch
    if not GpuDispatch.cupy_available():
        print('FAIL: CuPy/CUDA unavailable in this environment.')
        print('Install cupy-cuda12x (requirements/gpu-edt.txt) and run on '
              'a CUDA machine.')
        return 2

    import cupy as cp
    import scipy.ndimage as ndi
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f'cupy {cp.__version__} | CUDA runtime '
          f'{cp.cuda.runtime.runtimeGetVersion()} | device '
          f'{props["name"].decode()} | cc {dev.compute_capability}')

    import rasterio
    from utils import Quantification as Quant
    with rasterio.open(args.stem_map) as src:
        pred = src.read(1)
        profile = dict(src.profile)
    mask = Quant._as_binary_mask(pred)
    px, py = Quant._pixel_size(profile)
    print(f'stem map {mask.shape} | fg px {int(mask.sum())} | '
          f'pixel {px:.4f}x{py:.4f} m')

    ok = True

    # -- 1. raw EDT parity --------------------------------------------------
    cpu_edt = ndi.distance_transform_edt(mask, sampling=(py, px))
    import cupyx.scipy.ndimage as cndi
    gpu_edt = cp.asnumpy(cndi.distance_transform_edt(
        cp.asarray(mask), sampling=(py, px), float64_distances=True))
    exact = int((cpu_edt == gpu_edt).sum())
    total = cpu_edt.size
    max_diff = float(np.abs(cpu_edt - gpu_edt).max())
    print(f'raw EDT: exact-equal {exact}/{total} px | max abs diff '
          f'{max_diff:.3e} m')
    if not np.allclose(cpu_edt, gpu_edt, rtol=1e-12, atol=1e-9):
        print('FAIL: raw EDT differs beyond tolerance')
        ok = False

    # -- 2. pipeline parity: edt_backend cpu vs gpu -------------------------
    import helpers
    from helpers import read_json_gz
    fixture = os.path.join(REPO, 'tests', 'fixtures',
                           'stage_connect_stems.json.gz')
    records = read_json_gz(fixture)

    results = {}
    for backend in ('cpu', 'gpu'):
        stems = helpers.stems_from_canonical(copy.deepcopy(records))
        config = load_config(dict(diameter_method='edt',
                                  edt_backend=backend))
        results[backend] = Quant.quantify_stems(
            stems, pred, profile, config=config)

    d_cpu = np.array([d for s in results['cpu']
                      for d in s.segment_diameter_list])
    d_gpu = np.array([d for s in results['gpu']
                      for d in s.segment_diameter_list])
    v_cpu = np.array([v for s in results['cpu']
                      for v in s.segment_volume_list])
    v_gpu = np.array([v for s in results['gpu']
                      for v in s.segment_volume_list])
    print(f'pipeline: {len(results["cpu"])} stems | '
          f'{d_cpu.size} diameters | bit-identical '
          f'{int((d_cpu == d_gpu).sum())}/{d_cpu.size} | max abs diff '
          f'{float(np.abs(d_cpu - d_gpu).max()):.3e} m')
    print(f'volumes: total cpu {v_cpu.sum():.6f} m3 vs gpu '
          f'{v_gpu.sum():.6f} m3')
    if not (np.allclose(d_cpu, d_gpu, rtol=1e-12, atol=1e-9)
            and np.allclose(v_cpu, v_gpu, rtol=1e-12, atol=1e-9)):
        print('FAIL: quantified stems differ between edt backends')
        ok = False

    # -- 3. timing ----------------------------------------------------------
    rows, cols = np.nonzero(mask)
    keep = slice(0, max(1, rows.size), max(1, rows.size // 5000))
    rows, cols = rows[keep], cols[keep]

    best_cpu = min(_timed(lambda: ndi.distance_transform_edt(
        mask, sampling=(py, px))) for _ in range(args.repeat))
    best_gpu = min(_timed(lambda: GpuDispatch.edt_gather(
        mask, (py, px), rows, cols)) for _ in range(args.repeat))
    print(f'timing (best of {args.repeat}): scipy {best_cpu * 1e3:.1f} ms '
          f'| gpu edt+gather+D2H {best_gpu * 1e3:.1f} ms | '
          f'{best_cpu / max(best_gpu, 1e-9):.1f}x')

    print('PARITY: ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


if __name__ == '__main__':
    sys.exit(main())
