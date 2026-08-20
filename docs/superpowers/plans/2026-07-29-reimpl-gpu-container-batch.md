# reimpl/gpu-container-batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Program spec: `docs/superpowers/specs/2026-07-28-reimpl-off-main-design.md`.
> Single-task feature; task review doubles as final review.

**Goal:** Feature 5: parallel batch processing — `winmol_batch --jobs N`, one
child per orthomosaic pinned round-robin to GPUs via `CUDA_VISIBLE_DEVICES`
(reference: old #10 commit `7c0628b`), a per-job CPU budget so N jobs don't
oversubscribe the vector phase (rr-era `6bade04` via
`WINMOL_CONFIG_OVERRIDES_JSON`), and the batch-merge boundary-stems fix.

**DEFERRED (recorded):** the CUDA container (`docker/gpu/*`, GHCR workflow) —
cannot be validated before `reimpl/gpu-accelerator` lands `requirements/gpu.txt`
and provider verification. When it comes, port the three hard-won bits:
LD_LIBRARY_PATH fix (6748ab3), ENV-not-CMD mount points (cec4910),
PYTHONHASHSEED pin.

**Batch-merge semantics (explored, decisive):** per-ortho seam dedup + boundary
handling already happen INSIDE winmol_run (`run_merge_phase` passes
`stem_map_path`). The batch-level merge combines N per-ortho GeoPackages with
no rasters — `_detect_tiles` treats each gpkg as a rasterless tile and the
edge-buffer shrink drops true boundary stems per ortho. A single stem_map_path
cannot represent N orthos. Fix: the batch merge must NOT re-filter —
`edge_buffer_m=0.0` default for `--merge` (flag stays for explicit override).

## Task 1: --jobs + CPU budget + merge fix + tests (~220 LOC gross)

**Files:** Modify `winmol_batch.py`; Test `tests/test_batch_jobs.py` (port
`7c0628b:tests/test_batch_jobs.py`, 77 LOC, monkeypatch-only/portable) + CPU-
budget cases (~from 6bade04) + a batch-merge default test.

- `detect_gpu_count()` — `nvidia-smi --query-gpu=index` line count, 0 on any
  failure. `process_orthos(...)` — `queue.Queue` of GPU slots (`i % gpus`,
  `None` when 0 GPUs), `ThreadPoolExecutor(max_workers=jobs)`, child env gets
  `CUDA_VISIBLE_DEVICES=<slot>` only when pinned; `[gpu N]`-prefixed flushed
  output; failures collected `(path, reason)`, never abort; end summary.
  `--jobs/-j` default 1 keeps the sequential path.
- CPU budget: when jobs > 1, merge `{"max_cpu_workers": max(1, cpu_count()//jobs)}`
  into the child's `WINMOL_CONFIG_OVERRIDES_JSON` (respect an existing value in
  the env; adapt key name to what our Config actually has — grep Config.py; if
  no such knob exists on this branch, SKIP the override and record that it
  arrives with the autotune/accelerator feature).
- Batch merge: `--edge-buffer-m` default → 0.0 with a one-line help note
  ("per-ortho seam dedup already applied; leave 0 unless you know why").
- TDD; suite green (64 + new); flake8; commit
  `feat(batch): --jobs parallel orthos, GPU round-robin, no re-filter on batch merge`;
  push branch `reimpl/gpu-container-batch`. NO PR.

## Verification
Suite green; flake8; ledger LOC report. Real multi-GPU behavior untestable
here — pinned by the monkeypatch tests (spread, no-pin-without-GPU,
failure-continues); real-GPU validation in the user's e2e round.
