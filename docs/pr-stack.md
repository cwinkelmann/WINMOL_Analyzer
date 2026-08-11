# PR stack — current state (updated 2026-08-11)

Supersedes the 2026-07-21 restack notes in `push_restack.sh` and the
July merge-order audit. Two generations exist on origin; only the
reimpl generation is live.

## Live stack: the reimpl chain (merge top-down, in order)

Linear, each PR based on the previous:

| # | branch | one-liner |
|---|---|---|
| #26 | `reimpl/onnx-runtime` | TF-free ONNX segmenter + model dispatch |
| #27 | `reimpl/plugin-onnx` | plugin shell-out + sanitized child env |
| #28 | `reimpl/deterministic-stems` | deterministic stems |
| #29 | `reimpl/perf-vectorization` | cubic resample, edge-fix, vector-index hoist |
| #30 | `reimpl/gpu-container-batch` | batch mode, GPU round-robin |
| #31 | `reimpl/model-registry` | device-aware model registry + download |
| #32 | `reimpl/gpu-accelerator` | provider truth, ort-gpu venv, autotune |
| #33 | `reimpl/packaging` | plugin ZIP builder |
| #34 | `reimpl/plugin-gui` | Setup tab + dialog overhaul |
| #35 | `reimpl/rc11-parity` | py311 download, autotune cache, GPU verdict |
| #36 | `reimpl/gui-parity` | detection-tab parity, CoreML bind |
| #38 | `reimpl/ci` | cross-OS CI + ZIP build |
| #39 | `reimpl/cleanup` | drop template scaffolding/cruft |
| #42 | `reimpl/oom-resilience` | OOM survival + **overview-read cliff fix (02a8488)** |

After #42 the stack forks into two siblings (either order; rebase the
other after the first merges):

| # | branch | one-liner | caveat |
|---|---|---|---|
| **#45** | `winmol-oom-fix` | read-strategy flag (`graph` default, v0.5-parity **proven**: R13 −0.06 % / R12 −0.13 % stems, volume to 0.002 %), parity gate, graph_aa, resize-mechanics docs | none — carries the parity-proof table |
| #44 | `reimpl/docker-batch` | containerised batch, desktop → multi-GPU | **must be amended before merge**: its `docs/resampling-accuracy.md` / `performance-v05-to-now.md` carry the retracted era numbers (12,714/12,722 "v0.5", "30 % effect", "455 on every variant") — refuted 2026-08-11, root-caused to a Spruce_Deadwood model mix-up; see `docs/resize-mechanics.md` on #45 |

Suggested final order: **#26 → … → #39 → #42 → #45 → #44 (amended)**.

## Superseded: the July legacy stack (#1–#23)

The pre-reimpl generation (`test/standalone-metal-e2e`,
`feature/qgis-plugin-onnx`, `perf/*`, `feat/gpu-container`,
`rr6-split/*`). The reimpl program rebuilt all of it off `main` as the
chain above; the rr6 splits (#17–#23) are already CLOSED, and #6, #7,
#14, #16 are MERGED into their era branches. The remaining OPEN legacy
PRs (#1, #2, #3, #4, #5, #9, #10, #12, #15) are superseded — close
them when the reimpl chain lands, or before, to reduce noise. #13
(`perf/p3-gpu-edt-quant`, GPU EDT via CuPy) is the one legacy PR with
content not yet re-implemented in the reimpl chain; decide separately
whether to port it.

## Standing rules

- Nothing merges to `main` except through the chain, top-down.
- Any resize/resampling change on any branch must pass
  `benchmark/bench_resize_parity.py --probe rc12` at a ≥2.3× factor
  (see `docs/resize-mechanics.md`).
- Era-context numbers (the retracted A/B) must not be cited as
  baselines anywhere in the stack.
