# PR stack — current state (updated 2026-08-19)

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
| **#42** | `reimpl/oom-resilience` | **stack tip.** OOM survival + overview-read cliff fix (#43), containerised batch (was #44), resize mechanics + v0.5-parity `graph` default (was #45) |

#42 is the tip and is **no longer a draft**. The two siblings that used
to hang off it were folded into it on 2026-08-19 (merge `25c1eea`) to cut
reviewer load — three PRs, one review:

| was | branch | what it contributed to #42 |
|---|---|---|
| #45 | `winmol-oom-fix` | read-strategy flag (`graph` default, v0.5-parity **proven**: R13 −0.06 % / R12 −0.13 % stems, volume to 0.002 %), parity gate, `graph_aa`, `docs/resize-mechanics.md` |
| #44 | `reimpl/docker-batch` | containerised batch (desktop → multi-GPU), cgroup-aware `HardwareInfo`, `GDAL_CACHEMAX` sizing |

The amendment #44 needed before merge is **done** (2026-08-19): the
retracted era numbers in `docs/resampling-accuracy.md` and
`docs/performance-v05-to-now.md` (12,714/12,722 "v0.5", "30 % effect",
"455 on every variant") now carry retraction banners pointing at
`docs/resize-mechanics.md`, which holds the genuine-v0.5.0 comparison.

One conflict had to be resolved by hand, in `TileBatchProducer.run`
(`utils/Prediction.py`): #44 keyed the interior-window/boundless rule off
the `_BENCH_READ` global that #45 deleted, and #45 kept the rule only on
the `overview` branch. Since `graph` is now the default, the rule is
hoisted above the branch and applies to **every** strategy — which is
where #44's measured 3.2× read win (47.3 → 15.0 ms, pixel-identical)
actually lands.

## How Stefan merges it

All 14 are **out of draft** and form a GitHub *stacked* PR chain (public
preview): #26 targets `main`, every later PR targets the branch below it,
and no branch is behind its base.

**Merging the top PR (#42) merges the entire stack** — every PR below
comes with it, bottom-up, producing the same history as merging them one
at a time. Merging a mid-stack PR instead takes everything below it and
leaves the rest open, auto-retargeted onto `main`. Merge commit, squash
and rebase all work, and the stack is merge-queue aware.

Do not merge them individually top-down, and do not retarget any base by
hand — that breaks the chain GitHub uses to order the merge.

Order, bottom to top: **#26 → #27 → #28 → #29 → #30 → #31 → #32 → #33 →
#34 → #35 → #36 → #38 → #39 → #42**.

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
