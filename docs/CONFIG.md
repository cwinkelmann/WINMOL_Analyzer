# Configuration reference

Two different things are configured here:

- the **model registry** (`config.json`, repo root — shipped inside the QGIS
  plugin): which models exist, where they download from, and how they verify;
- the **pipeline `Config`** (`classes/Config.py`): tiling, workers, thresholds.

## Model registry (`config.json`, schema v2)

`config.json` is no longer a flat `{name: url}` map. It is a versioned
registry (`"schema": 2`) parsed by `plugin_utils/model_registry.py` and
consumed by the QGIS dialog, `plugin_utils/installer.py`, `winmol_batch.py`
and `scripts/convert_models_to_onnx.py`. The loader still accepts a legacy
flat map, and `Registry.flat_map()` re-serves `{id: url}` for anything that
wants the old shape — but external scripts that `json.load` the file and
iterate `.items()` must migrate.

What an entry carries: stable `id` (the four classic ids **General, Beech,
Spruce, Spruce_Deadwood are unchanged**), a human-readable
`label`/`description`, the download `url`, the mandatory on-disk `file` name
(always the URL basename; one naming rule for the plugin's `models/` dir
*and* the batch CLI's `--model-dir`, ending the old key-vs-basename split),
a checksum (`sha256` for the model-zoo assets, `md5` for the Zenodo HDF5
originals), and metadata (`family`, `backend`, `precision`, `f1`, `size_mb`,
`lossless`, `hidden`).

**Every downloadable entry is digest-pinned.** There is no unverifiable
download left in the registry, and
`tests/test_model_registry.py::test_every_entry_has_a_digest` fails the
build if an unpinned entry is added. Getting there required repointing the
four classic ids (2026-07-21) from the older `models-onnx-v1` release of
this repo — for which no checksums were ever published — onto the
`models-v1` assets `model_UNet_{GenDS,SpecDS_Beech,SpecDS_Spruce,
SpecDS_Spruce_Deadwood}_512_<upstream timestamp>.onnx`, which the zoo
manifest states are conversions of the *same* upstream Keras HDF5 and
numerically identical.
**Consequence:** the on-disk names changed (`General.onnx` →
`model_UNet_GenDS_512_2023-02-27_211141.onnx`, etc.), so a previously
downloaded classic model is no longer recognized and is re-fetched once
(124.6 MB each). The old release is untouched and still downloadable, but
it is no longer referenced by the registry.

**Asset names are checked, not trusted.** The `models-v1` release renamed
every asset once already (2026-07-21) to the scheme
`model_<Arch>_<Flavour>_512[_<timestamp>][_<qualifier>][_fp16|_int8].onnx`,
which 404'd all 22 URLs in a shipped release before anything caught it.
Two guards now exist: `tests/test_model_urls.py` checks offline that each
entry's `file` equals its URL basename and that every sha256 appears in the
vendored `tests/fixtures/models_v1_SHA256SUMS`, and — with
`WINMOL_NET_TESTS=1` — that every registry URL is anonymously reachable.
Run the latter before cutting a release; it stays skipped in CI.

**Defaults are ranked and device-aware.** `recommended` is an explicit
ranked list, best first, and `recommended[0]` must equal `gui_default` (the
loader rejects a registry where they disagree). Shipped ranking:

1. `Spruce_Deadwood_int8` — INT8 Spruce + standing deadwood (SpecDS),
   31.4 MB, the default.
2. `UNet_PT_int8` — `model_UNet_SpecDS_Beech_512_pytorch_w05_int8.onnx`,
   7.9 MB, TestDS F1 0.760.

`Registry.default_entry(device)` computes the **effective** default: it
keeps `recommended[0]`'s family (the domain choice) and takes that family's
`cpu` (int8) variant on a CPU host or its `gpu` (fp16) variant on a GPU
host. It never downloads and never touches the network beyond the local
`detect_device()` probe. It deliberately bypasses `resolve`'s lossless-only
gate, because here the registry has *itself* nominated an optimised entry —
honouring the device is the declared intent, not a silent substitution. Any
explicit selection (a GUI entry, `winmol_batch <MODEL>`, `--variant`)
overrides it. The GUI lists the recommended families first and preselects
the matching variant, so what is displayed is what runs.

> **Caveat, stated honestly:** the classic `_int8` builds (including the
> default) are described in the zoo manifest as "post-training static int8,
> domain-calibrated" — they are **not** certified lossless there, and carry
> `"lossless": false` here. Only the `_fp16` variants, and the PyTorch UNet
> w05 int8, are certified lossless. The int8 Spruce+Deadwood default is a
> product decision (31.4 MB, fast on CPU), not a measured-equivalence
> claim. On a GPU host the effective default is the lossless fp16 variant.

> **Naming, to stop a recurring mix-up (corrected 2026-07-21):** `w05`
> (width-0.5) belongs only to the **PyTorch UNet** family, asset
> `model_UNet_SpecDS_Beech_512_pytorch_w05_int8.onnx` (entry
> `UNet_PT_int8`). The `models-v1` rename settled its provenance: it *is*
> trained on SpecDS beech data, so the earlier claim here that `SpecDS`
> names only the classic Keras models was wrong. What still must not be
> conflated is the lineage — the classic converted-Keras entries
> (`General`, `Beech`, `Spruce`, `Spruce_Deadwood`) have no `w05` variant;
> only the `_pytorch` family does.

**Families and variants.** A family groups the precision variants of one
trained model: `default` (fp32 reference), `cpu` (int8), `gpu` (fp16).
Resolution rules (`Registry.resolve`):

- An **explicit model id is never rewritten** — `General` always means the
  fp32 GenDS model, on every device, under every `--variant`.
- A **family id** (`unet_pt`, `classic_spruce`, `hrnet`, …) picks the family
  default; with variant `auto` the device variant is substituted **only when
  it is certified `lossless`** (the classic int8s are domain-calibrated, not
  certified — auto never picks them; the PyTorch UNet int8/fp16 are lossless
  and are picked). `--variant fp32|int8|fp16` forces a variant or errors if
  the family lacks it.
- Device for `auto` = `WINMOL_DEVICE` env (`gpu`/`cpu`) if set, else an
  `nvidia-smi` probe. Apple-Silicon/CoreML machines report `cpu`; use
  `WINMOL_DEVICE`/`WINMOL_ONNX_PROVIDERS` to steer if needed.

**Sources.** All 22 ONNX entries (the classic four included, since the
repoint above) come from the `models-v1` release of
`cwinkelmann/WINMOL_segmentor_pt`, sha256-pinned from its
SHA256SUMS (all share one contract — NCHW `[b,3,512,512]` float32 in [0,1] →
`[b,1,512,512]`, sigmoid baked in, opset 17 — and load unchanged through
`OnnxSegmenter`). Originals: Zenodo record 15907576
(DOI 10.5281/zenodo.15907576), the four Keras `.hdf5`, md5-pinned, marked
`hidden` (the plugin venv is ONNX-only; the CLI and the converter can still
address them by id, e.g. `Spruce_Deadwood_hdf5`).

**Downloads are on-demand and verified.** `preload` ships empty, so plugin
startup does zero model network I/O; the dialog offers the download when you
Run with a model that is not on disk, and `winmol_batch.py` fetches before
processing (`--no-download` forbids it). All fetches stream to a `.part`
file, verify the checksum, then atomically rename — a crash can only leave a
`*.part`, never a truncated model that passes the old size>0 check. A cached
file that *fails* its checksum is treated as stale and re-downloaded, so URL
swaps in future registry updates actually take effect. Verified digests are
memoized in `<model_dir>/.winmol_verified.json` (hash 374 MB once, then a
`stat()` suffices).

CLI: `winmol_batch.py --list-models` prints ids, backends, sizes, F1 and
installed state; model names and family ids are case-insensitive
(`general` still works). GUI: the dropdown shows one entry per family plus
`Custom` (reserved id); the Variant selector picks fp32/int8/fp16.

## The plugin's Setup tab

Environment creation, environment deletion and model downloads live on the
**Setup** tab (`Detection | Setup | Log`), never on the detection tab and
never on the GUI thread. Every decision behind it — which button is
enabled, what the environment line says, why Run is blocked, what a
deletion would remove — is a pure function in `plugin_utils/setup_state.py`
plus `plugin_utils/model_status.py`, both Qt-free and unit-tested
(`tests/test_setup_state.py`, `tests/test_model_status.py`,
`tests/test_setup_tab_ui.py`). The long operations run on the two existing
worker/QThread pairs (`_env_*` for build/repair/delete, `_dl_*` for
download/verify/delete-model).

Two conventions a future change must keep:

* **Tabs are selected by widget, never by index.** `setCurrentIndex(1)`
  used to mean "the Log tab"; with Setup inserted between Detection and
  Log it would silently mean Setup. Use `_show_tab(page)`; an AST test
  bans integer literals on `log_widget.setCurrentIndex`.
* **i18n:** static user-visible strings belong in
  `winmol_analyzer_dialog_base.ui`, where `pyuic` makes them
  `lupdate`-extractable for free. Dynamically composed strings live as
  module-level `TXT_*` format templates in `plugin_utils/setup_state.py`,
  which must stay Qt-free and therefore must not call `tr()` — the dialog
  is the only place allowed to do `self.tr(setup_state.TXT_…).format(…)`.
  Paths, byte counts and file names are always named `{placeholders}`
  outside the translatable span, never concatenated fragments, so a
  translator can reorder them.

## Pipeline `Config`

Defaults live in `classes/Config.py`. Override any of them without editing code:

```bash
export WINMOL_CONFIG_OVERRIDES_JSON='{"max_cpu_workers":128}'
# in a container:
docker run -e WINMOL_CONFIG_OVERRIDES_JSON='{"max_cpu_workers":128}' ...
```

Unknown keys are reported and skipped (`winmol_run.py:91`).

## Read the run log carefully — it prints two different things

```
Execution plan:              <- what the planner DECIDED
  gpu_workers      = 2
  cpu_workers      = 32
  vector_tile_workers = 4
  vector_inner_workers = 1

Configurations:              <- the Config object AFTER the plan overwrote it
  cpu_workers    1           <- NOT the same number as above!
  gpu_workers    2
```

`_apply_plan_to_config` (`winmol_run.py:154`) writes plan results back into the
config, and in `vector_mode: tiled` it sets `config.cpu_workers =
plan.vector_inner_workers`. So the `cpu_workers 1` in the second block means
*one worker inside each vector tile*, not "one CPU". Compare against
`vector_tile_workers` to see the real parallelism: **4 tiles × 1 inner = 4
processes**.

## What is actually settable

**Critical:** `cpu_workers` and `gpu_workers` are **outputs, not inputs**. The
planner computes them and overwrites whatever you set — it never reads them
(`classes/ExecutionPlan.py`). Setting them has no effect. Tune the ceilings
below instead.

| Key | Default | Meaning |
|---|---|---|
| `max_cpu_workers` | 32 | Hard ceiling on CPU workers, applied as `min(this, cores-1)`. **The main lever on a many-core box.** |
| `max_gpu_workers` | 8 | Ceiling on GPU worker processes (one per GPU). |
| `single_gpu_cpu_workers` | 24 | CPU workers requested when 1 GPU is present; clamped by `max_cpu_workers`. |
| `multi_gpu_cpu_workers` | 48 | Same for >1 GPU. With the default ceiling of 32 this **never takes effect**. |
| `max_vector_tile_workers` | 4 | Ceiling on tiles vectorised in parallel. The vector phase is ~73 % of a run, so this caps the slowest stage. |
| `prediction_batch_gpu` / `_multi_gpu` / `_max_gpu` | 4 / 12 / 16 | Tiles per inference batch. |
| `prediction_producer_workers_*` | 1 / 6 / 6 | Threads reading + preparing tiles to feed the GPU. |
| `tile_inner_px` | 4096 | Vector tile size. **Changes results** (measured) — see the sweep. Bigger = fewer seams, more RAM per worker; smaller = proportionally more halo recomputation. |
| `tile_overlap_m` | 12.0 | Halo between vector tiles; also the merge de-duplication buffer. |
| `gpu_memory_fraction` | 0.9 | Fraction of GPU memory a worker may use. |
| `prediction_batch_autotune` | False | Off because it re-runs every prediction, costing minutes on CoreML/Metal for ~1 % throughput. |
| `stem_binary_threshold` | 0.5 | Mask binarisation cut-off. **Changing this changes results** — the golden fixtures assume 0.5. |
| `min_length` | 2.0 | Shortest stem kept (m). Also a results-changing knob. |
| `measuring_point_spacing_m` | 0.5 | Diameter sampling interval along a stem. |
| `diameter_method` | contour | `contour` or `edt`. |

## The GPU count is NOT configurable

`gpu_workers` is capped by the **number of prediction tiles**, hardcoded in
`ExecutionPlan.py`:

```python
gpu_workers = min(max_gpu_workers, gpus_available)
if   tiles < 1000: gpu_workers = min(gpu_workers, 2)
elif tiles < 2500: gpu_workers = min(gpu_workers, 4)
else:              gpu_workers = min(gpu_workers, 8)
```

On an 8-GPU node a job with fewer than 1000 tiles uses **two GPUs**, and no
config setting changes that. The reasoning is sound — each worker is a process
that loads the model and initialises CUDA, and that startup dominates a small
job — but the thresholds were not chosen with H100-class hardware in mind, where
startup is cheap and the job is often "small" by tile count while still worth
spreading. Raising them is a code change.

## Measured: a config sweep, and why nothing helped

Six configurations, 3 runs each, Spruce_Deadwood on Barnekow, RTX 4080 SUPER
(16 cores), via `benchmark/`:

| config | median | vs base | stems | output |
|---|---|---|---|---|
| **baseline** | **33 s** | — | 458 | — |
| `max_vector_tile_workers:1` | 43 s | +30 % | 458 | same |
| `max_cpu_workers:128` | 34 s | +3 % | 458 | same |
| + `max_vector_tile_workers:16` | 34 s | +3 % | 458 | same |
| `tile_inner_px:2048` | 46 s | +39 % | **459** | **changed** |
| `tile_inner_px:8192` | 40 s | +21 % | **456** | **changed** |

**No output-preserving config beat the defaults.** They are well tuned for a
normal workstation; do not tune this blindly.

### Why the overrides did nothing — read the resolved plan

Every run resolved to `cpu_workers = 11` and `vector_tile_workers = 2`,
*including* the one setting `max_cpu_workers: 128`:

```
max_cpu_workers = min(128, hw_cpu-1 = 15) = 15  →  ×0.75  →  cpu_workers = 11
vector_tile_workers = min(max_vector_tile_workers, cpu_workers//4 = 2, tiles) = 2
```

`max_vector_tile_workers: 16` was inert because **`cpu_workers//4` binds first**,
and that is pinned by the core count. The lesson generalises: after setting an
override, check the `Execution plan:` block in the log. An override that is
clamped away looks exactly like one that had no effect.

### What this does NOT tell you about a large node

On 224 cores the arithmetic lands in a different regime —
`cpu_workers ≈ 32` → `tile_workers = min(max_vector_tile_workers=4, 8) = 4` —
so there `max_vector_tile_workers` **is** the binding cap and raising it may
help. That is the one case worth testing, and it cannot be tested on a small
machine. Measure it on the target hardware before adopting anything.

### Two results that do generalise

- **Inner parallelism loses.** One tile with 11 inner workers was 30 % slower
  than two tiles with one each. Consistent with everything else measured in
  this project: a process pool in quantification was 10× slower than serial,
  and prediction batch 8 was a 36 % regression versus batch 4.
- **`tile_inner_px` changes results** — 459 and 456 stems versus 458. It is not
  a performance knob; it alters tiling and therefore seam handling.

## Results-changing vs performance-only

Performance-only — safe to tune, output must not move:
`max_cpu_workers`, `max_gpu_workers`, `*_cpu_workers`,
`max_vector_tile_workers`, `prediction_batch_*`, `prediction_producer_*`,
`producer_queue_batches`, `progress_interval_s*`, `compress_output`.

**Changes results** — the golden fixtures pin these, so a change invalidates
comparisons against earlier runs: `stem_binary_threshold`, `min_length`,
`max_distance`, `tolerance_angle`, `measuring_point_spacing_m`,
`diameter_method`, `max_tree_height`, `tile_overlap_m`, **`tile_inner_px`**,
`img_width`/`img_height`.

`tile_inner_px` is in this group on measured evidence, not on principle: the
sweep above produced 459 and 456 stems against a baseline of 458 purely by
changing it. It looks like a performance knob and is not one.

After tuning anything in the first group, confirm the stem count is unchanged.
A faster run that finds a different number of stems is not a faster run.
