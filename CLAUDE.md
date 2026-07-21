# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**WINMOL Analyzer** detects and quantifies windthrown (storm-felled) tree stems on UAV orthomosaics using a U-Net segmentation model plus a heuristic stem-reconstruction pipeline. It ships as **two runtimes sharing one core**:

1. **QGIS plugin** (Qt GUI) — `winmol_analyzer.py` + `winmol_analyzer_dialog.py`. The plugin *never runs TensorFlow itself*; it shells out to `winmol_run.py` inside a dedicated virtualenv (`winmol_venv`) created by `plugin_utils/installer.py`. This isolates CUDA/TF from QGIS's own Python.
2. **Standalone / batch CLI** — `winmol_run.py` (single image) and `winmol_batch.py` (folder), plus `standalone/` (notebook + a legacy `WINMOL_Analyzer.py` entry).

Both paths converge on `winmol_run.py`, the single orchestrator.

## Running the pipeline

The core entry point is invoked with 5 positional args:

```bash
python -u winmol_run.py <model_path> <input.tif> <stem_map.tif> <output_prefix> <Stems|Trees|Nodes>
```

- `process_type` controls how far the pipeline runs:
  - **`Stems`** — prediction only; writes the binary stem-map raster. No vectorization.
  - **`Trees` / `Nodes`** — full pipeline: prediction → tiled vectorization → merge into a GeoPackage of individual stems with diameters/volumes.
- **`WINMOL_CONFIG_OVERRIDES_JSON`** env var (a JSON object) overrides any `Config` attribute at runtime — the primary knob for tuning without editing code.

Batch over a folder (models resolved from `config.json`, expected under `standalone/model/`):

```bash
python winmol_batch.py <Spruce|Beech|Spruce_Deadwood|General> --input <dir> --output <dir> [--merge]
```

## Build / lint / release

- **Lint (the only automated check in CI):** `flake8` — config in `setup.cfg` (`max-line-length=80`, `max-doc-length=130`, ignores `T201,W503,W504`, plus per-file ignores). CI (`.github/workflows/on-pull-request.yml`) runs flake8 on Python 3.11 for every push/PR. `isort` uses `profile=black`. There is **no automated test suite** despite the `make test` stub (which needs a live QGIS).
- **Package the QGIS plugin:** the `Makefile` targets (`deploy`, `zip`, `package`) are for QGIS plugin packaging (`pb_tool`-style); `compile` builds `resources.py` from `resources.qrc` via `pyrcc5`. Release ZIPs are produced automatically on git **tag** push (`on-push-tags.yml`), which inlines the tag into `metadata.txt`. Only `v*` tags trigger a release; `scripts/plugin_version.py` derives the version (one leading `v` stripped, dotted-numeric with an optional alphanumeric suffix — `v0.6.0.2` and `v0.0.0-demo1` are both valid, a tag with no numeric lead like `models-onnx-v1` is rejected) and names the artifact `WINMOL_Analyzer-<version>.zip`. A second copy is uploaded as `WINMOL_Analyzer_QGIS_Plugin.zip` because `documentation/index.html` links `releases/latest/download/` under that fixed name. The directory *inside* the archive stays `WINMOL_Analyzer` — QGIS keys the installed plugin on it. Note a suffixed version such as `0.0.0-demo1` sorts *below* `0.6.0` in QGIS's installer, so demo tags never present themselves as upgrades.
- **Python / deps:** target 3.9–3.11. `requirements/base.txt` (Pillow, Shapely, geopandas, rasterio, scikit-image, scipy, pyogrio) + `requirements/tensorflow.txt` (`tensorflow[and-cuda]`) or `tensorflow-win.txt` on Windows.

## Architecture — the big picture

**`winmol_run.py::ImageProcessing`** is the orchestrator. Its `main()` flow:

1. **`HardwareInfo.detect()`** (`classes/HardwareInfo.py`) — probes CPU/RAM/GPU (via `nvidia-smi`).
2. **`build_execution_plan()`** (`classes/ExecutionPlan.py`) — the **central planner and the file to read first for runtime behavior**. Given `Config` + `HardwareInfo` + `RasterInfo`, it decides `prediction_mode` (`cpu_stream` | `stream` | `multi_gpu_stream`), `vector_mode`, tile sizes, batch sizes, and CPU/GPU worker splits. `Config.prediction_backend` (`auto|cpu|single_gpu|multi_gpu`) forces the scenario.
3. **Prediction phase** — `utils/Prediction.py` streams U-Net inference tile-by-tile straight to the stem-map raster; `utils/PredictWorkers.py` handles the multi-GPU path.
4. **Vector phase** (Trees/Nodes only) — the raster is cut into large tiles (`utils/Tiling.py`), empty tiles skipped, and each processed by `utils/VectorTilePipeline.py`, then stitched by `IO.merge_and_filter_tiled_results` (edge-buffer dedup across tile seams).

**The per-tile vector reconstruction sequence** (see `ImageProcessing.trees_processing` and `standalone/WINMOL_Analyzer.py`, which shows the un-tiled version most legibly):

`Skel.find_segments` → `Vec.restore_geoinformation` → `Vec.build_stem_parts` → `Vec.connect_stems` (heuristic joining of occluded/broken stems) → `Vec.rebuild_endnodes_from_stems` → `Quant.quantify_stems` (samples diameter every `measuring_point_spacing_m`, estimates volume via truncated-cone integration).

**Domain model** (`classes/`): a `Stem` is an ordered list of `Node`s (each a point + diameter) exposing `length`/`volume`; `Part`/`Vector` are the intermediate geometry primitives assembled before stems are connected.

**Config** (`classes/Config.py`) is a plain class of class-level attributes (not a dataclass) grouped by concern: backend selection, tiling/streaming, resource caps, prediction batching + autotune, segmentation (`img_width/height=512`, `num_classes=1`), vectorization thresholds, and diameter/volume method (`contour` vs `edt`). The planner writes resolved runtime values back onto the `Config` instance.

**Models** are U-Net HDF5 files hosted on Zenodo; URLs live in `config.json` (`Beech`, `Spruce`, `Spruce_Deadwood`, `General`). The plugin downloads them into `models/` on first run; batch expects them pre-downloaded in `standalone/model/`.

**Docker** (`Dockerfile` + `startDocker.sh`) runs the whole plugin inside the official `qgis/qgis` image for reproducible GUI testing via X11.

---

# context-mode — MANDATORY routing rules

You have context-mode MCP tools available. These rules are NOT optional — they protect your context window from flooding. A single unrouted command can dump 56 KB into context and waste the entire session.

## BLOCKED commands — do NOT attempt these

### curl / wget — BLOCKED
Any Bash command containing `curl` or `wget` is intercepted and replaced with an error message. Do NOT retry.
Instead use:
- `ctx_fetch_and_index(url, source)` to fetch and index web pages
- `ctx_execute(language: "javascript", code: "const r = await fetch(...)")` to run HTTP calls in sandbox

### Inline HTTP — BLOCKED
Any Bash command containing `fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, or `http.request(` is intercepted and replaced with an error message. Do NOT retry with Bash.
Instead use:
- `ctx_execute(language, code)` to run HTTP calls in sandbox — only stdout enters context

### WebFetch — BLOCKED
WebFetch calls are denied entirely. The URL is extracted and you are told to use `ctx_fetch_and_index` instead.
Instead use:
- `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` to query the indexed content

## REDIRECTED tools — use sandbox equivalents

### Bash (>20 lines output)
Bash is ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`, and other short-output commands.
For everything else, use:
- `ctx_batch_execute(commands, queries)` — run multiple commands + search in ONE call
- `ctx_execute(language: "shell", code: "...")` — run in sandbox, only stdout enters context

### Read (for analysis)
If you are reading a file to **Edit** it → Read is correct (Edit needs content in context).
If you are reading to **analyze, explore, or summarize** → use `ctx_execute_file(path, language, code)` instead. Only your printed summary enters context. The raw file content stays in the sandbox.

### Grep (large results)
Grep results can flood context. Use `ctx_execute(language: "shell", code: "grep ...")` to run searches in sandbox. Only your printed summary enters context.

## Tool selection hierarchy

1. **GATHER**: `ctx_batch_execute(commands, queries)` — Primary tool. Runs all commands, auto-indexes output, returns search results. ONE call replaces 30+ individual calls.
2. **FOLLOW-UP**: `ctx_search(queries: ["q1", "q2", ...])` — Query indexed content. Pass ALL questions as array in ONE call.
3. **PROCESSING**: `ctx_execute(language, code)` | `ctx_execute_file(path, language, code)` — Sandbox execution. Only stdout enters context.
4. **WEB**: `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` — Fetch, chunk, index, query. Raw HTML never enters context.
5. **INDEX**: `ctx_index(content, source)` — Store content in FTS5 knowledge base for later search.

## Subagent routing

When spawning subagents (Agent/Task tool), the routing block is automatically injected into their prompt. Bash-type subagents are upgraded to general-purpose so they have access to MCP tools. You do NOT need to manually instruct subagents about context-mode.

## Output constraints

- Keep responses under 500 words.
- Write artifacts (code, configs, PRDs) to FILES — never return them as inline text. Return only: file path + 1-line description.
- When indexing content, use descriptive source labels so others can `ctx_search(source: "label")` later.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call the `ctx_stats` MCP tool and display the full output verbatim |
| `ctx doctor` | Call the `ctx_doctor` MCP tool, run the returned shell command, display as checklist |
| `ctx upgrade` | Call the `ctx_upgrade` MCP tool, run the returned shell command, display as checklist |
