# Repository root layout

Why the root looks the way it does, what each file is for, and what could
still move. Written during the "clean up the root dir" audit so the next
person does not have to re-derive it.

The short version: **most of the root is not clutter, it is a QGIS
requirement.** QGIS loads a plugin from one flat directory and resolves
`metadata.txt`, `icon.png` and `classFactory()` relative to it. And the
release artifact already strips every development file — see
[What actually ships](#what-actually-ships).

## Verdict per root file

### Removed (dead, proven)

| File | Evidence |
| --- | --- |
| `plugin_upload.py` | No Python file imported it, and it could not be imported: it called `standard_library.install_aliases()` without importing `standard_library`, raising `NameError` (which `setup.cfg` silenced with an `F821` per-file ignore). Its only caller, `make upload`, expanded `$(c)/plugin_upload.py` with `$(c)` undefined — `make -n upload` printed `/plugin_upload.py`. It never shipped. |
| `pylintrc` | Single consumer was the Makefile `pylint` target. pylint is in no `requirements/` file and no workflow; CI runs flake8 only. |

Also removed with them: the Makefile `upload`, `pylint`, `transup`,
`transcompile`, `transclean` and `doc` targets, all of which referenced
files or directories that do not exist (`scripts/update-strings.sh`,
`scripts/compile-strings.sh`, `i18n/`, `help/`).

### Kept, and why

| File | Why it must stay at the root |
| --- | --- |
| `__init__.py` | `classFactory(iface)` is the QGIS entry contract; QGIS imports the plugin *directory* as a package. |
| `metadata.txt` | QGIS's metadata scanner reads `<plugin_dir>/metadata.txt`. `scripts/build_plugin_zip.sh` rewrites `version=` at release time. |
| `icon.png` | `metadata.txt` has `icon=icon.png`, resolved relative to the plugin dir, and `winmol_analyzer.py` loads it from disk by path. |
| `winmol_analyzer.py`, `winmol_analyzer_dialog.py` | The plugin GUI. |
| `winmol_analyzer_dialog_base.ui` | Loaded at runtime with `uic` from a `__file__`-relative path — it is not compiled, so it must ship. |
| `tasks_threads.py` | Live: `winmol_analyzer_dialog.py` imports `Worker` and `EnvSetupWorker` from it. Looks like scaffolding, is not. |
| `winmol_run.py`, `winmol_batch.py`, `config.json` | The compute core the plugin's venv subprocess executes. `config.json` is read via `os.path.dirname(__file__)` from the root by both `winmol_batch.py` and the dialog. |
| `Makefile`, `setup.cfg`, `.gitignore`, `README.md`, `LICENSE`, `CLAUDE.md` | Development infrastructure; already excluded from the shipped zip. |
| `Dockerfile`, `startDocker.sh` | The `qgis/qgis` GUI-test image. Stripped from the zip. |

### Kept, but on notice

**`resources.py` + `resources.qrc`.** Functionally dead, deliberately not
deleted in this pass.

Dead because: `resources.qrc` compiles exactly one file, `icon.png`; no
code anywhere resolves a `:/plugins/...` path (the only two hits in the
repository are comments); `winmol_analyzer.py` loads the icon from disk
on purpose, so it works under both Qt5 and Qt6; the `.ui` file's
`<resources/>` element is empty; and the `from .resources import *` is
inside a `try/except Exception: pass`.

Not deleted because: `scripts/build_plugin_zip.sh` lists `resources.py`
in its hard-fail `required` loop and aborts with
`FATAL: resources.py missing from package`. That script only runs in CI
on a **tag** push, so a naive `git rm resources.py` passes PR CI and
breaks the next release. Removing it is a coordinated edit — the file,
the `.qrc`, the try/except import, the `setup.cfg` per-file-ignore, five
Makefile references (`COMPILED_RESOURCE_FILES`, `RESOURCE_SRC`, the
`compile` target and its two dependents, the `cp` in `deploy`,
`PEP8EXCLUDE`), and both the `required` list and the `EXCLUDE` list in
the build script. Worth its own reviewed commit, not a sweep.

`tests/test_plugin_package.py::test_script_requires_its_own_manifest`
now fails immediately if the file is removed without the script edit.

**`pb_tool.cfg`.** *Removed.* Nothing invoked pb_tool — not a workflow,
not `requirements/`, and the Makefile never ran it. It was a stale second
copy of the shipped-file list that could silently drift from the
Makefile's copy, so it was deleted (and dropped from the build script's
`EXCLUDE` list). Its only remaining trace is the "pb_tool-style" wording
in `CLAUDE.md`, which describes the Makefile targets, not the file.

### Not repository files

`Dockerfile-1`, `Dockerfile.blackwell`, `Dockerfile_carrot`,
`Dockerfile_olive`, `push_restack.sh`, `WINMOL_Analyzer-*.zip`,
`.DS_Store`, `.idea/`, `.pytest_cache/` — untracked local files
(`git ls-files` lists only `Dockerfile`). The Dockerfile variants, the
restack helper and built release zips are now covered by `.gitignore`, so
they no longer clutter `git status`; the rest were already ignored.

## What actually ships

`scripts/build_plugin_zip.sh` is the single source of truth and already
strips `.github`, `Makefile`, `setup.cfg`, `docker`,
`startDocker.sh`, `scripts`, `tests`, `benchmark`, `docs`,
`documentation`, `standalone` and `resources.qrc`. The
"package the QGIS-only files away" goal is therefore already met in the
direction that reaches users; what remains in the root is development
tooling plus files QGIS requires to be flat.

`tests/test_plugin_package.py` pins that manifest in both directions.

## Proposal: a `qgis_plugin/` subpackage (not done)

This is the reverse move — hiding the GUI files from the root — and it is
the risky one, because there is no live-QGIS test anywhere in CI.

**Could move:** `winmol_analyzer.py`, `winmol_analyzer_dialog.py`,
`winmol_analyzer_dialog_base.ui`, `tasks_threads.py`, and `resources.py`
if it survives. Every reference to them is either a relative import or
`__file__`-relative, so all of them follow the move mechanically.

**Cannot move:** `__init__.py`, `metadata.txt`, `icon.png`,
`config.json`, `winmol_run.py`, and the `classes/`, `utils/`,
`plugin_utils/`, `requirements/` directories — for the reasons in the
table above.

**Costs, and why it was deferred:**

1. `make deploy` flat-`cp`s `PY_FILES` and `UI_FILES` into the QGIS
   profile directory. Nested paths land in the wrong place unless the
   target learns `mkdir -p`.
2. pb_tool flattens files into the plugin directory by design, so its
   config could not express a subdirectory. `pb_tool.cfg` has since been
   deleted, so this no longer blocks the move.
3. `winmol_analyzer_dialog.py` is ~53 KB and reaches into
   `plugin_utils/` and `winmol_run.py`. A wrong relative-import level
   fails only when QGIS loads the plugin, and no CI job loads QGIS.

**Sequence if it is wanted:** remove `resources.py` → move the files
(`pb_tool.cfg` is already deleted) → teach `make deploy` about nesting →
manual QGIS smoke test on macOS *and* Windows (the same manual gate the
ONNX migration used) → merge. Nothing in CI substitutes for that last
step.
