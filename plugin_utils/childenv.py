"""A sanitized environment for the child processes WINMOL spawns.

QGIS runs the plugin inside its OWN Python and exports environment variables
that describe it: PYTHONHOME/PYTHONPATH point at QGIS's interpreter and stdlib,
GDAL_DATA/PROJ_LIB at QGIS's geo data. Every process WINMOL spawns runs a
DIFFERENT Python (the compute venv) with its OWN vendored GDAL, so inheriting
those variables is actively harmful:

* Windows -- QGIS's stdlib layout is version-agnostic (``<home>\\Lib``), so a
  3.11 interpreter finds QGIS's 3.12 stdlib and half-loads it: C extensions
  come from the 3.11 binary, pure-Python modules from 3.12. The result is
  ``Could not import runpy module`` (with ``AssertionError: SRE module
  mismatch``) and env creation dies.
* macOS/Linux -- the path embeds the version (``lib/python3.11``), so the same
  leak misses entirely and gives ``No module named 'encodings'``.
* Any platform -- a leaked GDAL_DATA/PROJ_LIB points a *vendored* GDAL at a
  different GDAL's data files, which surfaces as ``Cannot find proj.db`` or
  silently wrong CRS handling.

Pure stdlib and free of QGIS imports, so plugin_utils, tasks_threads and the
dialog can all use it, and it is unit-testable off QGIS.
"""
import os

# Variables that tell a Python interpreter where it lives. Poison for any
# interpreter other than the one QGIS is running.
_INTERPRETER_VARS = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONEXECUTABLE",
    "PYTHONUSERBASE",
    "PYTHONPLATLIBDIR",
    "PYTHONFRAMEWORK",
    "PYTHONCASEOK",
    "PYTHONMALLOC",
    "PYTHONDEVMODE",
    "PYTHONOPTIMIZE",
    "PYTHONIOENCODING",
    "__PYVENV_LAUNCHER__",
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
)

# Variables that point a GDAL/PROJ build at its data files. The compute venv
# ships its own GDAL inside the rasterio/pyogrio wheels; QGIS's paths belong to
# a different GDAL version.
_GEO_VARS = (
    "GDAL_DATA",
    "GDAL_DRIVER_PATH",
    "GDAL_PLUGIN_PATH",
    "PROJ_LIB",
    "PROJ_DATA",
    "PROJ_NETWORK",
    "GEOTIFF_CSV",
    "CPL_ZIP_ENCODING",
)

STRIPPED = _INTERPRETER_VARS + _GEO_VARS


def child_env(extra=None):
    """A copy of os.environ safe to hand to a foreign Python interpreter.

    Removes the interpreter- and GDAL-locating variables QGIS exports, pins the
    hash seed so results stay reproducible (the same guarantee winmol_run.py's
    re-exec guard provides), and disables user site-packages so a stray
    ~/.local install cannot shadow the venv.

    ``extra`` is applied last, so a caller that genuinely needs one of the
    stripped variables (e.g. the compute-contract test injecting a PYTHONPATH)
    can set it explicitly. os.environ is never mutated.
    """
    env = dict(os.environ)
    for var in STRIPPED:
        env.pop(var, None)
    # Deterministic stem counts: connect_stems iterates a set of string-hashed
    # Part objects, so an unpinned seed changes the result (see PR #4).
    env["PYTHONHASHSEED"] = "0"
    # Ignore ~/.local/lib/pythonX.Y/site-packages; the venv must be complete.
    env["PYTHONNOUSERSITE"] = "1"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env
