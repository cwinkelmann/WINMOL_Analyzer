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
import glob
import os
import sys

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


def _loader_path_var():
    """The environment variable the platform's dynamic loader reads."""
    if os.name == "nt":
        return "PATH"
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def nvidia_lib_dirs(python_exe):
    """The CUDA/cuDNN library directories inside ``python_exe``'s env.

    ``onnxruntime-gpu`` does not bundle CUDA: it pulls in the
    ``nvidia-*-cu12`` wheels, which drop their shared objects in
    ``<venv>/lib/python3.X/site-packages/nvidia/<component>/lib`` (seven such
    directories) — a place no dynamic loader looks. The provider is then
    COMPILED IN and listed by ``get_available_providers()`` but cannot be
    CREATED: ``InferenceSession`` warns "Require cuDNN 9.* and CUDA 12.* …
    make sure they're in the PATH" and silently returns a CPU session.
    Measured cost of that silence on an RTX 4080 SUPER: 10311 ms per tile
    instead of 10.5 ms.

    Globbed rather than asked of the interpreter: this is called on paths
    that may not even exist, and spawning the child to find out would put a
    subprocess on the GUI thread.
    """
    if not python_exe:
        return []
    root = os.path.dirname(os.path.dirname(os.path.abspath(str(python_exe))))
    patterns = (
        os.path.join(root, "lib", "python3.*", "site-packages", "nvidia",
                     "*", "lib"),
        os.path.join(root, "lib64", "python3.*", "site-packages", "nvidia",
                     "*", "lib"),
        # Windows lays the venv out as <venv>\Lib\site-packages and ships the
        # DLLs in bin\ next to lib\.
        os.path.join(root, "Lib", "site-packages", "nvidia", "*", "lib"),
        os.path.join(root, "Lib", "site-packages", "nvidia", "*", "bin"),
    )
    dirs = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isdir(path) and path not in dirs:
                dirs.append(path)
    return dirs


def native_lib_extra(python_exe, env=None):
    """``{loader_var: value}`` putting :func:`nvidia_lib_dirs` on the path.

    Empty when there are no such directories — i.e. on every CPU-only and
    macOS/CoreML environment, which is why this is safe to call
    unconditionally. The existing value is PREPENDED to, never replaced: on
    Windows the variable is PATH, and clobbering it would break the child
    far more thoroughly than a missing GPU.

    Belt and braces for :func:`utils.onnx_runtime.preload_native_libs`,
    which does the same job in-process and needs no environment at all —
    but only exists in onnxruntime >= 1.21.
    """
    dirs = nvidia_lib_dirs(python_exe)
    if not dirs:
        return {}
    var = _loader_path_var()
    current = (os.environ if env is None else env).get(var, "")
    have = current.split(os.pathsep) if current else []
    new = [d for d in dirs if d not in have]
    if not new:
        return {}
    return {var: os.pathsep.join(new + have)}


def child_env(extra=None, python_exe=None):
    """A copy of os.environ safe to hand to a foreign Python interpreter.

    Removes the interpreter- and GDAL-locating variables QGIS exports, pins the
    hash seed so results stay reproducible (the same guarantee winmol_run.py's
    re-exec guard provides), and disables user site-packages so a stray
    ~/.local install cannot shadow the venv.

    ``python_exe`` names the interpreter the environment is FOR; when given,
    the CUDA/cuDNN directories of that interpreter's ``nvidia-*`` wheels are
    prepended to the loader path (:func:`native_lib_extra`).

    ``extra`` is applied last, so a caller that genuinely needs one of the
    stripped variables (e.g. the compute-contract test injecting a PYTHONPATH)
    can set it explicitly. os.environ is never mutated.
    """
    env = dict(os.environ)
    for var in STRIPPED:
        env.pop(var, None)
    # See native_lib_extra(): without the CUDA/cuDNN wheel directories on the
    # loader path, onnxruntime-gpu builds a CPU session and says nothing.
    env.update(native_lib_extra(python_exe, env=env))
    # Deterministic stem counts: connect_stems iterates a set of string-hashed
    # Part objects, so an unpinned seed changes the result (see PR #4).
    env["PYTHONHASHSEED"] = "0"
    # Ignore ~/.local/lib/pythonX.Y/site-packages; the venv must be complete.
    env["PYTHONNOUSERSITE"] = "1"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env
