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
import ntpath
import os
import sys
import tempfile

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

# Parent-environment variables that betray a QGIS/OSGeo4W install. Read BEFORE
# STRIPPED removes the GDAL/PROJ ones, so the PATH sanitizer can learn the
# install roots from them. QGIS_PREFIX_PATH and the GDAL/PROJ data paths point
# DEEP into the tree (``…\apps\gdal\share\gdal``); the install root is what sits
# before the ``\apps\`` segment.
_QGIS_MARKER_VARS = (
    "OSGEO4W_ROOT",
    "QGIS_PREFIX_PATH",
    "GDAL_DRIVER_PATH",
    "GDAL_DATA",
    "GDAL_PLUGIN_PATH",
    "PROJ_LIB",
    "PROJ_DATA",
)

# The ``\apps\`` layout segment shared by OSGeo4W and standalone-QGIS trees.
_APPS_MARKER = ntpath.sep + "apps" + ntpath.sep


def _nt_norm(path):
    """Case-fold + normalise a Windows path for comparison.

    Uses ``ntpath`` explicitly so the sanitizer is testable on macOS/Linux,
    where ``os.path`` is POSIX. Surrounding quotes (legal in PATH entries) are
    stripped first.
    """
    return ntpath.normcase(ntpath.normpath(path.strip().strip('"')))


def _apps_root(norm_path):
    """The install root implied by an OSGeo ``\\apps\\`` data path, or None."""
    idx = norm_path.find(_APPS_MARKER)
    if idx > 0:
        return norm_path[:idx]
    return None


def _qgis_osgeo_roots(parent_env):
    """Normalised QGIS/OSGeo install roots learnt from ``parent_env`` markers.

    ``OSGEO4W_ROOT`` is already a root; every other marker points inside the
    tree, so both the value itself and the ``\\apps\\``-derived install root are
    recorded. Empty when the parent shows no QGIS markers at all — which is why
    :func:`sanitize_windows_path` is a no-op outside QGIS.
    """
    roots = set()
    for var in _QGIS_MARKER_VARS:
        value = parent_env.get(var)
        if not value:
            continue
        norm = _nt_norm(value)
        if var == "OSGEO4W_ROOT":
            roots.add(norm)
            continue
        apps_root = _apps_root(norm)
        if apps_root:
            roots.add(apps_root)
    return {r for r in roots if r}


def _is_under(entry_norm, root_norm):
    """Whether a normalised path is at or under a normalised root."""
    if not root_norm:
        return False
    if entry_norm == root_norm:
        return True
    return entry_norm.startswith(root_norm + ntpath.sep)


def _looks_like_qgis(entry_norm):
    """Segment heuristic: a QGIS/OSGeo4W path even with no marker to prove it.

    Catches ``…\\qgis\\bin`` / ``…\\OSGeo4W\\bin`` and the ``\\apps\\`` layout
    directly, so a stray PATH entry survives a missing OSGEO4W_ROOT. Kept out of
    the venv's way by :func:`sanitize_windows_path`'s ``keep_roots``.
    """
    segments = [s for s in entry_norm.split(ntpath.sep) if s]
    for seg in segments:
        if seg.startswith("qgis") or seg.startswith("osgeo4w"):
            return True
    return _APPS_MARKER in entry_norm


def sanitize_windows_path(path_value, parent_env, keep_roots=()):
    """Drop QGIS/OSGeo directories from a Windows PATH; keep everything else.

    The child WINMOL spawns runs a self-contained venv (rasterio/pyogrio bundle
    their own GDAL); it needs NOTHING from QGIS's PATH. But on Windows the PATH
    is also the DLL search path, so QGIS's program directory — full of that
    QGIS's MSVC/Qt runtime DLLs — is inherited into the child and can shadow the
    dependency an onnxruntime native extension binds, which is the QGIS-3.28
    "DLL initialization routine failed" bug.

    An entry is dropped when it is at/under a root learnt from ``parent_env``
    (:func:`_qgis_osgeo_roots`) OR matches the segment heuristic
    (:func:`_looks_like_qgis`). Entries under ``keep_roots`` — the child's own
    venv — are always kept, even though the plugin venv itself lives under a
    ``…\\QGIS\\QGIS3\\…`` profile path. System32, %SystemRoot% and everything
    unrelated pass through untouched. Pure and order-preserving; a bare Windows
    PATH with no QGIS markers comes back unchanged.
    """
    if not path_value:
        return path_value
    roots = _qgis_osgeo_roots(parent_env)
    keep = [_nt_norm(r) for r in keep_roots if r]
    kept = []
    for entry in path_value.split(";"):
        if not entry:
            continue
        norm = _nt_norm(entry)
        if any(_is_under(norm, k) for k in keep):
            kept.append(entry)
            continue
        if any(_is_under(norm, r) for r in roots) or _looks_like_qgis(norm):
            continue
        kept.append(entry)
    return ";".join(kept)


def safe_child_cwd(python_exe=None):
    """A working directory safe to launch a child process from.

    On Windows the process's current directory is on the DLL search order, so a
    QGIS working directory can shadow the child's native extensions exactly as
    an inherited PATH can. Returns the child's own venv root when it can be
    derived from ``python_exe`` (self-contained and under our control), else a
    neutral temp directory. Harmless on every platform, and winmol_run.py takes
    absolute paths so it does not depend on the working directory.
    """
    if python_exe:
        root = os.path.dirname(
            os.path.dirname(os.path.abspath(str(python_exe))))
        if os.path.isdir(root):
            return root
    return tempfile.gettempdir()


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
    # Capture the QGIS/OSGeo markers BEFORE the strip loop removes the GDAL/PROJ
    # ones; the PATH sanitizer learns the install roots from them.
    qgis_markers = {k: env.get(k) for k in _QGIS_MARKER_VARS if env.get(k)}
    for var in STRIPPED:
        env.pop(var, None)
    # Windows only: QGIS's program directory is inherited on PATH and, because
    # PATH is also the DLL search path, its 2022-era MSVC/Qt runtime DLLs can
    # shadow what an onnxruntime native extension binds (the QGIS-3.28 "DLL
    # initialization routine failed" bug). Strip QGIS/OSGeo entries; keep the
    # child's own venv even though it lives under a QGIS profile path. No-op on
    # POSIX and when the parent shows no QGIS markers.
    if sys.platform == "win32":
        keep_roots = ()
        if python_exe:
            keep_roots = (os.path.dirname(
                os.path.dirname(os.path.abspath(str(python_exe)))),)
        env["PATH"] = sanitize_windows_path(
            env.get("PATH", ""), qgis_markers, keep_roots=keep_roots)
    # See native_lib_extra(): without the CUDA/cuDNN wheel directories on the
    # loader path, onnxruntime-gpu builds a CPU session and says nothing. Runs
    # AFTER the PATH sanitizer so it prepends to the sanitized value.
    env.update(native_lib_extra(python_exe, env=env))
    # Deterministic stem counts: connect_stems iterates a set of string-hashed
    # Part objects, so an unpinned seed changes the result (see PR #4).
    env["PYTHONHASHSEED"] = "0"
    # Ignore ~/.local/lib/pythonX.Y/site-packages; the venv must be complete.
    env["PYTHONNOUSERSITE"] = "1"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env
