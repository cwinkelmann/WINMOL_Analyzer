"""Sanitized environment for the child processes WINMOL spawns.

QGIS exports variables that describe ITS OWN Python (PYTHONHOME/
PYTHONPATH) and ITS OWN GDAL (GDAL_DATA/PROJ_LIB). Every child WINMOL
spawns runs a DIFFERENT Python (the compute venv) with its own vendored
GDAL, so inheriting those is actively harmful:

* Windows -- QGIS's stdlib layout is version-agnostic, so a
  different-version interpreter half-loads it: "Could not import runpy
  module" (``AssertionError: SRE module mismatch``).
* POSIX -- the same leak usually misses (the path embeds the version),
  but when it hits: "No module named 'encodings'".
* Any platform -- a leaked GDAL_DATA/PROJ_LIB points a *vendored* GDAL
  at the wrong data files: "Cannot find proj.db" or silently wrong CRS.

On Windows, PATH doubles as the DLL search path, so an inherited QGIS
PATH can shadow the DLL an onnxruntime native extension binds.

Pure stdlib, no QGIS imports -- unit-testable off QGIS.
"""
import ntpath
import os
import subprocess
import sys
import tempfile

#: The one-line version probe several callers feed to
#: :func:`run_isolated` ("3.11" on stdout when the interpreter runs).
PY_VERSION_PROBE = "import sys;print('%d.%d' % sys.version_info[:2])"

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

# Variables that point a GDAL/PROJ build at its data files. The compute
# venv ships its own GDAL (rasterio/pyogrio wheels); QGIS's paths belong
# to a different GDAL version.
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

# Parent-env variables that betray a QGIS/OSGeo4W install. Read BEFORE
# STRIPPED removes the GDAL/PROJ ones, so the PATH sanitizer can still
# learn the install roots from them.
_QGIS_MARKER_VARS = (
    "OSGEO4W_ROOT",
    "QGIS_PREFIX_PATH",
    "GDAL_DRIVER_PATH",
    "GDAL_DATA",
    "GDAL_PLUGIN_PATH",
    "PROJ_LIB",
    "PROJ_DATA",
)

# The "\apps\" layout segment shared by OSGeo4W and standalone-QGIS trees.
_APPS_MARKER = ntpath.sep + "apps" + ntpath.sep


def _nt_norm(path):
    """Case-fold + normalize a Windows path (testable off Windows)."""
    return ntpath.normcase(ntpath.normpath(path.strip().strip('"')))


def _apps_root(norm_path):
    """The install root implied by an OSGeo "\\apps\\" path, or None."""
    idx = norm_path.find(_APPS_MARKER)
    return norm_path[:idx] if idx > 0 else None


def _qgis_osgeo_roots(parent_env):
    """Normalized QGIS/OSGeo install roots learnt from marker vars.

    Empty when the parent shows no QGIS markers -- the reason
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
    """Whether a normalized path is at or under a normalized root."""
    if not root_norm:
        return False
    if entry_norm == root_norm:
        return True
    return entry_norm.startswith(root_norm + ntpath.sep)


def _looks_like_qgis(entry_norm):
    """Segment heuristic catching a QGIS/OSGeo4W path with no marker.

    Matches "...\\qgis\\bin", "...\\osgeo4w\\bin" and the "\\apps\\"
    layout directly, so a stray PATH entry survives a missing
    OSGEO4W_ROOT. ``keep_roots`` in :func:`sanitize_windows_path`
    exempts the child's own venv from this heuristic.
    """
    segments = [s for s in entry_norm.split(ntpath.sep) if s]
    for seg in segments:
        if seg.startswith("qgis") or seg.startswith("osgeo4w"):
            return True
    return _APPS_MARKER in entry_norm


def sanitize_windows_path(path_value, parent_env, keep_roots=()):
    """Drop QGIS/OSGeo directories from a Windows PATH; keep the rest.

    An entry is dropped when it is at/under a root learnt from
    ``parent_env`` (:func:`_qgis_osgeo_roots`) OR matches
    :func:`_looks_like_qgis`. Entries under ``keep_roots`` -- the
    child's own venv -- are always kept, even though that venv can live
    under a "...\\QGIS\\QGIS3\\..." profile path. Pure and
    order-preserving; a PATH with no QGIS markers comes back unchanged.
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
        if any(_is_under(norm, r) for r in roots) or \
                _looks_like_qgis(norm):
            continue
        kept.append(entry)
    return ";".join(kept)


def safe_child_cwd(python_exe=None):
    """A working directory safe to launch a child process from.

    On Windows the process cwd is also on the DLL search order, so a
    QGIS working directory can shadow the child's native extensions.
    Returns the child's own venv root when derivable from
    ``python_exe``, else a neutral temp directory.
    """
    if python_exe:
        root = os.path.dirname(
            os.path.dirname(os.path.abspath(str(python_exe))))
        if os.path.isdir(root):
            return root
    return tempfile.gettempdir()


def run_isolated(python_exe, code, timeout, args=()):
    """Run ``code`` in ``python_exe`` isolated (``-I``) under
    :func:`child_env`, returning the ``CompletedProcess`` (text mode,
    output captured, exit code never checked). ``args`` become the
    child's ``sys.argv[1:]``. Exceptions (missing executable,
    ``TimeoutExpired``) propagate — callers own their fallbacks."""
    return subprocess.run(
        [python_exe, "-I", "-c", code, *args],
        capture_output=True, text=True, timeout=timeout,
        env=child_env(python_exe=python_exe))


def child_env(extra=None, python_exe=None):
    """A copy of os.environ safe to hand to a foreign Python interpreter.

    Strips the interpreter- and GDAL-locating variables QGIS exports,
    pins the hash seed so results stay reproducible, and disables user
    site-packages so a stray ~/.local install cannot shadow the venv.
    On win32, PATH is rewritten via :func:`sanitize_windows_path` with
    ``keep_roots`` set to ``python_exe``'s venv root, since PATH is
    also the child's DLL search path. ``extra`` is applied LAST, so a
    caller that genuinely needs a stripped variable can set it
    explicitly. ``os.environ`` itself is never mutated.
    """
    env = dict(os.environ)
    # Capture the QGIS/OSGeo markers BEFORE the strip loop removes the
    # GDAL/PROJ ones; the PATH sanitizer learns the install roots from
    # them.
    qgis_markers = {k: env.get(k) for k in _QGIS_MARKER_VARS if env.get(k)}
    for var in STRIPPED:
        env.pop(var, None)
    if sys.platform == "win32":
        keep_roots = ()
        if python_exe:
            keep_roots = (os.path.dirname(
                os.path.dirname(os.path.abspath(str(python_exe)))),)
        env["PATH"] = sanitize_windows_path(
            env.get("PATH", ""), qgis_markers, keep_roots=keep_roots)
    # Deterministic stem counts: connect_stems iterates a set of
    # string-hashed Part objects, so an unpinned seed changes the
    # result (see PR #4).
    env["PYTHONHASHSEED"] = "0"
    # Ignore ~/.local/lib/pythonX.Y/site-packages; the venv must be
    # complete.
    env["PYTHONNOUSERSITE"] = "1"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env
