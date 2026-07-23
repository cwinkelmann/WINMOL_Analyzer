"""Honest, testable diagnosis of an ``import onnxruntime`` failure.

onnxruntime failing to import has three very different causes with three
different remedies, and the historical guidance ("`pip install onnxruntime`")
was wrong for two of them:

1. The distribution is genuinely absent -> install it.
2. The distribution IS installed but its **native library** fails to load. On
   Windows this is the DLL-shadowing bug: another application (an older QGIS,
   3.28) puts its own MSVC/Qt runtime DLLs on the child's DLL search path and
   ``onnxruntime_pybind11_state`` binds an incompatible dependency and dies
   with "A dynamic link library (DLL) initialization routine failed." The
   package is present and pip cannot fix it; the fix is a clean search path
   (which WINMOL now enforces, see plugin_utils/childenv.py). It is NOT a
   missing Visual C++ redistributable — that was proven present, because the
   SAME venv imports fine when launched from QGIS 3.44.
3. The distribution is installed but fails for some other reason -> surface the
   real error.

Pure stdlib, no onnxruntime import (importing it is exactly what fails), so it
is safe to call from the ``except`` branch that caught that failure, and it is
unit-testable off Windows with stubbed inputs. ``plugin_utils`` is the one
package both runtimes share — the compute child imports it top-level, the
plugin relatively — so this single copy backs both
``utils.IO._load_onnx_model`` (run time) and ``installer.import_verdict``
(setup time), and their guidance cannot drift.
"""
import sys

#: Distribution names that all install the importable ``onnxruntime`` module.
#: Presence of any one means "onnxruntime is installed" regardless of variant.
_ONNXRUNTIME_DISTS = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-silicon",
    "onnxruntime-directml",
    "onnxruntime-openvino",
    "onnxruntime-qnn",
    "onnxruntime-training",
)

#: Substrings (lower-cased) that mark a Windows native-library load failure as
#: opposed to a plain ModuleNotFoundError. Matches the exact text of the
#: measured 3.28 failure: "DLL load failed while importing
#: onnxruntime_pybind11_state: A dynamic link library (DLL) initialization
#: routine failed."
_NATIVE_LOAD_MARKERS = (
    "dll load failed",
    "onnxruntime_pybind11_state",
    "dynamic link library",
    "initialization routine failed",
)

_ABSENT_MESSAGE = (
    "onnxruntime is not installed in this environment. Install it: "
    "`pip install onnxruntime` (or `onnxruntime-gpu` on an NVIDIA machine)."
)

_NATIVE_MESSAGE = (
    "onnxruntime is installed but its native library could not initialize. "
    "This is caused by another application's runtime libraries shadowing it "
    "on the Windows DLL search path (observed when the plugin is launched "
    "from older QGIS 3.28; QGIS 3.44+ is unaffected). WINMOL now sanitizes "
    "the child environment to prevent this, so restarting QGIS and retrying "
    "should clear it. If it persists, an out-of-date system Visual C++ "
    "runtime can also cause it."
)


def onnxruntime_distribution_present(version_lookup=None) -> bool:
    """Whether an onnxruntime distribution is installed, WITHOUT importing it.

    Importing onnxruntime is the operation that fails in the bug this module
    diagnoses, so presence is read from the installed distribution metadata
    (``importlib.metadata``), which never executes the package. Any of the
    known variant distributions counts. ``version_lookup`` is a seam for tests:
    a callable ``dist -> version`` (or raising ``PackageNotFoundError``).
    """
    try:
        from importlib.metadata import (PackageNotFoundError,
                                        version as _pkg_version)
    except Exception:            # pragma: no cover - importlib.metadata is std
        return False
    lookup = version_lookup or _pkg_version
    for dist in _ONNXRUNTIME_DISTS:
        try:
            lookup(dist)
            return True
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return False


def is_native_load_failure(error, platform=None) -> bool:
    """True when ``error`` is a Windows native-library (DLL) load failure.

    Gated to Windows: the shadowing failure and its "DLL load failed" text are
    a Windows phenomenon, and diagnosing a POSIX import error as one would send
    the user to the wrong remedy. ``error`` may be an exception or its text.
    """
    if platform is None:
        platform = sys.platform
    if platform != "win32":
        return False
    text = str(error or "").lower()
    return any(marker in text for marker in _NATIVE_LOAD_MARKERS)


def onnx_import_error_message(error, platform=None, present=None) -> str:
    """The honest, user-facing message for an ``import onnxruntime`` failure.

    ``error`` is the exception (or its text) that was raised. ``present``
    overrides the distribution check when the caller already knows the answer
    (the installer's probe reports the installed packages); leave it ``None``
    to have it read from distribution metadata. ``platform`` defaults to the
    host and exists so tests can assert the Windows branch off Windows.

    Never tells the user to ``pip install`` when the package is already there,
    and never asserts the Visual C++ redistributable as the primary fix.
    """
    if platform is None:
        platform = sys.platform
    if present is None:
        present = onnxruntime_distribution_present()
    text = str(error or "").strip()
    if not present:
        return _ABSENT_MESSAGE
    if is_native_load_failure(text, platform):
        if text:
            return f"{_NATIVE_MESSAGE} (original error: {text})"
        return _NATIVE_MESSAGE
    detail = text or "unknown error"
    return f"onnxruntime is installed but failed to load: {detail}."
