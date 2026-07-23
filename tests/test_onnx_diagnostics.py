"""Honest onnxruntime-import diagnosis, tested without Windows or a broken
onnxruntime.

The Windows DLL-shadowing bug (QGIS 3.28) leaves onnxruntime *installed* but
its native library unable to initialize. The old guidance —
"`pip install onnxruntime`" — was wrong for that case and for the not-yet-a-
problem "installed but fails for another reason" case. These tests pin the
three-way classification and the two consumers that share it: the run-time
message (`utils.IO._load_onnx_model`) and the setup-time verdict
(`installer.import_verdict`). All inputs are stubbed; nothing here needs
Windows, QGIS, a GPU, a network or a broken onnxruntime.
"""
import importlib.metadata as md

from plugin_utils import installer as inst
from plugin_utils.onnx_diagnostics import (is_native_load_failure,
                                           onnx_import_error_message,
                                           onnxruntime_distribution_present)

# The verbatim text the measured QGIS-3.28 failure raised.
DLL_ERROR = (
    "ImportError: DLL load failed while importing onnxruntime_pybind11_state: "
    "A dynamic link library (DLL) initialization routine failed.")


def _present(dist):
    """version_lookup stub: onnxruntime is installed, nothing else is."""
    if dist == "onnxruntime":
        return "1.27.0"
    raise md.PackageNotFoundError(dist)


def _absent(dist):
    raise md.PackageNotFoundError(dist)


# --- presence, read without importing onnxruntime --------------------------

def test_distribution_present_reads_metadata_not_the_module():
    assert onnxruntime_distribution_present(version_lookup=_present) is True
    assert onnxruntime_distribution_present(version_lookup=_absent) is False


def test_gpu_variant_counts_as_present():
    def _gpu(dist):
        if dist == "onnxruntime-gpu":
            return "1.27.0"
        raise md.PackageNotFoundError(dist)

    assert onnxruntime_distribution_present(version_lookup=_gpu) is True


# --- native-load-failure detection is Windows-only -------------------------

def test_dll_failure_is_a_native_failure_on_windows():
    assert is_native_load_failure(DLL_ERROR, platform="win32") is True


def test_the_same_text_is_not_a_native_failure_off_windows():
    assert is_native_load_failure(DLL_ERROR, platform="linux") is False
    assert is_native_load_failure(DLL_ERROR, platform="darwin") is False


def test_a_plain_module_not_found_is_not_a_native_failure():
    assert is_native_load_failure(
        "No module named 'onnxruntime'", platform="win32") is False


# --- the three-way message -------------------------------------------------

def test_absent_says_install_it():
    msg = onnx_import_error_message(
        "No module named 'onnxruntime'", platform="win32", present=False)
    assert "not installed" in msg
    assert "pip install onnxruntime" in msg


def test_installed_but_dll_shadowed_does_not_say_pip_install():
    msg = onnx_import_error_message(DLL_ERROR, platform="win32", present=True)
    assert "pip install" not in msg          # it IS installed
    assert "native library" in msg
    assert "shadow" in msg.lower()
    assert "3.28" in msg and "3.44" in msg   # names the observed cause
    # Visual C++ is only a secondary hint, never the asserted primary fix.
    assert "Visual C++" in msg
    assert not msg.lower().startswith("install the visual")


def test_installed_but_other_failure_is_generic_and_names_the_error():
    msg = onnx_import_error_message(
        "ImportError: libcudart.so.13", platform="linux", present=True)
    assert "pip install" not in msg
    assert "failed to load" in msg
    assert "libcudart.so.13" in msg


def test_present_defaults_to_the_metadata_check(monkeypatch):
    monkeypatch.setattr(
        "plugin_utils.onnx_diagnostics.onnxruntime_distribution_present",
        lambda: False)
    msg = onnx_import_error_message("boom", platform="win32")
    assert "not installed" in msg


# --- installer.import_verdict: the setup-time consumer ---------------------

def _report(ok=True, error=None, packages=("onnxruntime",), version="1.27.0",
            providers=("CPUExecutionProvider",)):
    return {"ok": ok, "error": error, "packages": list(packages),
            "version": version, "providers": list(providers),
            "session_providers": None}


def test_import_verdict_ok_reports_the_version_and_providers():
    ok, message = inst.import_verdict(_report())
    assert ok is True
    assert "1.27.0" in message
    assert "CPUExecutionProvider" in message


def test_import_verdict_flags_the_windows_dll_shadowing_honestly():
    ok, message = inst.import_verdict(
        _report(ok=False, error=DLL_ERROR, packages=["onnxruntime"]),
        platform="win32")
    assert ok is False
    assert "native library" in message
    assert "pip install" not in message


def test_import_verdict_says_install_when_nothing_is_there():
    ok, message = inst.import_verdict(
        _report(ok=False, error="No module named 'onnxruntime'",
                packages=[]),
        platform="win32")
    assert ok is False
    assert "not installed" in message
    assert "pip install onnxruntime" in message


def test_import_verdict_surfaces_any_other_import_error_verbatim():
    ok, message = inst.import_verdict(
        _report(ok=False, error="ImportError: libcudart.so.13",
                packages=["onnxruntime-gpu"]),
        platform="linux")
    assert ok is False
    assert "libcudart.so.13" in message


def test_verify_runtime_import_uses_the_probe_and_the_verdict(monkeypatch):
    """The setup-time check runs the probe (through child_env) and reports the
    verdict — without importing onnxruntime here."""
    captured = {}

    def _fake_probe(python_exe, model_path=None, want=None, timeout=300):
        captured["exe"] = python_exe
        return _report(ok=False, error=DLL_ERROR, packages=["onnxruntime"])

    monkeypatch.setattr(inst, "probe_runtime", _fake_probe)
    monkeypatch.setattr("plugin_utils.onnx_diagnostics.sys.platform", "win32")
    log = []
    result = inst.verify_runtime_import("/venv/py", progress=log.append)
    assert captured["exe"] == "/venv/py"
    assert result["ok"] is False
    assert "native library" in result["message"]
    assert any("native library" in line for line in log)
