"""Unit tests for the child-process environment sanitizer.

QGIS exports PYTHONHOME/PYTHONPATH pointing at its own interpreter, and
GDAL_DATA/PROJ_LIB pointing at its own geo data. Both are poison for the
child processes WINMOL spawns, which run a *different* Python with its own
vendored GDAL. See docs: the Windows "could not import runpy module" crash.
"""
import os

import pytest

from plugin_utils.childenv import child_env


@pytest.fixture
def polluted(monkeypatch):
    """os.environ as QGIS leaves it."""
    monkeypatch.setenv("PYTHONHOME", r"C:\PROGRA~1\QGIS34~1.12\apps\qgis-ltr")
    monkeypatch.setenv("PYTHONPATH",
                       r"C:\PROGRA~1\QGIS34~1.12\apps\qgis-ltr\python")
    monkeypatch.setenv("PYTHONSTARTUP", "/tmp/startup.py")
    monkeypatch.setenv("PYTHONEXECUTABLE", "/qgis/python.exe")
    monkeypatch.setenv("__PYVENV_LAUNCHER__", "/qgis/python.exe")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/other/venv")
    monkeypatch.setenv("GDAL_DATA", "/qgis/share/gdal")
    monkeypatch.setenv("PROJ_LIB", "/qgis/share/proj")
    monkeypatch.setenv("PROJ_DATA", "/qgis/share/proj")


def test_strips_interpreter_vars_that_break_a_foreign_python(polluted):
    env = child_env()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP",
                "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__", "VIRTUAL_ENV"):
        assert var not in env, f"{var} leaked into the child environment"


def test_strips_geo_data_vars_that_break_a_vendored_gdal(polluted):
    env = child_env()
    for var in ("GDAL_DATA", "PROJ_LIB", "PROJ_DATA"):
        assert var not in env, f"{var} leaked into the child environment"


def test_preserves_vars_the_child_needs(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("WINMOL_ONNX_FORCE_CPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = child_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert env["WINMOL_ONNX_FORCE_CPU"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_pins_hash_seed_so_stem_counts_stay_deterministic(polluted):
    # Same guarantee winmol_run.py's re-exec guard provides (PR #4).
    assert child_env()["PYTHONHASHSEED"] == "0"


def test_disables_user_site_packages(polluted):
    assert child_env()["PYTHONNOUSERSITE"] == "1"


def test_extra_is_applied_after_stripping(polluted):
    # tests/test_plugin_compute_contract.py needs to inject a PYTHONPATH
    # deliberately; an explicit extra must win over the strip list.
    env = child_env(extra={"PYTHONPATH": "/block/tf"})
    assert env["PYTHONPATH"] == "/block/tf"


def test_does_not_mutate_the_parent_environment(polluted):
    child_env()
    assert "PYTHONHOME" in os.environ, "child_env mutated os.environ"


# --- the CUDA/cuDNN loader path ------------------------------------------
#
# onnxruntime-gpu does not bundle CUDA: it depends on the nvidia-*-cu12
# wheels, which unpack into <venv>/lib/python3.X/site-packages/nvidia/*/lib.
# Nothing puts those seven directories on the loader path, so on the
# measured RTX 4080 SUPER machine a correctly installed GPU runtime listed
# CUDAExecutionProvider and then ran every session on the CPU.

from plugin_utils.childenv import (_loader_path_var,  # noqa: E402
                                   native_lib_extra, nvidia_lib_dirs)

NVIDIA_COMPONENTS = ("cublas", "cuda_runtime", "cudnn", "cufft", "curand",
                     "cusolver", "cusparse")


@pytest.fixture
def gpu_venv(tmp_path):
    """A venv laid out the way pip leaves one after onnxruntime-gpu."""
    site = tmp_path / "lib" / "python3.11" / "site-packages"
    for name in NVIDIA_COMPONENTS:
        (site / "nvidia" / name / "lib").mkdir(parents=True)
    exe = tmp_path / "bin" / "python"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("")
    return str(exe)


def test_finds_every_nvidia_wheel_library_directory(gpu_venv):
    dirs = nvidia_lib_dirs(gpu_venv)
    assert len(dirs) == len(NVIDIA_COMPONENTS)
    assert all(os.path.isdir(d) for d in dirs)
    assert any(d.endswith(os.path.join("cudnn", "lib")) for d in dirs)


def test_a_cpu_only_environment_gets_no_loader_path_at_all(tmp_path):
    """Nothing to add on a CPU wheel or on macOS — so nothing is added."""
    exe = tmp_path / "bin" / "python"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert nvidia_lib_dirs(str(exe)) == []
    assert native_lib_extra(str(exe)) == {}
    assert nvidia_lib_dirs(None) == []


def test_the_users_existing_loader_path_is_prepended_to_never_replaced(
        gpu_venv, monkeypatch):
    var = _loader_path_var()
    monkeypatch.setenv(var, "/opt/mine/lib")
    value = native_lib_extra(gpu_venv)[var]
    assert value.endswith(os.pathsep + "/opt/mine/lib")
    assert value.split(os.pathsep)[0].endswith("lib")
    assert "/opt/mine/lib" in value.split(os.pathsep)


def test_child_env_puts_the_cuda_libraries_on_the_loader_path(gpu_venv):
    var = _loader_path_var()
    env = child_env(python_exe=gpu_venv)
    assert "nvidia" in env[var]
    # ...and only when asked about that interpreter.
    assert "nvidia" not in child_env().get(var, "")


def test_extra_still_wins_over_the_loader_path(gpu_venv):
    var = _loader_path_var()
    env = child_env({var: "/explicit"}, python_exe=gpu_venv)
    assert env[var] == "/explicit"
