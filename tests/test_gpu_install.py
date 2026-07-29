"""Opt-in GPU runtime install: requirements algebra, gpu_probe
verdicts, and the conflict-safe runtime swap. onnxruntime and
onnxruntime-gpu ship the SAME module; pip never uninstalls the other,
so the installer must — these tests pin that behavior.
"""
import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

installer = importlib.import_module("plugin_utils.installer")
gpu_probe = importlib.import_module("plugin_utils.gpu_probe")

from test_plugin_installer import _requirement_names  # noqa: E402


# --- requirements algebra ----------------------------------------------------

def test_gpu_txt_is_core_plus_one_gpu_runtime_and_psutil():
    text = (REPO / "requirements" / "gpu.txt").read_text()
    includes = [line.strip() for line in text.splitlines()
                if line.strip().startswith("-r")]
    assert includes == ["-r core.txt"]
    names = _requirement_names(text)
    assert names.count("onnxruntime-gpu") == 1
    assert "onnxruntime" not in names
    assert "psutil" in names


def test_gpu_runtime_pin_window_is_verbatim_and_unique():
    """Both ends are load-bearing: the floor kills pip's silent
    backtrack to a wheel with zero nvidia deps (CPU fallback that
    claims success), the ceiling kills the cu13 ImportError."""
    pin = "onnxruntime-gpu[cuda,cudnn]>=1.26,<1.27"
    hits = [p.name for p in sorted((REPO / "requirements").glob("*.txt"))
            if pin in p.read_text()]
    assert hits == ["gpu.txt"]


def test_no_requirements_file_names_both_runtimes():
    for path in sorted((REPO / "requirements").glob("*.txt")):
        names = _requirement_names(path.read_text())
        assert not ("onnxruntime" in names
                    and "onnxruntime-gpu" in names), (
            f"{path.name} names both runtimes")


# --- plugin_requirements_path matrix -----------------------------------------

def test_requirements_path_matrix():
    assert installer.plugin_requirements_path().name == "cpu.txt"
    assert installer.plugin_requirements_path(gpu=False).name == "cpu.txt"
    assert installer.plugin_requirements_path(gpu=True).name == "gpu.txt"


def test_requirements_path_gpu_falls_back_when_gpu_txt_missing(
        tmp_path, monkeypatch):
    (tmp_path / "cpu.txt").write_text("-r core.txt\nonnxruntime>=1.17\n")
    monkeypatch.setattr(installer, "repo_requirements_dir",
                        lambda: tmp_path)
    assert installer.plugin_requirements_path(gpu=True).name == "cpu.txt"


# --- gpu_probe ---------------------------------------------------------------

def _probe(runner):
    return gpu_probe.probe(system="Linux", machine="x86_64",
                           runner=runner)


def test_probe_ok_when_nvidia_smi_lists_a_gpu():
    out = "NVIDIA GeForce RTX 4080 SUPER, 580.65.06\n"
    result = _probe(lambda timeout: (None, out))
    assert result.status == gpu_probe.STATUS_OK
    assert result.present
    assert result.driver_version == "580.65.06"
    assert gpu_probe.wants_gpu_runtime(result)


def test_probe_no_driver_when_nvidia_smi_absent(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(gpu_probe.subprocess, "run", missing)
    result = _probe(None)  # falls through to the real runner
    assert result.status == gpu_probe.STATUS_NO_DRIVER
    assert not result.present
    assert not gpu_probe.wants_gpu_runtime(result)


def test_probe_no_driver_when_nvidia_smi_errors():
    result = _probe(lambda timeout: (gpu_probe.STATUS_NO_DRIVER, ""))
    assert result.status == gpu_probe.STATUS_NO_DRIVER
    assert not result.present


def test_probe_timeout_means_cpu_only():
    result = _probe(lambda timeout: (gpu_probe.STATUS_TIMEOUT, ""))
    assert result.status == gpu_probe.STATUS_TIMEOUT
    assert not result.present
    assert not gpu_probe.wants_gpu_runtime(result)


def test_probe_unsupported_platform_never_runs_nvidia_smi():
    def bomb(timeout):
        raise AssertionError("nvidia-smi must not run on macOS/ARM")
    result = gpu_probe.probe(system="Darwin", machine="arm64",
                             runner=bomb)
    assert result.status == gpu_probe.STATUS_UNSUPPORTED
    assert not result.present


def test_probe_old_driver_is_refused():
    result = _probe(lambda timeout: (None, "NVIDIA T400, 470.10\n"))
    assert result.status == gpu_probe.STATUS_OLD_DRIVER
    assert not result.present


# --- variant-aware sentinel --------------------------------------------------

def _patched_requirements(tmp_path, monkeypatch):
    reqs = tmp_path / "requirements"
    reqs.mkdir()
    (reqs / "cpu.txt").write_text("-r core.txt\nonnxruntime>=1.17\n")
    (reqs / "gpu.txt").write_text(
        "-r core.txt\nonnxruntime-gpu[cuda,cudnn]>=1.26,<1.27\n")
    monkeypatch.setattr(installer, "repo_requirements_dir",
                        lambda: reqs)


def test_marker_records_variant_and_gates_readiness(
        tmp_path, monkeypatch):
    _patched_requirements(tmp_path, monkeypatch)
    venv = tmp_path / "venv"
    venv.mkdir()
    installer._write_marker(str(venv), gpu=True)
    assert installer.installed_variant(str(venv)) == "gpu"
    assert installer.marker_matches(str(venv))          # own variant
    assert installer.marker_matches(str(venv), gpu=True)
    assert not installer.marker_matches(str(venv), gpu=False)
    installer._write_marker(str(venv), gpu=False)
    assert installer.installed_variant(str(venv)) == "cpu"
    assert installer.marker_matches(str(venv), gpu=False)
    assert not installer.marker_matches(str(venv), gpu=True)


def test_legacy_marker_without_variant_counts_as_cpu(
        tmp_path, monkeypatch):
    _patched_requirements(tmp_path, monkeypatch)
    venv = tmp_path / "venv"
    venv.mkdir()
    req = installer.plugin_requirements_path()
    marker = {"req_hash": installer._file_hash(req),
              "requirements": str(req)}
    (venv / installer.READY_MARKER).write_text(json.dumps(marker))
    assert installer.installed_variant(str(venv)) == "cpu"
    assert installer.marker_matches(str(venv))
    assert not installer.marker_matches(str(venv), gpu=True)


def test_installed_variant_none_without_marker(tmp_path):
    assert installer.installed_variant(str(tmp_path)) is None


# --- conflict-safe runtime swap ----------------------------------------------

def test_uninstall_removes_exactly_the_conflicting_dist(monkeypatch):
    calls = []
    monkeypatch.setattr(installer, "_run_streamed",
                        lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(installer, "distribution_installed",
                        lambda py, dist, **kw: True)
    assert installer.uninstall_conflicting_runtime("py", gpu=True)
    assert calls[-1][-4:] == ["pip", "uninstall", "-y", "onnxruntime"]
    assert installer.uninstall_conflicting_runtime("py", gpu=False)
    assert calls[-1][-4:] == ["pip", "uninstall", "-y",
                              "onnxruntime-gpu"]


def test_uninstall_is_a_noop_when_nothing_conflicts(monkeypatch):
    calls = []
    monkeypatch.setattr(installer, "_run_streamed",
                        lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(installer, "distribution_installed",
                        lambda py, dist, **kw: False)
    assert not installer.uninstall_conflicting_runtime("py", gpu=True)
    assert calls == []


def test_install_requirements_swaps_runtime_before_pip_install(
        tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(
        installer, "uninstall_conflicting_runtime",
        lambda py, gpu, progress=None: order.append(("swap", gpu)))
    monkeypatch.setattr(installer, "_run_streamed",
                        lambda cmd, **kw: order.append(("pip", cmd[-1])))
    installer.install_requirements(str(tmp_path), gpu=True)
    assert order[0] == ("swap", True)
    assert order[1][0] == "pip" and order[1][1].endswith("gpu.txt")
    order.clear()
    installer.install_requirements(str(tmp_path), gpu=False)
    assert order[0] == ("swap", False)
    assert order[1][1].endswith("cpu.txt")


# --- opt-in wiring (WINMOL_GPU=1) --------------------------------------------

def test_gpu_requested_reads_winmol_gpu(monkeypatch):
    monkeypatch.delenv("WINMOL_GPU", raising=False)
    assert not installer.gpu_requested()
    for yes in ("1", "true", "YES", "on"):
        monkeypatch.setenv("WINMOL_GPU", yes)
        assert installer.gpu_requested()
    monkeypatch.setenv("WINMOL_GPU", "0")
    assert not installer.gpu_requested()


def test_setup_environment_threads_gpu_to_install_and_marker(
        tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(installer, "is_ready",
                        lambda venv, gpu=None: False)
    monkeypatch.setattr(installer, "create_venv", lambda *a, **k: None)
    monkeypatch.setattr(installer, "ensure_pip", lambda *a, **k: None)
    monkeypatch.setattr(
        installer, "install_requirements",
        lambda venv, progress=None, gpu=False: seen.update(install=gpu))
    monkeypatch.setattr(
        installer, "_write_marker",
        lambda venv, gpu=False: seen.update(marker=gpu))
    installer.setup_environment(str(tmp_path / "v"), gpu=True)
    assert seen == {"install": True, "marker": True}


def test_resolve_environment_rebuilds_cpu_venv_when_gpu_requested(
        tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "configured_python_executable",
                        lambda: None)
    # a venv that is ready as-is (gpu=None) but is NOT the gpu variant
    monkeypatch.setattr(installer, "is_ready",
                        lambda venv, gpu=None: gpu is None)
    monkeypatch.delenv("WINMOL_GPU", raising=False)
    assert installer.resolve_environment(
        str(tmp_path), build=False)["status"] == "ready"
    monkeypatch.setenv("WINMOL_GPU", "1")
    assert installer.resolve_environment(
        str(tmp_path), build=False)["status"] == "needs_setup"
