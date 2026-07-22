"""The GPU-runtime decision, tested on a machine that has no GPU.

The bug this covers, reproduced on a Lenovo/RTX 4080 SUPER box: the QGIS
plugin's venv installed plain ``onnxruntime`` (requirements/plugin.txt), so
``get_available_providers()`` returned ``['AzureExecutionProvider',
'CPUExecutionProvider']`` — CUDAExecutionProvider was not a fallback, it was
absent from the build. Inference ran at 5265.7 ms per tile instead of the
12.4 ms the same box does on CUDA.

Every decision that fixes it is a pure function over a stubbed nvidia-smi
and a stubbed provider list, so nothing here needs a GPU, a network, a pip
install or a live QGIS.
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plugin_utils import gpu_probe as gp          # noqa: E402
from plugin_utils import installer as inst        # noqa: E402
from plugin_utils import setup_state as ss        # noqa: E402

# What nvidia-smi prints on the machine the bug was measured on.
SMI_4080 = "NVIDIA GeForce RTX 4080 SUPER, 580.159.03\n"
SMI_OLD_DRIVER = "NVIDIA GeForce GTX 1080, 470.199.02\n"

# Providers, as measured. The first list is the bug.
PROVIDERS_CPU_BUILD = ["AzureExecutionProvider", "CPUExecutionProvider"]
PROVIDERS_GPU_BUILD = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                       "CPUExecutionProvider"]
PROVIDERS_COREML = ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def _runner(stdout="", status=None):
    """A stand-in for installer's nvidia-smi call."""
    def _run(_timeout):
        return status, stdout
    return _run


def _report(providers, packages=("onnxruntime",), ok=True, error=None,
            session=None, version="1.27.0"):
    return {"ok": ok, "providers": list(providers),
            "packages": list(packages), "version": version,
            "error": error, "session_providers": session}


# --- detection: what nvidia-smi says, and how long it may take -------------

def test_a_gpu_with_a_current_driver_is_detected():
    probe = gp.probe(system="Linux", machine="x86_64",
                     runner=_runner(SMI_4080))
    assert probe.status == gp.STATUS_OK
    assert probe.present is True
    assert probe.driver_version == "580.159.03"
    assert probe.label == "NVIDIA GeForce RTX 4080 SUPER"


def test_no_gpu_when_nvidia_smi_lists_none():
    probe = gp.probe(system="Linux", machine="x86_64", runner=_runner(""))
    assert probe.status == gp.STATUS_NONE
    assert probe.present is False
    assert probe.has_hardware is False


def test_nvidia_smi_missing_is_not_an_error():
    """The common case: most machines simply have no NVIDIA driver."""
    probe = gp.probe(system="Linux", machine="x86_64",
                     runner=_runner(status=gp.STATUS_NO_DRIVER))
    assert probe.status == gp.STATUS_NO_DRIVER
    assert probe.present is False


def test_a_wedged_nvidia_smi_times_out_instead_of_hanging():
    probe = gp.probe(system="Linux", machine="x86_64", timeout=8.0,
                     runner=_runner(status=gp.STATUS_TIMEOUT))
    assert probe.status == gp.STATUS_TIMEOUT
    assert probe.present is False
    assert "8s" in probe.detail


def test_nvidia_smi_is_never_run_without_a_timeout():
    """A driver stuck in an ioctl blocks the call for as long as the kernel
    module takes to give up. Both call sites must bound it."""
    import inspect
    import classes.HardwareInfo as hw
    for func in (gp._run_nvidia_smi,
                 hw.HardwareInfo._detect_gpu_names_nvidia_smi,
                 hw.HardwareInfo._detect_gpu_memory_gb_nvidia_smi):
        assert "timeout" in inspect.getsource(func), \
            f"{func.__name__} runs nvidia-smi without a timeout"


@pytest.mark.parametrize("system,machine", [
    ("Darwin", "arm64"),       # macOS: no NVIDIA driver exists at all
    ("Darwin", "x86_64"),
    ("Linux", "aarch64"),      # ARM: no onnxruntime-gpu wheels
    ("Windows", "arm64"),
])
def test_platforms_without_wheels_are_refused_before_nvidia_smi(system,
                                                                machine):
    def _explode(_timeout):                        # pragma: no cover
        raise AssertionError("nvidia-smi must not be run here")

    probe = gp.probe(system=system, machine=machine, runner=_explode)
    assert probe.status == gp.STATUS_UNSUPPORTED
    assert probe.present is False


def test_an_old_driver_is_reported_rather_than_installed_for():
    probe = gp.probe(system="Linux", machine="x86_64",
                     runner=_runner(SMI_OLD_DRIVER))
    assert probe.status == gp.STATUS_OLD_DRIVER
    assert probe.has_hardware is True      # the GPU is real...
    assert probe.present is False          # ...but the wheels cannot serve it
    assert "470.199.02" in probe.detail


def test_an_unreadable_driver_version_does_not_veto_the_install():
    probe = gp.probe(system="Linux", machine="x86_64",
                     runner=_runner("NVIDIA GeForce RTX 4080 SUPER, N/A\n"))
    assert probe.present is True


# --- which requirements file gets installed --------------------------------

@pytest.mark.parametrize("stdout,status,system,machine,expected", [
    (SMI_4080, None, "Linux", "x86_64", "plugin-gpu.txt"),
    (SMI_4080, None, "Windows", "amd64", "plugin-gpu.txt"),
    ("", None, "Linux", "x86_64", "plugin.txt"),            # no GPU
    ("", gp.STATUS_NO_DRIVER, "Linux", "x86_64", "plugin.txt"),
    ("", gp.STATUS_TIMEOUT, "Linux", "x86_64", "plugin.txt"),
    (SMI_4080, None, "Darwin", "arm64", "plugin.txt"),      # macOS
    (SMI_4080, None, "Linux", "aarch64", "plugin.txt"),     # ARM
    (SMI_OLD_DRIVER, None, "Linux", "x86_64", "plugin.txt"),
])
def test_requirements_choice(stdout, status, system, machine, expected):
    probe = gp.probe(system=system, machine=machine,
                     runner=_runner(stdout, status))
    assert ss.requirements_choice(probe) == expected


def test_the_installer_resolves_both_requirements_files():
    assert inst.plugin_requirements_path().name == "plugin.txt"
    assert inst.plugin_requirements_path(gpu=True).name == "plugin-gpu.txt"


def test_the_gpu_requirements_file_never_pulls_the_cpu_runtime():
    """onnxruntime and onnxruntime-gpu both provide the `onnxruntime`
    module and MUST NOT be co-installed (requirements/gpu.txt says so).
    plugin-gpu.txt must therefore not reach base.txt, which pins the CPU
    build."""
    names = inst._requirement_names(inst.plugin_requirements_path(gpu=True))
    assert "onnxruntime-gpu" in names
    assert "onnxruntime" not in names


def test_the_cpu_requirements_file_never_pulls_the_gpu_runtime():
    names = inst._requirement_names(inst.plugin_requirements_path())
    assert "onnxruntime" in names
    assert "onnxruntime-gpu" not in names


def test_a_gpu_marker_is_not_read_as_a_stale_cpu_marker(tmp_path,
                                                        monkeypatch):
    """A GPU environment is installed from plugin-gpu.txt, so its sentinel
    carries that file's hash. Comparing it only against plugin.txt would
    call every GPU install 'incomplete' forever."""
    venv = tmp_path / "winmol_venv"
    venv.mkdir()
    inst._write_marker(str(venv), gpu=True)
    assert inst.marker_variant(str(venv)) == "gpu"
    assert inst.marker_matches(str(venv)) is True

    inst._write_marker(str(venv), gpu=False)
    assert inst.marker_variant(str(venv)) == "cpu"


def test_an_unknown_marker_still_invalidates(tmp_path, monkeypatch):
    venv = tmp_path / "winmol_venv"
    venv.mkdir()
    inst._write_marker(str(venv))
    monkeypatch.setattr(inst, "_variant_hashes",
                        lambda: {"cpu": "x", "gpu": "y"})
    assert inst.marker_matches(str(venv)) is False


# --- the conflicting-runtime rule ------------------------------------------

def test_the_other_runtime_is_uninstalled_before_the_new_one_lands(
        monkeypatch):
    calls = []
    monkeypatch.setattr(inst, "distribution_installed",
                        lambda exe, dist, **kw: dist == "onnxruntime")
    monkeypatch.setattr(inst, "_run_streamed",
                        lambda cmd, **kw: calls.append(cmd))
    removed = inst.uninstall_conflicting_runtime("/fake/python", gpu=True)
    assert removed is True
    assert calls == [["/fake/python", "-u", "-m", "pip", "uninstall", "-y",
                      "onnxruntime"]]


def test_nothing_is_uninstalled_when_there_is_no_conflict(monkeypatch):
    monkeypatch.setattr(inst, "distribution_installed",
                        lambda exe, dist, **kw: False)
    monkeypatch.setattr(inst, "_run_streamed", lambda cmd, **kw: pytest.fail(
        "pip uninstall ran with nothing to uninstall"))
    assert inst.uninstall_conflicting_runtime("/fake/python",
                                              gpu=True) is False


def test_a_failed_uninstall_is_logged_not_raised(monkeypatch):
    def _boom(cmd, **kw):
        raise RuntimeError("pip exploded")

    monkeypatch.setattr(inst, "distribution_installed",
                        lambda exe, dist, **kw: True)
    monkeypatch.setattr(inst, "_run_streamed", _boom)
    seen = []
    assert inst.uninstall_conflicting_runtime(
        "/fake/python", gpu=False, progress=seen.append) is False
    assert any("pip exploded" in line for line in seen)


# --- the post-install proof ------------------------------------------------

def test_a_gpu_install_that_delivers_cuda_is_confirmed():
    ok, message = inst.gpu_verdict(
        _report(PROVIDERS_GPU_BUILD, packages=["onnxruntime-gpu"],
                session=["CUDAExecutionProvider", "CPUExecutionProvider"],
                version="1.26.0"),
        checked_with_model=True)
    assert ok is True
    assert "CUDAExecutionProvider" in message


def test_a_gpu_install_without_the_provider_is_called_out():
    """Never claim GPU and deliver CPU."""
    ok, message = inst.gpu_verdict(
        _report(PROVIDERS_CPU_BUILD, packages=["onnxruntime-gpu"]))
    assert ok is False
    assert "NOT available" in message
    assert "CPU" in message


def test_an_unimportable_runtime_reports_the_real_reason():
    """onnxruntime-gpu 1.27's extras moved to CUDA 13; on a cu12 machine
    the import dies with libcudart.so.13, and that string is the whole
    diagnosis."""
    ok, message = inst.gpu_verdict(
        _report([], packages=["onnxruntime-gpu"], ok=False,
                error="ImportError: libcudart.so.13"))
    assert ok is False
    assert "libcudart.so.13" in message


def test_a_session_that_falls_back_to_cpu_is_a_failure():
    ok, message = inst.gpu_verdict(
        _report(PROVIDERS_GPU_BUILD, packages=["onnxruntime-gpu"],
                session=["CPUExecutionProvider"]),
        checked_with_model=True)
    assert ok is False
    assert "fell back" in message


def test_no_model_on_disk_is_reported_as_unproven_not_as_proof():
    ok, message = inst.gpu_verdict(
        _report(PROVIDERS_GPU_BUILD, packages=["onnxruntime-gpu"]))
    assert ok is True
    assert "no session was created" in message.lower()


def test_probe_runtime_survives_an_interpreter_that_is_not_there():
    report = inst.probe_runtime("/definitely/not/a/python", timeout=5)
    assert report["ok"] is False
    assert report["providers"] == []


def test_probe_runtime_reads_the_last_json_line(monkeypatch):
    class _Out:
        stdout = ('some pip warning\n'
                  '{"ok": true, "providers": ["CUDAExecutionProvider"], '
                  '"packages": ["onnxruntime-gpu"], "version": "1.26.0", '
                  '"error": null, "session_providers": null}\n')
        stderr = ""

    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: _Out())
    report = inst.probe_runtime("/fake/python")
    assert report["providers"] == ["CUDAExecutionProvider"]
    assert report["version"] == "1.26.0"


# --- the Setup tab's three states ------------------------------------------

def _probe(stdout=SMI_4080, status=None, system="Linux", machine="x86_64"):
    return gp.probe(system=system, machine=machine,
                    runner=_runner(stdout, status))


def test_state_i_no_gpu_says_so_and_offers_nothing():
    status = ss.accelerator_status(_probe(""), _report(PROVIDERS_CPU_BUILD))
    assert status.state == ss.ACCEL_CPU_ONLY
    assert status.can_install is False
    assert "5 s" in status.text and "12 ms" in status.text


def test_state_ii_gpu_present_but_cpu_runtime_warns_and_offers_the_fix():
    """The user's actual state when the bug was measured."""
    status = ss.accelerator_status(_probe(), _report(PROVIDERS_CPU_BUILD))
    assert status.state == ss.ACCEL_GPU_IDLE
    assert status.can_install is True
    assert "RTX 4080 SUPER" in status.text
    assert "12 ms" in status.text


def test_state_iii_cuda_active_is_confirmed_positively():
    status = ss.accelerator_status(
        _probe(), _report(PROVIDERS_GPU_BUILD, packages=["onnxruntime-gpu"]))
    assert status.state == ss.ACCEL_GPU_ACTIVE
    assert status.can_install is False
    assert "RTX 4080 SUPER" in status.text


def test_a_gpu_package_that_cannot_reach_cuda_is_not_offered_again():
    status = ss.accelerator_status(
        _probe(), _report(PROVIDERS_CPU_BUILD, packages=["onnxruntime-gpu"],
                          error="ImportError: libcudart.so.13"))
    assert status.state == ss.ACCEL_GPU_BROKEN
    assert status.can_install is False
    assert "libcudart.so.13" in status.text


def test_a_gpu_we_cannot_serve_is_named_not_hidden():
    """An out-of-date driver is a GPU we can SEE and must not install for.
    Saying "no GPU" there would send the user hunting for hardware they
    already own."""
    status = ss.accelerator_status(_probe(stdout=SMI_OLD_DRIVER),
                                   _report(PROVIDERS_CPU_BUILD))
    assert status.state == ss.ACCEL_GPU_UNUSABLE
    assert status.can_install is False
    assert "GTX 1080" in status.text
    assert "470.199.02" in status.text


def test_arm_never_offers_cuda_and_never_claims_a_gpu():
    """No onnxruntime-gpu wheels exist for ARM, so nvidia-smi is not even
    consulted and the honest answer is 'no usable GPU'."""
    status = ss.accelerator_status(_probe(system="Linux", machine="aarch64"),
                                   _report(PROVIDERS_CPU_BUILD))
    assert status.state == ss.ACCEL_CPU_ONLY
    assert status.can_install is False
    assert "usable" in status.text


def test_macos_keeps_its_coreml_answer_and_is_never_offered_cuda():
    """Stock onnxruntime ships CoreML on macOS and it is already selected;
    calling that 'CPU only' would send the user shopping for a GPU."""
    status = ss.accelerator_status(_probe(system="Darwin", machine="arm64"),
                                   _report(PROVIDERS_COREML))
    assert status.state == ss.ACCEL_COREML
    assert status.can_install is False


@pytest.mark.parametrize("status_probe,report", [
    (None, None),
    (_probe(), None),
])
def test_nothing_is_claimed_before_the_probe_lands(status_probe, report):
    status = ss.accelerator_status(status_probe, report)
    assert status.state == ss.ACCEL_UNKNOWN
    assert status.can_install is False


def test_timed_out_detection_reads_as_cpu_only_never_as_gpu():
    status = ss.accelerator_status(_probe(status=gp.STATUS_TIMEOUT),
                                   _report(PROVIDERS_CPU_BUILD))
    assert status.state == ss.ACCEL_CPU_ONLY
    assert status.can_install is False


# --- the offer, and the button that carries it -----------------------------

def test_the_offer_names_the_gpu_the_size_and_the_measured_speedup():
    text = ss.gpu_offer_text(_probe())
    assert "RTX 4080 SUPER" in text
    assert "2.4 GB" in text
    assert "12 ms" in text and "5.3 s" in text


def _info(**kw):
    kw.setdefault("exe", "/env/bin/python")
    kw.setdefault("exists", True)
    kw.setdefault("version", (3, 11))
    kw.setdefault("deps_ok", True)
    kw.setdefault("managed", True)
    kw.setdefault("venv_path", "/env")
    kw.setdefault("runtime_path", "/rt")
    kw.setdefault("marker_ok", True)
    return ss.EnvInfo(**kw)


def test_the_gpu_button_is_live_only_in_the_offerable_state():
    idle = ss.accelerator_status(_probe(), _report(PROVIDERS_CPU_BUILD))
    active = ss.accelerator_status(
        _probe(), _report(PROVIDERS_GPU_BUILD, packages=["onnxruntime-gpu"]))
    assert ss.button_states(_info(), [], accel=idle)["env_gpu_button"]
    assert not ss.button_states(_info(), [], accel=active)["env_gpu_button"]
    # ...and dead until the probe has answered at all
    assert not ss.button_states(_info(), [])["env_gpu_button"]


def test_the_gpu_button_obeys_the_one_job_at_a_time_interlock():
    idle = ss.accelerator_status(_probe(), _report(PROVIDERS_CPU_BUILD))
    states = ss.button_states(_info(), [], accel=idle, env_busy=True)
    assert states["env_gpu_button"] is False


def test_the_gpu_button_is_dead_without_a_usable_environment():
    """There is nothing to install onnxruntime-gpu INTO yet."""
    idle = ss.accelerator_status(_probe(), _report(PROVIDERS_CPU_BUILD))
    info = _info(exe=None, exists=False, version=None, deps_ok=None,
                 managed=False, marker_ok=False)
    assert ss.button_states(info, [], accel=idle)["env_gpu_button"] is False


def test_a_machine_with_no_environment_yet_is_still_offered_the_gpu_build():
    """Otherwise an NVIDIA box downloads the whole environment twice: once
    as CPU-only, then again after the Setup tab points out the GPU."""
    status = ss.accelerator_from_machine(
        None, detect=lambda: _probe(),
        probe_fn=lambda *a, **k: pytest.fail("no interpreter to probe"))
    assert status.state == ss.ACCEL_GPU_IDLE
    assert status.can_install is True
    assert "Create the environment" in status.text


def test_accelerator_from_machine_uses_the_child_interpreter():
    seen = []
    status = ss.accelerator_from_machine(
        "/env/bin/python", detect=lambda: _probe(),
        probe_fn=lambda exe, **k: seen.append(exe) or _report(
            PROVIDERS_GPU_BUILD, packages=["onnxruntime-gpu"]))
    assert seen == ["/env/bin/python"]
    assert status.state == ss.ACCEL_GPU_ACTIVE


# --- the dialog's wiring, pinned statically (no QGIS in CI) ----------------

def _dialog_function(name):
    import ast
    src = open(os.path.join(REPO, "winmol_analyzer_dialog.py")).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined on the dialog")


def _called_names(node):
    import ast
    return {getattr(call.func, "attr", None) or getattr(call.func, "id", None)
            for call in ast.walk(node) if isinstance(call, ast.Call)}


def test_creating_the_environment_asks_before_downloading_2_4_gb():
    assert "_confirm_gpu_runtime" in _called_names(
        _dialog_function("_setup_create_env"))


def test_the_gpu_install_slot_confirms_and_never_probes_inline():
    node = _dialog_function("_setup_install_gpu_runtime")
    called = _called_names(node)
    assert "_confirm_gpu_runtime" in called
    assert "invalidate_marker" in called, \
        "setup_environment short-circuits on a valid sentinel"
    for banned in ("detect_gpu", "probe_runtime", "run", "Popen"):
        assert banned not in called, f"{banned} runs on the GUI thread"


def test_reinstalling_dependencies_keeps_a_gpu_environment_on_the_gpu():
    """A repair that silently reinstalled the CPU runtime over a working
    CUDA one would undo a 2.4 GB download and read as a random slowdown."""
    import ast
    node = _dialog_function("_setup_repair_env")
    called = _called_names(node)
    assert "marker_variant" in called
    body = ast.dump(node)
    assert body.index("marker_variant") < body.index("invalidate_marker"), \
        "the variant must be read before the sentinel is dropped"
    assert any(kw.arg == "gpu"
               for call in ast.walk(node) if isinstance(call, ast.Call)
               for kw in call.keywords), "the repair drops the gpu flag"
