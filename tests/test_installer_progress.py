"""Unit tests for the installer's progress plumbing.

The QGIS plugin builds its compute environment (venv + pip + model downloads)
in a background thread and streams status lines into the same log panel the
prediction run uses. These tests exercise that plumbing with a stubbed
subprocess: no venv is created, no network is touched.
"""

import os
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pytest                            # noqa: E402
import plugin_utils.installer as inst    # noqa: E402


# --- stubs ------------------------------------------------------------------

class _Stream:
    """A stdout stand-in: iterable of lines, closeable."""

    def __init__(self, lines, gate=None):
        self._lines = list(lines)
        self._gate = gate
        self.closed = False

    def __iter__(self):
        if self._gate is not None:
            self._gate.wait(5.0)
        for line in self._lines:
            yield line

    def close(self):
        self.closed = True


class FakePopen:
    """Minimal subprocess.Popen replacement driven by canned output."""

    calls = []

    def __init__(self, lines=(), rc=0, gate=None):
        self.stdout = _Stream(lines, gate)
        self._rc = rc
        self.killed = False

    @classmethod
    def factory(cls, lines=(), rc=0, gate=None):
        def _make(cmd, **kwargs):
            cls.calls.append((cmd, kwargs))
            return cls(lines, rc, gate)
        return _make

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return self._rc

    def kill(self):
        self.killed = True
        self._rc = -9


@pytest.fixture(autouse=True)
def _reset_calls():
    FakePopen.calls = []
    yield
    FakePopen.calls = []


def _patch_popen(monkeypatch, lines=(), rc=0, gate=None):
    monkeypatch.setattr(inst.subprocess, "Popen",
                        FakePopen.factory(lines, rc, gate))


PIP_LINES = [
    "Collecting rasterio>=1.3\n",
    "  Downloading rasterio-1.3.9-cp311.whl (21 MB)\n",
    "Collecting geopandas\n",
    "  Downloading geopandas-0.14.3-py3-none-any.whl (1.1 MB)\n",
    "Installing collected packages: rasterio, geopandas\n",
    "Successfully installed geopandas-0.14.3 rasterio-1.3.9\n",
]


# --- streaming --------------------------------------------------------------

def test_install_requirements_streams_each_line(monkeypatch):
    _patch_popen(monkeypatch, PIP_LINES)
    seen = []
    inst.install_requirements_into("/fake/python", progress=seen.append)

    body = "\n".join(seen)
    assert "rasterio" in body and "geopandas" in body
    assert "Downloading rasterio-1.3.9-cp311.whl (21 MB)" in body
    assert "Installing collected packages" in body
    # every pip line is surfaced, in order
    idx_r = body.index("rasterio-1.3.9-cp311.whl")
    idx_g = body.index("geopandas-0.14.3-py3-none-any.whl")
    assert idx_r < idx_g


def test_pip_is_invoked_unbuffered_and_noninteractive(monkeypatch):
    _patch_popen(monkeypatch, PIP_LINES)
    inst.install_requirements_into("/fake/python", progress=None)

    cmd, kwargs = FakePopen.calls[-1]
    assert cmd[:2] == ["/fake/python", "-u"]
    assert "--no-input" in cmd
    assert cmd[cmd.index("--progress-bar") + 1] == "off"
    # merged streams: pip writes most failure detail to stdout
    assert kwargs["stderr"] is inst.subprocess.STDOUT
    # PYTHONUNBUFFERED must survive child_env()'s PYTHON* stripping
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"


def test_progress_messages_carry_elapsed_time(monkeypatch):
    ticks = iter([0.0, 1.0, 2.0, 7.0, 42.0] + [99.0] * 200)
    monkeypatch.setattr(inst.time, "monotonic", lambda: next(ticks))
    seen = []
    p = inst._Progress(seen.append)
    p("one")
    p("two")
    assert seen[0].startswith("[") and "s]" in seen[0]
    assert "one" in seen[0] and "two" in seen[1]
    first = int(seen[0].split("s]")[0].strip("[ "))
    second = int(seen[1].split("s]")[0].strip("[ "))
    assert second > first


def test_progress_counts_packages(monkeypatch):
    _patch_popen(monkeypatch, PIP_LINES)
    seen = []
    monkeypatch.setattr(inst, "_requirement_names",
                        lambda path: ["rasterio", "geopandas", "psutil"])
    inst.install_requirements_into("/fake/python", progress=seen.append)
    body = "\n".join(seen)
    assert "Installing rasterio>=1.3 (1 of 3)" in body
    assert "Installing geopandas (2 of 3)" in body


# --- heartbeat --------------------------------------------------------------

def test_heartbeat_fires_when_child_is_quiet(monkeypatch):
    gate = threading.Event()
    _patch_popen(monkeypatch, ["done\n"], gate=gate)
    seen = []

    def _release():
        time.sleep(0.35)
        gate.set()

    t = threading.Thread(target=_release, daemon=True)
    t.start()
    inst._run_streamed(["/fake/python", "-m", "pip"], progress=seen.append,
                       label="pip install", timeout=60, heartbeat=0.1)
    t.join(5)
    body = "\n".join(seen)
    assert "still working" in body
    assert "done" in body


def test_streamed_run_kills_child_on_timeout(monkeypatch):
    gate = threading.Event()   # never set: the child never speaks
    _patch_popen(monkeypatch, ["late\n"], gate=gate)
    with pytest.raises(RuntimeError) as err:
        inst._run_streamed(["/fake/python"], label="pip install",
                           timeout=0.2, heartbeat=0.05)
    gate.set()
    assert "timed out" in str(err.value)


# --- failures ---------------------------------------------------------------

def test_nonzero_exit_raises_with_real_output_tail(monkeypatch):
    lines = [
        "Collecting onnxruntime\n",
        "ERROR: Could not find a version that satisfies onnxruntime==9.9\n",
    ]
    _patch_popen(monkeypatch, lines, rc=1)
    seen = []
    with pytest.raises(RuntimeError) as err:
        inst.install_requirements_into("/fake/python", progress=seen.append)
    msg = str(err.value)
    assert "Could not find a version that satisfies onnxruntime==9.9" in msg
    assert "exit 1" in msg
    # and the tail also reaches the log panel, not just the exception
    assert any("Could not find a version" in m for m in seen)


def test_create_venv_failure_keeps_actionable_text(monkeypatch):
    _patch_popen(monkeypatch, ["Error: no ensurepip\n"], rc=1)
    with pytest.raises(RuntimeError) as err:
        inst.create_venv("/tmp/nope", base_python="/fake/py3.11")
    msg = str(err.value)
    assert "venv creation failed" in msg
    assert "/fake/py3.11" in msg
    assert "no ensurepip" in msg


# --- phases -----------------------------------------------------------------

def test_create_venv_announces_phase(monkeypatch):
    _patch_popen(monkeypatch, ["created\n"])
    seen = []
    inst.create_venv("/tmp/venv", base_python="/fake/py3.11",
                     progress=seen.append)
    assert any("virtual environment" in m for m in seen)
    assert any("/fake/py3.11" in m for m in seen)


def test_ensure_pip_announces_phase_and_skips_when_present(monkeypatch):
    monkeypatch.setattr(
        inst.subprocess, "run",
        lambda *a, **k: type("R", (), {"returncode": 0})())
    seen = []
    inst.ensure_pip("/tmp/venv", progress=seen.append)
    assert any("pip" in m for m in seen)


# --- requirement counting ---------------------------------------------------

def test_requirement_names_follows_dash_r(tmp_path):
    (tmp_path / "core.txt").write_text(
        "# geo stack\nrasterio>=1.3\ngeopandas\nshapely ; python_version>'3'\n")
    plugin = tmp_path / "cpu.txt"
    plugin.write_text("-r core.txt\npsutil>=5.9  # optional\n\n")
    names = inst._requirement_names(str(plugin))
    assert names == ["rasterio", "geopandas", "shapely", "psutil"]


def test_requirement_names_survives_a_missing_include(tmp_path):
    plugin = tmp_path / "cpu.txt"
    plugin.write_text("-r nope.txt\nonnxruntime\n")
    assert inst._requirement_names(str(plugin)) == ["onnxruntime"]


# --- model downloads --------------------------------------------------------

def test_download_models_reports_per_model(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"General": "https://h/General.onnx", '
                   '"Beech": "https://h/Beech.onnx"}')

    def _fake_retrieve(url, dest, reporthook=None):
        with open(dest, "wb") as fh:
            fh.write(b"onnx")
        if reporthook:
            reporthook(1, 1 << 20, 4 << 20)
        return dest, None

    monkeypatch.setattr(inst.urllib.request, "urlretrieve", _fake_retrieve)
    seen = []
    missing = inst.download_models(str(tmp_path), str(cfg),
                                   progress=seen.append)
    assert missing == []
    body = "\n".join(seen)
    assert "General" in body and "Beech" in body
    assert "(1 of 2)" in body and "(2 of 2)" in body


# --- headless safety --------------------------------------------------------

def test_progress_none_is_safe(monkeypatch, tmp_path):
    """Every step must work with the default progress=None (batch/CLI)."""
    _patch_popen(monkeypatch, PIP_LINES)
    inst.install_requirements_into("/fake/python")
    inst.create_venv("/tmp/venv", base_python="/fake/py")
    inst._run_streamed(["/fake/x"], label="x", timeout=5)
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    assert inst.download_models(str(tmp_path), str(cfg)) == []


def test_progress_sink_errors_do_not_break_the_build(monkeypatch):
    """A Qt signal that dies mid-build must not fail the install."""
    def _boom(msg):
        raise RuntimeError("widget gone")

    _patch_popen(monkeypatch, PIP_LINES)
    inst.install_requirements_into("/fake/python", progress=_boom)


def test_setup_environment_reports_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(inst, "is_ready", lambda p: True)
    monkeypatch.setattr(inst, "download_models",
                        lambda d, progress=None: [])
    seen = []
    info = inst.setup_environment(str(tmp_path / "venv"),
                                  plugin_dir=str(tmp_path),
                                  progress=seen.append)
    assert info["missing_models"] == []
    assert any("Environment ready" in m for m in seen)
