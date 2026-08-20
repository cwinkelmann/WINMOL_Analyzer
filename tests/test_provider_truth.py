"""Provider truth: what the ONNX session actually bound, not just what was
requested. onnxruntime silently falls back to CPU on an unavailable or
unbindable provider (a UserWarning, no exception) -- these tests pin the
two distinct demotion reasons and the loud runtime warning that catches it."""
from utils import onnx_runtime as onx


def test_demotion_reason_names_onnxruntime_gpu_when_not_in_build(monkeypatch):
    monkeypatch.setattr(onx, "_available_providers",
                        lambda: ["CPUExecutionProvider"])
    active, demoted, reason = onx.verify_session_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"])
    assert demoted == ["CUDAExecutionProvider"]
    assert "onnxruntime-gpu" in reason


def test_demotion_reason_is_factual_when_offered_but_not_bound(monkeypatch):
    monkeypatch.setattr(
        onx, "_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    active, demoted, reason = onx.verify_session_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"])
    assert demoted == ["CUDAExecutionProvider"]
    assert "onnxruntime-gpu" not in reason
    assert "CUDAExecutionProvider" in reason


def test_no_demotion_when_requested_matches_active():
    active, demoted, reason = onx.verify_session_providers(
        ["CPUExecutionProvider"], ["CPUExecutionProvider"])
    assert demoted == []
    assert reason is None


class _StubIO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _StubSession:
    """Fakes a session that bound only CPU despite CUDA being requested."""

    def __init__(self, *args, **kwargs):
        pass

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [_StubIO("input", [1, 512, 512, 3])]

    def get_outputs(self):
        return [_StubIO("output", [1, 512, 512, 1])]


def test_onnx_segmenter_warns_loudly_on_demotion(monkeypatch, capsys):
    monkeypatch.setattr(onx.ort, "InferenceSession", _StubSession)
    seg = onx.OnnxSegmenter(
        "fake.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "CUDAExecutionProvider" in out
    report = onx.last_active_report()
    assert report["active_providers"] == ["CPUExecutionProvider"]
    assert report["demoted"] == ["CUDAExecutionProvider"]
    assert seg.demoted == ["CUDAExecutionProvider"]


def test_onnx_segmenter_silent_when_active_matches_requested(
        monkeypatch, capsys):
    class _MatchingSession(_StubSession):
        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(onx.ort, "InferenceSession", _MatchingSession)
    onx.OnnxSegmenter("fake.onnx", providers=["CPUExecutionProvider"])
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert onx.last_active_report()["reason"] is None
