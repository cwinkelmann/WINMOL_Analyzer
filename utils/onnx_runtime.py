"""ONNX inference adapter — makes an .onnx segmentation model duck-type the
Keras model object the pipeline expects (``predict_on_batch``).

Vendored into the analyzer so running .onnx models needs only numpy +
onnxruntime — NO TensorFlow, NO external winmol_unet dependency. Layout-aware:
it reads the model's declared I/O names and layout from the session, so it
handles both the Zenodo models converted in NHWC
(scripts/convert_models_to_onnx.py) and NCHW exports (the PyTorch bridge). The
external contract is always NHWC: ``predict_on_batch([N,512,512,3]) ->
[N,512,512,1]``.
"""
import os

import numpy as np
import onnxruntime as ort

IMG_SIZE = 512
IN_CHANNELS = 3
OUT_CHANNELS = 1


class OnnxOutOfMemoryError(RuntimeError):
    """Raised on ONNX-runtime OOM; caught by the analyzer's batch-backoff."""


def _truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _default_providers():
    """Select execution providers, preferring an available accelerator.

    Precedence: ``WINMOL_ONNX_PROVIDERS`` (explicit list) >
    ``WINMOL_ONNX_FORCE_CPU`` > CUDA > CoreML (macOS) > CPU. CPU is always the
    final fallback so any op an accelerator cannot run still executes.
    """
    override = os.environ.get("WINMOL_ONNX_PROVIDERS")
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    if _truthy(os.environ.get("WINMOL_ONNX_FORCE_CPU", "")):
        return ["CPUExecutionProvider"]
    avail = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in avail:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _layout(shape, channels):
    """'NHWC' or 'NCHW' for a 4D I/O shape, by which axis holds `channels`
    (3 for input, 1 for output). Defaults to NHWC when ambiguous."""
    if len(shape) == 4:
        if shape[-1] == channels:
            return "NHWC"
        if shape[1] == channels:
            return "NCHW"
    return "NHWC"


class OnnxSegmenter:
    def __init__(self, model_path, providers=None):
        self.model_path = model_path
        self.providers = providers or _default_providers()
        # Optional op-level profiling: WINMOL_ONNX_PROFILE=1 makes onnxruntime
        # emit a chrome-trace JSON with per-op time AND the execution provider
        # each op ran on (e.g. CoreMLExecutionProvider vs CPUExecutionProvider)
        # -- i.e. how much actually ran on the Metal/ANE accelerator. No sudo.
        so = None
        self._profile_prefix = None
        if os.environ.get("WINMOL_ONNX_PROFILE", "").strip().lower() in (
                "1", "true", "yes"):
            so = ort.SessionOptions()
            so.enable_profiling = True
            self._profile_prefix = (
                os.environ.get("WINMOL_ONNX_PROFILE_PREFIX")
                or "winmol_onnx_profile")
            so.profile_file_prefix = self._profile_prefix
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=self.providers)
        if self._profile_prefix:
            import atexit
            atexit.register(self._flush_profile)
            print(f"[onnx-profile] enabled (prefix {self._profile_prefix})",
                  flush=True)
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name
        self.input_layout = _layout(inp.shape, IN_CHANNELS)
        self.output_layout = _layout(out.shape, OUT_CHANNELS)

    def _flush_profile(self):
        try:
            path = self.session.end_profiling()
            print(f"[onnx-profile] wrote {path}", flush=True)
        except Exception:
            pass

    @staticmethod
    def _as_numpy(x):
        if hasattr(x, "numpy"):        # TF/torch tensor
            x = x.numpy()
        return np.ascontiguousarray(np.asarray(x, dtype=np.float32))

    def predict_on_batch(self, x):
        """x: NHWC [N,512,512,3] -> NHWC [N,512,512,1]."""
        x = self._as_numpy(x)
        feed = x if self.input_layout == "NHWC" else \
            np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
        try:
            out = self.session.run(
                [self.output_name], {self.input_name: feed})[0]
        except Exception as exc:       # normalize OOM for the retry loop
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg or "cudaerror" in msg:
                raise OnnxOutOfMemoryError(str(exc)) from exc
            raise
        if self.output_layout == "NCHW":
            out = np.transpose(out, (0, 2, 3, 1))
        return np.ascontiguousarray(out.astype(np.float32))

    def summary(self):
        print(
            f"OnnxSegmenter(providers={self.providers}) "
            f"in={self.input_name}[{self.input_layout}] "
            f"out={self.output_name}[{self.output_layout}] "
            f"-> NHWC [N,{IMG_SIZE},{IMG_SIZE},1]"
        )
