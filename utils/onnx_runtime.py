"""ONNX inference adapter — makes an .onnx segmentation model duck-type the
Keras model object the pipeline expects (``predict_on_batch``).

Vendored so running .onnx models needs only numpy + onnxruntime, no
TensorFlow. Layout-aware: it reads the model's declared I/O shapes from the
session, so it handles both NHWC and NCHW exports. The external contract is
always NHWC: ``predict_on_batch([N,512,512,3]) -> [N,512,512,1]``.
"""
import os
import platform

import numpy as np
import onnxruntime as ort

IN_CHANNELS = 3
OUT_CHANNELS = 1

#: Human-readable name of the device inference actually runs on, keyed by
#: the accelerator "kind" the rest of the codebase reasons about.
ACCELERATOR_LABELS = {
    "cuda": "NVIDIA GPU (CUDA)",
    "coreml": "Apple Silicon GPU (Metal/CoreML)",
    "cpu": "CPU",
}

#: Providers whose native libraries ship in separate wheels and therefore
#: need preloading (see preload_native_libs) before a session can be built.
_CUDA_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")

_PRELOADED = None

# Last verified session result, so a banner printed before the model loads
# can be corrected afterwards with what actually bound.
_LAST_ACTIVE = None

#: onnxruntime words an out-of-memory failure differently per execution
#: provider, and most variants contain NEITHER "out of memory" NOR "oom":
#: the CUDA BFC arena says "Failed to allocate memory for requested buffer of
#: size N" (bfc_arena.cc), CUDA proper says "cudaErrorMemoryAllocation", and
#: TensorRT says "ResourceExhausted". Missing any of these means the batch-size
#: back-off in Prediction never fires and the whole run aborts on an OOM that
#: a smaller micro-batch would have survived (issue #40).
_OOM_MARKERS = (
    "out of memory", "oom", "cudaerror", "failed to allocate memory",
    "bfc_arena", "resource_exhausted", "resourceexhausted", "bad_alloc",
)


def _looks_like_oom(message) -> bool:
    """True if an exception message reads like an allocation/OOM failure from
    any onnxruntime execution provider (see _OOM_MARKERS)."""
    msg = str(message).lower()
    return any(marker in msg for marker in _OOM_MARKERS)


def _truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def preload_native_libs(providers=None):
    """Best-effort ctypes preload of the CUDA/cuDNN libs before a session is
    created. ``onnxruntime-gpu`` ships them in ``nvidia-*-cu12`` wheels that
    aren't on any loader path, so without this a requested CUDA session can
    silently fall back to CPU with nothing but a warning. No-op for anything
    but CUDA/TensorRT; never raises -- reporting must not block inference.
    """
    global _PRELOADED
    providers = list(providers or [])
    if providers and not any(p in _CUDA_PROVIDERS for p in providers):
        return False
    if _PRELOADED is not None:
        return _PRELOADED
    fn = getattr(ort, "preload_dlls", None)
    if fn is None:
        _PRELOADED = False
        return False
    try:
        fn()
        _PRELOADED = True
    except Exception as exc:                     # never block inference
        print(f"NOTE: onnxruntime.preload_dlls() failed ({exc}); relying on "
              "the loader path for the CUDA libraries.", flush=True)
        _PRELOADED = False
    return _PRELOADED


def _available_providers():
    """Execution providers this onnxruntime build offers (seam for tests)."""
    return list(ort.get_available_providers())


def _default_providers():
    """Select execution providers, preferring an available accelerator.

    Precedence: ``WINMOL_ONNX_PROVIDERS`` (explicit list) >
    ``WINMOL_ONNX_FORCE_CPU`` > CUDA > CoreML (Apple Silicon macOS) > CPU.
    CPU is always the final fallback so any op an accelerator cannot run
    still executes.
    """
    override = os.environ.get("WINMOL_ONNX_PROVIDERS")
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    if _truthy(os.environ.get("WINMOL_ONNX_FORCE_CPU", "")):
        return ["CPUExecutionProvider"]
    avail = set(_available_providers())
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    is_apple_silicon = (platform.system() == "Darwin"
                        and platform.machine() == "arm64")
    if "CoreMLExecutionProvider" in avail and is_apple_silicon:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def selected_providers():
    """The providers the analyzer will hand to onnxruntime."""
    return _default_providers()


def active_accelerator(providers):
    """``(kind, label)`` for a bound provider list, e.g.
    ``session.get_providers()``. CoreML is listed on Intel macOS builds too,
    so it only counts here on Apple Silicon, which actually has the GPU/ANE
    this label promises."""
    providers = list(providers or [])
    if "CUDAExecutionProvider" in providers:
        return "cuda", ACCELERATOR_LABELS["cuda"]
    if ("CoreMLExecutionProvider" in providers
            and platform.system() == "Darwin"
            and platform.machine() == "arm64"):
        return "coreml", ACCELERATOR_LABELS["coreml"]
    return "cpu", ACCELERATOR_LABELS["cpu"]


def verify_session_providers(requested, active):
    """Compare what was asked for against what a session actually bound.

    Returns ``(active, demoted, reason)``. ``demoted`` lists requested
    providers missing from ``active`` -- CPU excluded, since it is always
    appended as a deliberate fallback and its absence is not a demotion.
    ``reason`` explains the first demotion, or None when there is none, as
    one of two distinct causes with two very different remedies: the
    provider is not in this onnxruntime build at all (the CPU-only
    ``onnxruntime`` wheel has no CUDA -- fix: install ``onnxruntime-gpu``),
    or it is offered but did not bind (typically a CUDA/cuDNN/driver
    mismatch we cannot diagnose from here, so the message stays factual).
    """
    active = list(active or [])
    requested = list(requested or [])
    demoted = [p for p in requested
               if p not in active and p != "CPUExecutionProvider"]
    if not demoted:
        return active, [], None

    try:
        available = set(_available_providers())
    except Exception:
        available = set()

    first = demoted[0]
    if first not in available:
        reason = (
            f"{first} is not provided by this onnxruntime build. "
            "'onnxruntime' (CPU-only) and 'onnxruntime-gpu' are DIFFERENT "
            "packages and cannot be co-installed -- install "
            "'onnxruntime-gpu' to get CUDA support"
        )
    else:
        reason = (
            f"{first} is offered by this build but did not initialise, so "
            f"it is not in the active provider list; inference runs on "
            f"{', '.join(active) or 'CPU'}"
        )
    return active, demoted, reason


def last_active_report():
    """The most recent verified OnnxSegmenter session result, or None."""
    return dict(_LAST_ACTIVE) if _LAST_ACTIVE else None


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
        # Before the session, not after: an unloadable libcudnn is the
        # difference between 10 ms and 10 s a tile, and onnxruntime reports
        # it as a warning on a session that otherwise looks fine.
        preload_native_libs(self.providers)
        self.session = ort.InferenceSession(
            str(model_path), providers=self.providers)
        self._verify_providers()
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name
        self.input_layout = _layout(inp.shape, IN_CHANNELS)
        # A graph that carries its own normalize+resize takes RAW uint8
        # NHWC tiles; casting to float32 here would undo the point (4x
        # the PCIe traffic) and break the feed type.
        self.input_is_uint8 = str(getattr(inp, 'type', '')) == 'tensor(uint8)'
        self.output_layout = _layout(out.shape, OUT_CHANNELS)

    def _verify_providers(self):
        """Record what the session BOUND and warn loudly when it is not
        what we asked for. onnxruntime does not raise on an unavailable
        provider -- it quietly builds a CPU session -- so without this the
        analyzer would go on reporting a GPU it is not using."""
        global _LAST_ACTIVE
        try:
            active = list(self.session.get_providers())
        except Exception:
            active = list(self.providers)
        self.active_providers, self.demoted, self.demotion_reason = (
            verify_session_providers(self.providers, active))
        self.accelerator, self.accelerator_label = active_accelerator(
            self.active_providers)
        _LAST_ACTIVE = {
            "active_providers": list(self.active_providers),
            "requested_providers": list(self.providers),
            "demoted": list(self.demoted),
            "reason": self.demotion_reason,
            "accelerator": self.accelerator,
            "accelerator_label": self.accelerator_label,
        }
        if self.demoted:
            print(
                "WARNING: requested execution provider(s) "
                f"{', '.join(self.demoted)} are NOT active -- onnxruntime "
                f"is running this model on: "
                f"{', '.join(self.active_providers)} "
                f"(device: {self.accelerator_label}). "
                f"Reason: {self.demotion_reason}",
                flush=True,
            )

    @staticmethod
    def _as_numpy(x):
        return np.ascontiguousarray(np.asarray(x, dtype=np.float32))

    def predict_on_batch(self, x):
        """x: NHWC [N,512,512,3] -> NHWC [N,512,512,1].

        With an in-graph preprocessing head the input is instead raw
        uint8 NHWC at NATIVE tile size, and the graph resizes it."""
        if getattr(self, 'input_is_uint8', False):
            x = np.ascontiguousarray(x, dtype=np.uint8)
        else:
            x = self._as_numpy(x)
        feed = x if self.input_layout == "NHWC" else \
            np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
        try:
            out = self.session.run(
                [self.output_name], {self.input_name: feed})[0]
        except Exception as exc:       # normalize OOM for the retry loop
            if _looks_like_oom(exc):
                raise MemoryError(str(exc)) from exc
            raise
        if self.output_layout == "NCHW":
            out = np.transpose(out, (0, 2, 3, 1))
        return np.ascontiguousarray(out.astype(np.float32))
