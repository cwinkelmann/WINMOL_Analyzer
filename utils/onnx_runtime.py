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
import platform

import numpy as np
import onnxruntime as ort

IMG_SIZE = 512
IN_CHANNELS = 3
OUT_CHANNELS = 1

# Human-readable name of the device inference actually runs on. Keyed by the
# accelerator "kind" the rest of the codebase reasons about.
ACCELERATOR_LABELS = {
    "cuda": "NVIDIA GPU (CUDA)",
    "coreml": "Apple Silicon GPU (Metal/CoreML)",
    "cpu": "CPU",
}


#: Providers whose native libraries ship in separate wheels and therefore
#: need :func:`preload_native_libs` before a session can be created.
_CUDA_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")

_PRELOADED = None


class OnnxOutOfMemoryError(RuntimeError):
    """Raised on ONNX-runtime OOM; caught by the analyzer's batch-backoff."""


def preload_native_libs(providers=None):
    """Load the CUDA/cuDNN shared libraries before a session is created.

    ``onnxruntime-gpu`` gets CUDA from the ``nvidia-*-cu12`` wheels, which
    install into ``site-packages/nvidia/*/lib`` — not on any loader path. The
    provider is then listed by ``get_available_providers()`` (that reports
    what the build was COMPILED with) and still fails to initialise, so
    onnxruntime falls back to the CPU with nothing but a warning. Measured on
    an RTX 4080 SUPER: 10311 ms per tile instead of 10.5 ms — the GPU install
    delivering nothing at all.

    ``onnxruntime.preload_dlls()`` (>= 1.21) fixes that in-process, which is
    why it is preferred over environment plumbing: no child env, no re-exec.
    Returns True when it ran. Only called for CUDA/TensorRT, so the CPU wheel
    and macOS/CoreML are untouched by construction; still guarded, because the
    function does not exist on older wheels.
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


def _truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _available_providers():
    """Execution providers this onnxruntime build offers (seam for tests)."""
    return list(ort.get_available_providers())


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
    avail = set(_available_providers())
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in avail:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def selected_providers():
    """The providers the analyzer will hand to onnxruntime.

    Public name for :func:`_default_providers` so hardware reporting asks the
    inference runtime what it will do instead of re-deriving the precedence.
    """
    return _default_providers()


def _provider_override_env():
    """Name of the env var pinning the provider list, or None."""
    if os.environ.get("WINMOL_ONNX_PROVIDERS"):
        return "WINMOL_ONNX_PROVIDERS"
    if _truthy(os.environ.get("WINMOL_ONNX_FORCE_CPU", "")):
        return "WINMOL_ONNX_FORCE_CPU"
    return None


def accelerator():
    """``(kind, label)`` of the device inference is EXPECTED to use.

    Derived from :func:`selected_providers`, i.e. from what will be REQUESTED —
    a prediction, not an observation. onnxruntime happily builds a CPU session
    when a requested provider is unavailable (only a Python UserWarning), so
    this must never be the last word once a session exists: use
    :func:`active_accelerator` on ``session.get_providers()`` for that.

    Note ``ort.get_device()`` is deliberately NOT used: it reports "CPU" on
    macOS even when CoreML is available and selected.
    """
    return active_accelerator(selected_providers())


def active_accelerator(providers):
    """``(kind, label)`` for a provider list, e.g. ``session.get_providers()``.

    Same mapping as :func:`accelerator`, but driven by whichever list the
    caller passes — so it can report what a session actually BOUND rather than
    what was asked for.
    """
    providers = list(providers or [])
    if "CUDAExecutionProvider" in providers:
        return "cuda", ACCELERATOR_LABELS["cuda"]
    # onnxruntime lists CoreMLExecutionProvider on Intel macOS builds too;
    # only Apple Silicon has the GPU/ANE this label promises.
    if ("CoreMLExecutionProvider" in providers
            and platform.system() == "Darwin"
            and platform.machine() == "arm64"):
        return "coreml", ACCELERATOR_LABELS["coreml"]
    return "cpu", ACCELERATOR_LABELS["cpu"]


def verify_session_providers(requested, active):
    """Compare what was asked for against what the session actually bound.

    Returns ``(active, demoted, reason)``. ``demoted`` lists the requested
    accelerator providers missing from ``active`` — CPU is excluded because it
    is always appended as a deliberate fallback, so its absence is not a
    demotion. ``reason`` explains the FIRST demotion and is None when there is
    none.

    Two distinct causes, which need two very different remedies:

    * the provider is not in this onnxruntime build at all — the CPU-only
      ``onnxruntime`` wheel has no CUDA. Fix: install ``onnxruntime-gpu``.
    * the provider is offered but did not bind — typically a CUDA/cuDNN/driver
      mismatch. We cannot diagnose that from here, so the message stays factual.
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
            "packages and cannot be co-installed — see docs/GPU.md"
        )
    else:
        reason = (
            f"{first} is offered by this build but did not initialise, so it "
            "is not in the active provider list; inference runs on the CPU"
        )
    return active, demoted, reason


# Last verified session result, so the banner printed BEFORE the model is
# loaded ("Hardware detected: ...") can be corrected afterwards.
_LAST_ACTIVE = None


def last_active_report():
    """The most recent verified session result, or None if none was built."""
    return dict(_LAST_ACTIVE) if _LAST_ACTIVE else None


def runtime_report():
    """Everything the startup banner needs about the inference runtime.

    ``accelerator``/``accelerator_label`` are EXPECTED values derived from the
    requested providers. Once a session exists, prefer
    :func:`last_active_report` — that is the observed truth.
    """
    kind, label = accelerator()
    return {
        "onnxruntime_version": getattr(ort, "__version__", "unknown"),
        "available_providers": _available_providers(),
        "selected_providers": selected_providers(),
        "accelerator": kind,
        "accelerator_label": label,
        "override": _provider_override_env(),
    }


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
        # Before the session, not after: an unloadable libcudnn is the
        # difference between 10 ms and 10 s a tile, and onnxruntime reports it
        # as a warning on a session that works.
        preload_native_libs(self.providers)
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=self.providers)
        if self._profile_prefix:
            import atexit
            atexit.register(self._flush_profile)
            print(f"[onnx-profile] enabled (prefix {self._profile_prefix})",
                  flush=True)
        self._verify_providers()
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name
        self.input_layout = _layout(inp.shape, IN_CHANNELS)
        self.output_layout = _layout(out.shape, OUT_CHANNELS)

    def _verify_providers(self):
        """Record what the session BOUND and say so when it is not what we
        asked for. onnxruntime does not raise on an unavailable provider — it
        quietly builds a CPU session — so without this the analyzer would go on
        reporting a GPU it is not using."""
        global _LAST_ACTIVE
        try:
            active = list(self.session.get_providers())
        except Exception:
            # Never let reporting break inference.
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
                f"{', '.join(self.demoted)} are NOT active — onnxruntime is "
                f"running this model on: {', '.join(self.active_providers)} "
                f"(device: {self.accelerator_label}). "
                f"Reason: {self.demotion_reason}",
                flush=True,
            )

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
