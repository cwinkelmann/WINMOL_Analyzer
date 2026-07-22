# GPU inference

WINMOL runs its U-Net through **onnxruntime**. Which device that uses is
decided entirely by the *execution provider* onnxruntime binds when the session
is created — not by your driver, not by `nvidia-smi`, and not by anything the
plugin installs by default.

The run now tells you the truth about this. At startup:

```
Environment:
  Inference runtime: onnxruntime 1.27.0
  Available providers: CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider
  Selected providers: CoreMLExecutionProvider, CPUExecutionProvider
  Active providers: CoreMLExecutionProvider, CPUExecutionProvider
  Device: Apple Silicon GPU (Metal/CoreML) (verified)
```

**Selected** is what was requested; **Active** is what the session actually
bound. Only the `(verified)` device line reflects reality. If they differ you
get a loud `WARNING:` naming the demoted provider and why.

---

## Windows / Linux with an NVIDIA GPU

**The shipped plugin environment can never use CUDA.** `requirements/plugin.txt`
pulls in `requirements/base.txt`, which installs `onnxruntime` — the CPU-only
wheel. `onnxruntime-gpu` is a **different package**; CUDA support is not a flag
you turn on, it is a different distribution. This is why an RTX 4080 can sit at
0 % utilisation while a run crawls.

You will see this at startup:

```
WARNING: nvidia-smi reports ['NVIDIA GeForce RTX 4080'] but this onnxruntime
build cannot use them (CPUExecutionProvider only), so inference will run on the
CPU. Install onnxruntime-gpu to use them — see docs/GPU.md.
```

### Fix

Find the environment the analyzer runs in. For the QGIS plugin this is the
`winmol_venv` the installer created (the plugin dialog shows the path; it lives
under your QGIS profile directory). Then **swap** the wheel — never install both:

```bash
# 1. remove the CPU-only wheel (both packages provide the `onnxruntime` module)
<venv>/bin/pip uninstall -y onnxruntime

# 2. install the CUDA build
<venv>/bin/pip install -r requirements/onnxruntime-gpu.txt

# 3. VERIFY — installing is not the same as working
<venv>/bin/python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

On Windows use `<venv>\Scripts\pip.exe` and `<venv>\Scripts\python.exe`.

Step 3 must print a list containing `CUDAExecutionProvider`. If it does not,
the wheel does not match your driver — see *Version pinning* below. Re-run the
analyzer and confirm the banner says:

```
  Active providers: CUDAExecutionProvider, CPUExecutionProvider
  Device: NVIDIA GPU (CUDA) (verified)
```

### Version pinning

The `[cuda,cudnn]` extras ship the CUDA userspace as wheels, so you do **not**
need a system CUDA toolkit — only a recent enough NVIDIA driver. But
`onnxruntime-gpu` 1.27 moved those extras to **CUDA 13**, while earlier releases
use CUDA 12. If your driver predates CUDA 13, pin a cu12-era release in
`requirements/onnxruntime-gpu.txt`:

```
onnxruntime-gpu[cuda,cudnn]==1.22.0
```

Blackwell (sm_120) needs a recent CUDA — verify with step 3 on the actual
hardware before trusting any timing.

### The other failure mode

If `onnxruntime-gpu` **is** installed but CUDA/cuDNN/driver versions do not
match, onnxruntime does not raise. It builds a working **CPU** session and emits
only a Python `UserWarning`. `CUDAExecutionProvider` then appears in
`get_available_providers()` (so it looks installed) but is absent from
`session.get_providers()`. The analyzer detects exactly this and reports:

```
WARNING: requested execution provider(s) CUDAExecutionProvider are NOT active
— onnxruntime is running this model on: CPUExecutionProvider (device: CPU).
Reason: CUDAExecutionProvider is offered by this build but did not initialise,
so it is not in the active provider list; inference runs on the CPU
```

Containers are unaffected: `docker/gpu/` builds from `requirements/gpu.txt`,
which already uses `onnxruntime-gpu`. See `docs/CONTAINERS.md`.

---

## macOS (Apple Silicon)

**Nothing to install.** The CoreML execution provider ships in the default
`onnxruntime` wheel and the analyzer already selects it automatically. Verified
on an M2 with onnxruntime 1.27.0: the session binds
`['CoreMLExecutionProvider', 'CPUExecutionProvider']`, and onnxruntime reports
at session build:

```
CoreMLExecutionProvider::GetCapability, number of partitions supported by
CoreML: 5 number of nodes in the graph: 74 number of nodes supported by
CoreML: 69
```

So 69 of 74 graph nodes run on the GPU/ANE; the remaining 5 fall back to CPU
across 5 partitions. CoreML is genuinely active — it was never the problem.

### CoreML and batch size

CoreML prices batching the opposite way to CUDA. Measured on
`standalone/model_onnx/Spruce.onnx` (M2, onnxruntime 1.27.0, one fresh process
per row, median of repeats):

| micro-batch | per-image cost |
|-------------|----------------|
| 1           | 0.228 s        |
| 2           | 0.285 s        |
| 6           | 0.435 s        |
| 8           | 0.498 s        |

Per-image cost **rises** with batch size — it does not fall. CPU, by contrast,
is roughly flat. Two consequences, both fixed:

* The planner sizes the micro-batch from GPU memory, and on Apple Silicon
  `HardwareInfo` reports half of unified RAM as "GPU memory", so a Mac landed in
  the 4–8 batch tier: ~2x slower per tile than batch 1. `Config` now caps it via
  `prediction_batch_max_coreml` (default 2). Set it to `None` to disable.
* The batch autotune swept upward through candidates that were all worse, each
  costing more wall time than the last. It now abandons a sweep as soon as a
  candidate is `prediction_batch_autotune_runaway_factor` (default 1.5x) slower
  than the best seen.

Absolute numbers are machine- and load-dependent; the *direction* is not, and
reproduces in fresh processes. Batch size does not affect output — max absolute
difference between batch 1, 4 and 8 predictions is exactly 0.0 — so both changes
are pure throughput, not results.

`WINMOL_DISABLE_METAL=1` forces CoreML off (CPU only) if you need to compare.

---

## Forcing a device

| Variable | Effect |
|----------|--------|
| `WINMOL_ONNX_PROVIDERS` | Explicit comma-separated provider list; highest precedence |
| `WINMOL_ONNX_FORCE_CPU=1` | Pin to `CPUExecutionProvider` |
| `WINMOL_DISABLE_METAL=1` | Apple-Silicon-only: skip CoreML |
| `CUDA_VISIBLE_DEVICES` | Standard NVIDIA device filtering; `""` or `-1` means CPU-only |

`WINMOL_ONNX_PROFILE=1` writes a chrome-trace JSON with per-op timings **and the
provider each op ran on** — the way to see which ops fell back to CPU.
