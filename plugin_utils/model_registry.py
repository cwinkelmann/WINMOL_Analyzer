"""WINMOL model registry: load, resolve, and fetch ONNX segmentation
models from ``config.json``.

Two shapes are accepted:

* **schema v2** — ``{"schema": 2, "gui_default": ..., "recommended": [...],
  "families": {fid: {"label", "default"}}, "models": {mid: {"label",
  "family", "precision", "url", "file", "sha256", "size_mb"}}}``. A
  family names only its fp32 default entry; the device-appropriate
  variant (cpu->int8, gpu->fp16, coreml->fp32) is found among the
  family's other entries by matching ``precision``, falling back to the
  family default when no such variant exists.
* **legacy v1** — a flat ``{"Name": "https://...url"}`` map, normalized
  into minimal entries (no family, no checksum).

Import-safe off QGIS: stdlib only, no Qt/QGIS imports — unit-testable
and usable from the batch CLI.
"""

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from .gpu_probe import run_nvidia_smi_query

_CHUNK_BYTES = 1024 * 1024
#: device -> the precision its family variant must carry.
_DEVICE_PRECISION = {"cpu": "int8", "gpu": "fp16", "coreml": "fp32"}


class ModelDownloadError(RuntimeError):
    """A model could not be fetched or failed integrity verification."""

    def __init__(self, message, model_id=None, url=None, cause=None):
        super().__init__(message)
        self.model_id = model_id
        self.url = url
        self.cause = cause


@dataclass(frozen=True)
class ModelEntry:
    """One downloadable model; ``file`` is the shared on-disk name used
    by both the plugin's models dir and the batch CLI's --model-dir."""

    id: str
    label: str
    url: str
    file: str
    family: str = ""
    precision: str = "fp32"
    sha256: Optional[str] = None
    size_mb: Optional[float] = None
    #: Certified to reproduce the fp32 reference's results. Gates the
    #: ``variant="auto"`` (lossless-only) resolution rule; an fp32
    #: reference is lossless by definition, quantised builds only when
    #: the registry says so.
    lossless: bool = True
    #: Never shown in choosers (forward-compat with registries that
    #: carry non-runnable formats, e.g. the TensorFlow .hdf5 originals).
    hidden: bool = False


@dataclass
class Family:
    """A group of precision variants of one trained model."""

    id: str
    label: str
    default: str    # entry id of the fp32 reference


class Registry:
    """Parsed model registry: entries, families, and resolution rules."""

    def __init__(self, entries, families=None, schema=1,
                 gui_default=None, recommended=None):
        self.entries: Dict[str, ModelEntry] = dict(entries)
        self.families: Dict[str, Family] = dict(families or {})
        self.schema = schema
        self.gui_default = gui_default
        #: Ranked entry ids, best first; recommended[0] == gui_default.
        self.recommended: List[str] = list(recommended or [])
        self._entry_lookup = {k.lower(): k for k in self.entries}
        self._family_lookup = {k.lower(): k for k in self.families}
        #: family id -> {precision: entry id}, for _device_variant.
        self._by_family_precision: Dict[str, Dict[str, str]] = {}
        for e in self.entries.values():
            if e.family:
                self._by_family_precision.setdefault(
                    e.family, {})[e.precision] = e.id

    def get(self, name) -> ModelEntry:
        """Entry by id, case-insensitive. KeyError names the known ids."""
        key = str(name).strip().lower()
        canonical = self._entry_lookup.get(key)
        if canonical is None:
            known = ", ".join(sorted(self.entries))
            raise KeyError(f"unknown model {name!r}; known models: {known}")
        return self.entries[canonical]

    def _family_precision_entry(self, fam, precision):
        """``fam``'s entry carrying ``precision``, or None."""
        eid = self._by_family_precision.get(fam.id, {}).get(precision)
        return self.entries[eid] if eid else None

    def _device_variant(self, fam, device) -> ModelEntry:
        """``fam``'s entry whose precision matches ``device``'s rule
        (cpu->int8, gpu->fp16, coreml->fp32), else ``fam``'s default."""
        precision = _DEVICE_PRECISION.get(device)
        entry = (self._family_precision_entry(fam, precision)
                 if precision else None)
        return entry if entry else self.entries[fam.default]

    def resolve(self, name, device=None, variant=None) -> ModelEntry:
        """Resolve a model or family id to a concrete entry.

        An explicit entry id is returned as-is — neither ``device`` nor
        ``variant`` ever rewrites it. A family id resolves by
        ``variant``:

        * ``None`` / ``"default"`` — the device rule (cpu->int8,
          gpu->fp16, coreml->fp32, see ``_device_variant``), falling
          back to the family default. Unchanged legacy behavior.
        * ``"auto"`` — lossless-only: the device variant is substituted
          ONLY when it is certified lossless (``ModelEntry.lossless``),
          otherwise the fp32 family default — results stay identical to
          the published reference.
        * ``"fp32"``/``"int8"``/``"fp16"`` — that precision within the
          family; KeyError (naming the family and what it does provide)
          when the family lacks it.

        ``device`` ``None``/``"auto"`` probes the machine. Lookup
        order: exact entry id, exact family id, case-insensitive
        entry, case-insensitive family.
        """
        name_s = str(name).strip()
        if name_s in self.entries:
            return self.entries[name_s]
        fam = self.families.get(name_s)
        if fam is None:
            key = name_s.lower()
            if key in self._entry_lookup:
                return self.get(name_s)
            fam_id = self._family_lookup.get(key)
            if fam_id is None:
                return self.get(name)   # raises the descriptive KeyError
            fam = self.families[fam_id]
        if device in (None, "auto"):
            device = detect_device()
        v = str(variant).strip().lower() if variant is not None else "default"
        if v in ("", "default"):
            return self._device_variant(fam, device)
        if v == "auto":
            cand = self._family_precision_entry(
                fam, _DEVICE_PRECISION.get(device))
            if cand is not None and cand.lossless:
                return cand
            return self.entries[fam.default]
        if v in ("fp32", "int8", "fp16"):
            entry = self._family_precision_entry(fam, v)
            if entry is None:
                have = sorted(self._by_family_precision.get(fam.id, {}))
                raise KeyError(
                    f"family '{fam.id}' has no {v} variant; "
                    f"available precisions: {', '.join(have) or 'none'}")
            return entry
        raise ValueError(
            f"unknown variant {variant!r} "
            "(use default/auto/fp32/int8/fp16)")

    def visible(self) -> List[ModelEntry]:
        """Non-hidden entries in registry (curated) order."""
        return [e for e in self.entries.values() if not e.hidden]

    def default_entry(self, device="auto") -> ModelEntry:
        """The effective default entry for ``device``: the declared
        ``recommended``/``gui_default`` model's device variant, or the
        first entry when the registry declares no default at all."""
        declared_id = None
        for mid in list(self.recommended) + [self.gui_default]:
            if mid and mid in self.entries:
                declared_id = mid
                break
        if declared_id is None:
            if not self.entries:
                raise KeyError("registry has no models")
            declared_id = next(iter(self.entries))
        declared = self.entries[declared_id]
        fam = self.families.get(declared.family)
        if fam is None:
            return declared
        if device == "auto":
            device = detect_device()
        return self._device_variant(fam, device)


# --- loading -----------------------------------------------------------

def load_registry(config_path) -> Registry:
    """Parse config.json (schema v2 or legacy flat v1) into a Registry."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"model registry (config.json) not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid/empty config.json: {config_path}")
    schema = raw.get("schema")
    if isinstance(schema, int) and schema >= 2:
        return _parse_v2(raw, config_path)
    return _parse_v1(raw, config_path)


def _v1_filename(name, url):
    """installer.py's historical dest naming: <Name><ext-from-URL>."""
    path = url.split("?")[0] if url else ""
    ext = os.path.splitext(path)[1] if path else ""
    return f"{name}{ext or '.onnx'}"


def _parse_v1(raw, config_path) -> Registry:
    entries = {}
    for name, url in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        u = url if isinstance(url, str) else ""
        entries[name] = ModelEntry(
            id=name, label=name, url=u, file=_v1_filename(name, u))
    if not entries:
        raise ValueError(
            f"No model entries found in {config_path}. Expected "
            "{name: url}.")
    return Registry(entries, schema=1)


def _parse_v2(raw, config_path) -> Registry:
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"schema-2 registry without models: {config_path}")
    entries = {}
    for mid, spec in models.items():
        if not isinstance(spec, dict):
            raise ValueError(f"model {mid!r} is not an object")
        url = spec.get("url")
        if not isinstance(url, str) or not url.lower().startswith("http"):
            raise ValueError(f"model {mid!r} has no http(s) url")
        file = spec.get("file")
        if not isinstance(file, str) or not file.strip():
            raise ValueError(f"model {mid!r} has no 'file' name")
        entries[mid] = ModelEntry(
            id=mid,
            label=str(spec.get("label", mid)),
            family=str(spec.get("family", "")),
            precision=str(spec.get("precision", "fp32")),
            url=url,
            file=file.strip(),
            sha256=spec.get("sha256") or None,
            size_mb=spec.get("size_mb"),
            lossless=bool(spec.get("lossless", True)),
            hidden=bool(spec.get("hidden", False)),
        )

    families = {}
    for fid, spec in (raw.get("families") or {}).items():
        default = spec.get("default") if isinstance(spec, dict) else None
        if default is None or default not in entries:
            raise ValueError(f"family {fid!r} has no valid default model")
        families[fid] = Family(
            id=fid, label=str(spec.get("label", fid)), default=default)

    gui_default = raw.get("gui_default")
    if gui_default is not None and gui_default not in entries:
        raise ValueError(
            f"gui_default references unknown model {gui_default!r}")
    recommended = raw.get("recommended") or []
    if not isinstance(recommended, list):
        raise ValueError("'recommended' must be a list of model ids")
    for rid in recommended:
        if rid not in entries:
            raise ValueError(
                f"recommended references unknown model {rid!r}")

    return Registry(entries, families, schema=int(raw["schema"]),
                    gui_default=gui_default, recommended=recommended)


# --- paths / device ------------------------------------------------------

def local_path(entry, model_dir) -> str:
    """The single on-disk naming rule shared by plugin and CLI."""
    return os.path.join(model_dir, entry.file)


def gpu_runtime_installed(venv_path=None) -> bool:
    """Whether the interpreter that will RUN the model has a CUDA EP.

    A card is not the same thing as a runtime that can use it. The
    managed venv is built from requirements/cpu.txt (plain
    ``onnxruntime``) unless the user opts into the GPU variant, so on
    any NVIDIA box a default install has a GPU present and a CPU-only
    runtime. Choosing the model by the CARD alone hands that install the
    fp16 GPU variant instead of the int8 CPU one -- it still runs (ORT
    converts the fp16 weights to fp32 once at session load), just as the
    wrong, slower variant.

    Two contexts, two sources of truth:

    * the compute child already has onnxruntime imported -- ask it;
    * the QGIS-side plugin must never import it, so read the venv
      sentinel instead (``installed_variant`` is pure file I/O).

    Unknown (no marker, no imported runtime) stays True so the probe
    alone decides, exactly as before.
    """
    ort = sys.modules.get("onnxruntime")
    if ort is not None:
        try:
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            pass
    if venv_path:
        try:
            from .installer import installed_variant
            variant = installed_variant(venv_path)
        except Exception:
            variant = None
        if variant is not None:
            return variant == "gpu"
    return True


def detect_device(venv_path=None) -> str:
    """"gpu" (CUDA), "coreml" (Apple Silicon) or "cpu".

    ``WINMOL_DEVICE`` env overrides; else Apple Silicon is recognised
    from the platform; else an ``nvidia-smi`` probe; default "cpu".

    A GPU verdict additionally requires a runtime that can actually use
    the card (``gpu_runtime_installed``) -- otherwise a CPU-only install
    on an NVIDIA machine selects the fp16 variant it cannot accelerate.
    Skipping the probe in that case also skips its 20 s timeout.
    """
    forced = os.environ.get("WINMOL_DEVICE", "").strip().lower()
    if forced in ("cpu", "gpu", "coreml"):
        return forced
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "coreml"
    if not gpu_runtime_installed(venv_path):
        return "cpu"
    return _probe_nvidia()


def _probe_nvidia() -> str:
    """"gpu" if ``nvidia-smi`` lists at least one GPU, else "cpu"."""
    return "gpu" if run_nvidia_smi_query("index", timeout=20) else "cpu"


# --- integrity verification -----------------------------------------------

def _hash_file(path, algo="sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_file(path, sha256=None) -> bool:
    """True when ``path``'s sha256 matches, or ``sha256`` is falsy
    (unpinned entries pass — legacy behavior)."""
    if not sha256:
        return True
    if not os.path.exists(path):
        return False
    return _hash_file(path, "sha256") == sha256.lower()


# --- download --------------------------------------------------------------

def _urllib_fetcher(url, tmp_path, progress, timeout):
    """Stream ``url`` to ``tmp_path`` in 1 MiB chunks, hashing sha256 on
    the fly. ``progress(bytes_done, total_or_None)``. Returns the hex
    digest so callers skip a second pass over the file."""
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "WINMOL-Analyzer"})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total and str(total).isdigit() else None
        done = 0
        with open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    return h.hexdigest()


#: Injectable for offline tests (pass fetcher= to download_model).
_DEFAULT_FETCHER = _urllib_fetcher


def _discard(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def download_model(entry, dest_dir, progress=None, fetcher=None,
                   timeout=30.0) -> str:
    """Atomic verified fetch: stream to ``<file>.part``, check sha256,
    then ``os.replace`` onto the final name. A crash can only ever leave
    a ``.part`` file, never a truncated model at the final path — and
    ``.part`` is removed on any failure. Raises ModelDownloadError.

    ``fetcher(url, tmp_path, progress2, timeout)`` writes the payload
    to ``tmp_path``; may return a sha256 hex digest to skip a re-hash.
    """
    fetch = fetcher or _DEFAULT_FETCHER
    if not entry.url or not entry.url.lower().startswith("http"):
        raise ModelDownloadError(
            f"model '{entry.id}' has no downloadable URL",
            model_id=entry.id, url=entry.url)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, entry.file)
    tmp = dest + ".part"
    if progress is None:
        inner = None
    else:
        def inner(done, total):
            progress(done, total, entry)
    try:
        digest = fetch(entry.url, tmp, inner, timeout)
        if entry.sha256:
            actual = digest or _hash_file(tmp, "sha256")
            if actual.lower() != entry.sha256.lower():
                raise ModelDownloadError(
                    f"checksum mismatch for {entry.file}: expected "
                    f"sha256 {entry.sha256}, got {actual} — corrupt or "
                    "tampered download, file discarded",
                    model_id=entry.id, url=entry.url)
        os.replace(tmp, dest)
        return dest
    except ModelDownloadError:
        _discard(tmp)
        raise
    except Exception as exc:
        _discard(tmp)
        raise ModelDownloadError(
            f"download failed for {entry.file} from {entry.url}: {exc}",
            model_id=entry.id, url=entry.url, cause=exc) from exc


def ensure_model(name_or_entry, model_dir, registry=None, progress=None,
                 fetcher=None, no_download=False) -> str:
    """The call-site API: return a verified local path for a model,
    downloading it if needed.

    An existing file that matches its checksum (or has none) short-
    circuits. A missing or checksum-failing file is (re)downloaded,
    unless ``no_download``, which raises ModelDownloadError instead.
    """
    if isinstance(name_or_entry, ModelEntry):
        entry = name_or_entry
    else:
        if registry is None:
            raise ValueError(
                "registry is required when passing a model name")
        entry = registry.resolve(name_or_entry)
    path = local_path(entry, model_dir)
    if (os.path.exists(path) and os.path.getsize(path) > 0
            and verify_file(path, entry.sha256)):
        return path
    if no_download:
        raise ModelDownloadError(
            f"model file missing or stale: {path} (downloads disabled; "
            f"fetch {entry.url} manually)",
            model_id=entry.id, url=entry.url)
    return download_model(entry, model_dir, progress=progress,
                          fetcher=fetcher)
