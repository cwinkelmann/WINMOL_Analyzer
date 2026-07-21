"""WINMOL model registry: load, resolve, and fetch segmentation models.

The single source of truth is ``config.json`` (repo root; shipped inside
the QGIS plugin). Two shapes are accepted:

* **schema v2** — ``{"schema": 2, "families": {...}, "models": {...}}``.
  Each model entry carries a stable id, a human-readable label and
  description, its download URL, the mandatory on-disk ``file`` name
  (always the URL basename), an optional checksum (``sha256`` for the
  model-zoo assets, ``md5`` for the Zenodo HDF5 originals), and metadata
  (family, backend, precision, F1, size). Families group precision
  variants (fp32 reference / int8 CPU / fp16 GPU) of one trained model.
* **legacy v1** — a flat ``{"Name": "https://...url"}`` map, normalized
  into equivalent entries (installer-compatible ``<Name><ext>`` file
  naming, no checksums).

Sources of the shipped registry:
* Model zoo: the ``models-v1`` release of cwinkelmann/WINMOL_segmentor_pt
  (sha256-pinned ONNX). This now includes the classic four
  (General/Beech/Spruce/Spruce_Deadwood), which were repointed from the
  older, unchecksummed ``models-onnx-v1`` release of
  cwinkelmann/WINMOL_Analyzer to the numerically-identical, pinned
  conversions of the same Keras weights. ONNX contract for all of them:
  input [batch,3,512,512] float32 in [0,1] NCHW, output
  [batch,1,512,512], sigmoid baked in, opset 17, dynamic batch — loads
  unchanged through utils/onnx_runtime.OnnxSegmenter.
* Originals: Zenodo record 15907576 (DOI 10.5281/zenodo.15907576), the
  four Keras .hdf5 (374 MB each, md5-pinned, need TensorFlow).

Every shipped entry is digest-pinned (sha256, or md5 for the Zenodo
originals); ``Registry.unpinned()`` reports any that are not and a test
guards the property.

Import-safe off QGIS: stdlib only, no Qt/QGIS imports (same contract as
installer.py) — unit-testable and usable from the batch CLI.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

#: Reserved by the GUI as the "pick your own file" escape hatch.
RESERVED_IDS = ("Custom",)
#: Per-model-dir memo of verified digests (avoid re-hashing 374 MB files).
VERIFIED_CACHE = ".winmol_verified.json"

_CHUNK_BYTES = 1024 * 1024
_DEVICE_PROBE_CACHE = {}


class ModelDownloadError(RuntimeError):
    """A model could not be fetched or failed integrity verification."""

    def __init__(self, message, model_id=None, url=None, cause=None):
        super().__init__(message)
        self.model_id = model_id
        self.url = url
        self.cause = cause


@dataclass(frozen=True)
class ModelEntry:
    """One downloadable model. ``file`` is the on-disk filename used by
    BOTH the QGIS plugin (<plugin>/models/) and the batch CLI
    (--model-dir); for v2 entries it equals the URL basename."""

    id: str
    label: str
    url: str
    file: str
    description: str = ""
    family: str = ""
    backend: str = "any"          # "any" | "cpu" | "gpu"
    precision: str = "fp32"       # "fp32" | "fp16" | "int8"
    format: str = "onnx"          # "onnx" | "hdf5"
    sha256: Optional[str] = None
    md5: Optional[str] = None
    size_mb: Optional[float] = None
    f1: Optional[float] = None
    lossless: bool = True
    hidden: bool = False
    tile_px: int = 512


@dataclass
class Family:
    """A group of precision variants of one trained model."""

    id: str
    label: str
    default: str                  # entry id (fp32 reference)
    cpu: Optional[str] = None     # entry id of the int8/CPU variant
    gpu: Optional[str] = None     # entry id of the fp16/GPU variant


class Registry:
    """Parsed model registry: entries, families, and resolution rules."""

    def __init__(self, entries, families=None, schema=1,
                 gui_default=None, preload=None, tile_px=512,
                 recommended=None):
        self.entries: Dict[str, ModelEntry] = dict(entries)
        self.families: Dict[str, Family] = dict(families or {})
        self.schema = schema
        self.gui_default = gui_default
        self.preload: List[str] = list(preload or [])
        self.tile_px = tile_px
        #: Ranked entry ids, best first; recommended[0] == gui_default.
        self.recommended: List[str] = list(recommended or [])
        self._entry_lookup = {k.lower(): k for k in self.entries}
        self._family_lookup = {k.lower(): k for k in self.families}

    def get(self, name) -> ModelEntry:
        """Entry by id, case-insensitive. KeyError names the visible
        ids (the lowercase-name guarantee winmol_batch relies on)."""
        key = str(name).strip().lower()
        canonical = self._entry_lookup.get(key)
        if canonical is None:
            known = ", ".join(sorted(
                e.id for e in self.visible()) or sorted(self.entries))
            raise KeyError(
                f"unknown model {name!r}; known models: {known}")
        return self.entries[canonical]

    def resolve(self, name, device="auto", variant="auto") -> ModelEntry:
        """Resolve a model or family name to a concrete entry.

        * An explicit entry id is returned as-is — never rewritten by
          ``variant``/``device`` (so "General" always means the fp32
          GenDS model, exactly as before the registry existed).
        * A family id picks the family default; with ``variant="auto"``
          the device variant (cpu->int8, gpu->fp16) is substituted ONLY
          when that variant is certified lossless, keeping results
          stable. A forced variant ("fp32"/"int8"/"fp16") selects that
          variant or raises ValueError if the family lacks it.

        Precedence (family ids like "unet_pt" collide case-insensitively
        with entry ids like "UNet_PT"): exact entry id, exact family id,
        then case-insensitive entry, then case-insensitive family.
        """
        name_s = str(name).strip()
        if name_s in self.entries:
            return self.entries[name_s]
        if name_s in self.families:
            fam = self.families[name_s]
        else:
            key = name_s.lower()
            canonical = self._entry_lookup.get(key)
            if canonical is not None:
                return self.entries[canonical]
            fam_id = self._family_lookup.get(key)
            if fam_id is None:
                return self.get(name)   # raises the descriptive KeyError
            fam = self.families[fam_id]
        default = self.entries[fam.default]
        variant = str(variant or "auto").strip().lower()
        if device == "auto":
            device = detect_device()
        if variant == "auto":
            cand_id = fam.cpu if device == "cpu" else fam.gpu
            cand = self.entries.get(cand_id) if cand_id else None
            if cand is not None and cand.lossless:
                return cand
            return default
        if variant == "fp32":
            return default
        if variant in ("int8", "cpu"):
            if not fam.cpu:
                raise ValueError(
                    f"family '{fam.id}' has no int8/CPU variant")
            return self.entries[fam.cpu]
        if variant in ("fp16", "gpu"):
            if not fam.gpu:
                raise ValueError(
                    f"family '{fam.id}' has no fp16/GPU variant")
            return self.entries[fam.gpu]
        raise ValueError(
            f"unknown variant {variant!r} (use auto/fp32/int8/fp16)")

    def default_entry(self, device="auto") -> ModelEntry:
        """The EFFECTIVE default model for ``device``.

        The registry declares a ranked ``recommended`` list; entry 0 is
        also ``gui_default``. That declaration fixes the *domain* (which
        trained model), and this method fixes the *precision* for the
        machine at hand: the declared default's family supplies the
        int8 variant on CPU and the fp16 variant on GPU, falling back to
        the declared entry itself when the family has no such variant.

        This deliberately does NOT go through :meth:`resolve`'s
        lossless-only gate. ``resolve`` protects users who picked a
        *family* from a silent, results-changing precision swap; here the
        registry has explicitly nominated an optimised entry as the
        default, so honouring the device is the declared intent, not a
        substitution behind the user's back. Any explicit selection (an
        entry id from the GUI, ``winmol_batch <MODEL>``, or a forced
        ``--variant``) bypasses this method entirely.

        Never downloads and never touches the network beyond the local
        ``detect_device()`` probe.
        """
        declared = None
        for mid in list(self.recommended) + [self.gui_default]:
            if mid and mid in self.entries:
                declared = self.entries[mid]
                break
        if declared is None:
            visible = self.visible()
            if not visible:
                raise KeyError("registry has no selectable model")
            return visible[0]
        fam = self.families.get(declared.family)
        if fam is None:
            return declared
        if device == "auto":
            device = detect_device()
        cand_id = fam.cpu if device == "cpu" else fam.gpu
        cand = self.entries.get(cand_id) if cand_id else None
        return cand if cand is not None else declared

    def recommended_entries(self) -> List[ModelEntry]:
        """The declared ranked recommendations, best first."""
        return [self.entries[mid] for mid in self.recommended
                if mid in self.entries]

    def unpinned(self) -> List[ModelEntry]:
        """Downloadable entries with no sha256/md5 digest — i.e. whose
        download cannot be integrity-verified. Must be empty."""
        return [e for e in self.entries.values()
                if e.url and not (e.sha256 or e.md5)]

    def flat_map(self) -> Dict[str, str]:
        """The legacy v1 shape {entry_id: url}, for consumers that still
        want a flat name->url mapping."""
        return {mid: e.url for mid, e in self.entries.items()}

    def visible(self) -> List[ModelEntry]:
        """Non-hidden entries in registry (curated) order."""
        return [e for e in self.entries.values() if not e.hidden]


# --- loading ----------------------------------------------------------------

def load_registry(config_path) -> Registry:
    """Parse config.json (schema v2 or legacy flat v1) into a Registry.

    Raises FileNotFoundError / ValueError exactly like the legacy
    winmol_batch.load_model_paths did, so callers' error handling holds.
    """
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
    """installer.py's historical dest naming: <Key><ext-from-URL>, with
    '.onnx' preferred/fallback."""
    path = url.split("?")[0] if url else ""
    if path.lower().endswith(".onnx"):
        ext = ".onnx"
    else:
        ext = os.path.splitext(path)[1] or ".onnx"
    return f"{name}{ext}"


def _parse_v1(raw, config_path):
    entries = {}
    for name, url in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name in RESERVED_IDS:
            continue    # reserved for the GUI's local-file escape hatch
        u = url if isinstance(url, str) else ""
        entries[name] = ModelEntry(
            id=name, label=name, url=u, file=_v1_filename(name, u))
    if not entries:
        raise ValueError(
            f"No model entries found in {config_path}. "
            "Expected {name: url}.")
    return Registry(entries, schema=1)


def _parse_recommended(raw, entries, gui_default):
    """Validate the ranked default list. Absent -> gui_default alone."""
    recommended = raw.get("recommended")
    if recommended is None:
        # Older v2 registries: the single gui_default IS the ranking.
        return [gui_default] if gui_default else []
    if not isinstance(recommended, list):
        raise ValueError("'recommended' must be a list of model ids")
    for rid in recommended:
        if rid not in entries:
            raise ValueError(
                f"recommended references unknown model {rid!r}")
    if len(set(recommended)) != len(recommended):
        raise ValueError("'recommended' has duplicate ids")
    # One default, declared once: the ranked head and gui_default cannot
    # disagree about what the GUI opens on.
    if recommended and gui_default and recommended[0] != gui_default:
        raise ValueError(
            f"recommended[0] ({recommended[0]!r}) must equal "
            f"gui_default ({gui_default!r})")
    return recommended


def _parse_v2(raw, config_path):
    tile_px = int(raw.get("tile_px", 512))
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"schema-2 registry without models: {config_path}")

    entries = {}
    for mid, spec in models.items():
        if mid in RESERVED_IDS:
            raise ValueError(
                f"model id {mid!r} is reserved (GUI escape hatch): "
                f"{config_path}")
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
            description=str(spec.get("description", "")),
            family=str(spec.get("family", "")),
            backend=str(spec.get("backend", "any")),
            precision=str(spec.get("precision", "fp32")),
            format=str(spec.get("format", "onnx")),
            url=url,
            file=file.strip(),
            sha256=spec.get("sha256") or None,
            md5=spec.get("md5") or None,
            size_mb=spec.get("size_mb"),
            f1=spec.get("f1"),
            lossless=bool(spec.get("lossless", True)),
            hidden=bool(spec.get("hidden", False)),
            tile_px=int(spec.get("tile_px", tile_px)),
        )

    families = {}
    for fid, spec in (raw.get("families") or {}).items():
        if fid in RESERVED_IDS:
            raise ValueError(f"family id {fid!r} is reserved")
        default = spec.get("default")
        fam = Family(id=fid, label=str(spec.get("label", fid)),
                     default=default, cpu=spec.get("cpu"),
                     gpu=spec.get("gpu"))
        for ref in (fam.default, fam.cpu, fam.gpu):
            if ref is not None and ref not in entries:
                raise ValueError(
                    f"family {fid!r} references unknown model {ref!r}")
        if fam.default is None:
            raise ValueError(f"family {fid!r} has no default model")
        families[fid] = fam

    preload = raw.get("preload") or []
    for pid in preload:
        if pid not in entries:
            raise ValueError(f"preload references unknown model {pid!r}")

    gui_default = raw.get("gui_default")
    if gui_default is not None and gui_default not in entries:
        raise ValueError(
            f"gui_default references unknown model {gui_default!r}")
    recommended = _parse_recommended(raw, entries, gui_default)

    return Registry(entries, families, schema=int(raw["schema"]),
                    gui_default=gui_default, preload=preload,
                    tile_px=tile_px, recommended=recommended)


# --- paths / device ---------------------------------------------------------

def local_path(entry, model_dir) -> str:
    """The single on-disk naming rule shared by plugin and CLI."""
    return os.path.join(model_dir, entry.file)


def detect_device() -> str:
    """"gpu" or "cpu". WINMOL_DEVICE env overrides; else an nvidia-smi
    probe (same pattern as winmol_batch.detect_gpu_count). No
    onnxruntime import — the QGIS process must not need it."""
    forced = os.environ.get("WINMOL_DEVICE", "").strip().lower()
    if forced in ("gpu", "cuda"):
        return "gpu"
    if forced == "cpu":
        return "cpu"
    if "probe" not in _DEVICE_PROBE_CACHE:
        _DEVICE_PROBE_CACHE["probe"] = _probe_nvidia()
    return _DEVICE_PROBE_CACHE["probe"]


def _probe_nvidia() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and any(
                ln.strip() for ln in out.stdout.splitlines()):
            return "gpu"
    except Exception:
        pass
    return "cpu"


# --- integrity verification -------------------------------------------------

def _expected_digest(entry):
    """(algo, lowercase hexdigest) or (None, None) when unpinned."""
    if entry.sha256:
        return "sha256", entry.sha256.lower()
    if entry.md5:
        return "md5", entry.md5.lower()
    return None, None


def _hash_file(path, algo) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _cache_load(cache_dir) -> dict:
    try:
        with open(os.path.join(cache_dir, VERIFIED_CACHE)) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _cache_store(cache_dir, filename, size, mtime, algo, digest):
    data = _cache_load(cache_dir)
    data[filename] = {"size": size, "mtime": mtime,
                      "algo": algo, "digest": digest}
    try:
        with open(os.path.join(cache_dir, VERIFIED_CACHE), "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
    except OSError:
        pass    # cache is an optimization only


def verify_file(path, sha256=None, md5=None, cache_dir=None) -> bool:
    """True when the file matches the given checksum (or none is given —
    legacy behavior). A passing digest is memoized per (size, mtime) in
    ``<dir>/.winmol_verified.json`` so a 374 MB model is hashed once,
    not on every run."""
    if sha256:
        algo, expected = "sha256", sha256.lower()
    elif md5:
        algo, expected = "md5", md5.lower()
    else:
        return True
    if not os.path.exists(path):
        return False
    st = os.stat(path)
    cache_dir = cache_dir or os.path.dirname(path) or "."
    rec = _cache_load(cache_dir).get(os.path.basename(path))
    if (rec and rec.get("size") == st.st_size
            and rec.get("mtime") == st.st_mtime
            and rec.get("algo") == algo
            and rec.get("digest") == expected):
        return True
    ok = _hash_file(path, algo) == expected
    if ok:
        _cache_store(cache_dir, os.path.basename(path),
                     st.st_size, st.st_mtime, algo, expected)
    return ok


# --- download ---------------------------------------------------------------

def _urllib_fetcher(url, tmp_path, progress, timeout):
    """Stream ``url`` to ``tmp_path`` in 1 MiB chunks, hashing on the
    fly (no second pass over a 374 MB file). Returns {algo: hexdigest}.
    ``progress`` is called as progress(bytes_done, total_or_None). The
    timeout applies per socket operation — a real bound, unlike the old
    urlretrieve."""
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "WINMOL-Analyzer"})
    hashes = {"sha256": hashlib.sha256(), "md5": hashlib.md5()}
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
                for h in hashes.values():
                    h.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    return {name: h.hexdigest() for name, h in hashes.items()}


#: Injectable for offline tests (monkeypatch this, or pass fetcher=).
_DEFAULT_FETCHER = _urllib_fetcher


def _discard(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def download_model(entry, dest_dir, progress=None, fetcher=None,
                   timeout=30.0) -> str:
    """Atomic verified fetch: stream to ``<file>.part``, check the
    entry's sha256/md5, then os.replace onto the final name. A crash or
    kill can only ever leave a ``*.part`` file — never a truncated model
    at the final path. Raises ModelDownloadError on any failure.

    ``progress``: callable(bytes_done, total_or_None, entry).
    ``fetcher``: callable(url, tmp_path, progress2, timeout) writing the
    payload to tmp_path; may return {algo: hexdigest} to skip re-hash.
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
        digests = fetch(entry.url, tmp, inner, timeout)
        algo, expected = _expected_digest(entry)
        if expected is not None:
            actual = None
            if isinstance(digests, dict):
                actual = digests.get(algo)
            if actual is None:
                actual = _hash_file(tmp, algo)
            if actual.lower() != expected:
                raise ModelDownloadError(
                    f"checksum mismatch for {entry.file}: expected "
                    f"{algo} {expected}, got {actual} — corrupt or "
                    "tampered download, file discarded",
                    model_id=entry.id, url=entry.url)
        os.replace(tmp, dest)
        if expected is not None:
            st = os.stat(dest)
            _cache_store(dest_dir, entry.file, st.st_size, st.st_mtime,
                         algo, expected)
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
                 fetcher=None, allow_download=True) -> str:
    """The call-site API: return a verified local path for a model,
    downloading it if needed.

    An existing file that matches its checksum (or has none — legacy) is
    returned as-is. An existing file that FAILS its checksum is treated
    as stale/truncated and re-downloaded — closing the old size>0-only
    hole where a corrupt file passed forever. With
    ``allow_download=False`` a missing/stale file raises
    ModelDownloadError instead."""
    if isinstance(name_or_entry, ModelEntry):
        entry = name_or_entry
    else:
        if registry is None:
            raise ValueError(
                "registry is required when passing a model name")
        entry = registry.resolve(name_or_entry)
    path = local_path(entry, model_dir)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        if verify_file(path, entry.sha256, entry.md5,
                       cache_dir=model_dir):
            return path
        if not allow_download:
            raise ModelDownloadError(
                f"{path} exists but fails {entry.id} checksum "
                "verification, and downloads are disabled",
                model_id=entry.id, url=entry.url)
    elif not allow_download:
        raise ModelDownloadError(
            f"model file missing: {path} (downloads disabled; fetch "
            f"{entry.url} manually)",
            model_id=entry.id, url=entry.url)
    return download_model(entry, model_dir, progress=progress,
                          fetcher=fetcher)
