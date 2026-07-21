"""Contract tests for plugin_utils/model_registry.py — the model-zoo
registry (schema v2 config.json), the resolver (family/variant/device),
and the verified atomic downloader.

Entirely offline: every download goes through an injected fake fetcher;
no test constructs a socket. The shipped repo config.json is validated
against the ground-truth checksums of the models-v1 zoo release
(github.com/cwinkelmann/WINMOL_segmentor_pt) and Zenodo record 15907576.
"""

import hashlib
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plugin_utils import model_registry as mr   # noqa: E402
import plugin_utils.installer as inst           # noqa: E402
import winmol_batch as wb                       # noqa: E402

SHIPPED_CONFIG = os.path.join(REPO, "config.json")

# Ground truth: SHA256SUMS of the models-v1 release of
# github.com/cwinkelmann/WINMOL_segmentor_pt (22 assets), vendored
# verbatim at tests/fixtures/models_v1_SHA256SUMS so this file never
# drifts from the release again (the 2026-07-21 asset rename made every
# hardcoded name here stale while the digests stayed valid).
ZOO_SUMS_FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures", "models_v1_SHA256SUMS")


def _parse_sha256sums(path):
    """Parse a coreutils SHA256SUMS file -> {filename: sha256}."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = digest
    return out


ZOO_SHA256 = _parse_sha256sums(ZOO_SUMS_FIXTURE)

# Ground truth: md5 of the four original Keras models, Zenodo record
# 15907576 (DOI 10.5281/zenodo.15907576).
ZENODO_MD5 = {
    "model_UNet_GenDS_512_2023-02-27_211141.hdf5":
        "19dd5cdd00f4ea47cddfee5fba3f55ba",
    "model_UNet_SpecDS_Beech_512_2023-02-28_042751.hdf5":
        "6684ca1ab0663ff388fc441fe70b820a",
    "model_UNet_SpecDS_Spruce_512_2023-02-27_061925.hdf5":
        "f44d26b6a4181d0853e48907703a1926",
    "model_UNet_SpecDS_Spruce_Deadwood_512_2024-12-19_194758.hdf5":
        "64998766801caac8ad560b5bba547e93",
}

CLASSIC = ("General", "Beech", "Spruce", "Spruce_Deadwood")

#: models-v1 assets the classic ids were repointed onto, same order.
CLASSIC_ASSETS = (
    "model_UNet_GenDS_512_2023-02-27_211141.onnx",
    "model_UNet_SpecDS_Beech_512_2023-02-28_042751.onnx",
    "model_UNet_SpecDS_Spruce_512_2023-02-27_061925.onnx",
    "model_UNet_SpecDS_Spruce_Deadwood_512_2024-12-19_194758.onnx",
)


def _entry(tmp_path, data=b"DATA", checksum=True, **kw):
    """A minimal ModelEntry whose sha256 matches `data` (or has none)."""
    kw.setdefault("id", "X")
    kw.setdefault("label", "X test model")
    kw.setdefault("url", "https://example.invalid/x.onnx")
    kw.setdefault("file", "x.onnx")
    if checksum:
        kw.setdefault("sha256", hashlib.sha256(data).hexdigest())
    return mr.ModelEntry(**kw)


def _writer(data):
    """A fake fetcher that writes `data` to the tmp path."""
    def fetch(url, tmp, progress, timeout):
        with open(tmp, "wb") as f:
            f.write(data)
        if progress is not None:
            progress(len(data), len(data))
    return fetch


def _boom(*_a, **_k):
    raise AssertionError("fetcher must not be called")


# --- shipped registry --------------------------------------------------------

def test_load_registry_v2_shipped():
    reg = mr.load_registry(SHIPPED_CONFIG)
    assert reg.schema == 2
    assert len(reg.entries) == 26
    assert reg.preload == []
    assert "Custom" not in reg.entries
    assert reg.gui_default == "Spruce_Deadwood_int8"

    # The four classic public IDS are unchanged (stable API), but their
    # assets were repointed from the unchecksummed models-onnx-v1
    # release to the sha256-pinned, numerically-identical models-v1
    # conversions of the same Keras weights.
    for name, asset in zip(CLASSIC, CLASSIC_ASSETS):
        e = reg.entries[name]
        assert e.file == asset
        assert "models-v1" in e.url and "models-onnx-v1" not in e.url
        assert e.url.endswith(f"/{asset}")
        assert e.sha256 == ZOO_SHA256[asset]

    # Every family reference resolves to a real entry.
    for fam in reg.families.values():
        assert fam.default in reg.entries
        for ref in (fam.cpu, fam.gpu):
            assert ref is None or ref in reg.entries

    # Zoo assets carry the ground-truth sha256 from SHA256SUMS.
    zoo = [e for e in reg.entries.values()
           if "WINMOL_segmentor_pt" in e.url]
    assert len(zoo) == 22
    for e in zoo:
        assert e.sha256 == ZOO_SHA256[e.file], e.id

    # The hdf5 originals point at Zenodo record 15907576 with known md5.
    hdf5 = [e for e in reg.entries.values() if e.format == "hdf5"]
    assert len(hdf5) == 4
    for e in hdf5:
        assert "records/15907576" in e.url
        assert e.md5 == ZENODO_MD5[e.file], e.id
        assert e.hidden          # plugin venv is ONNX-only

    # Labels/descriptions tell the user which model is which.
    for e in reg.entries.values():
        assert e.label and e.label != e.id
        assert e.description


def test_load_registry_v1_flat(tmp_path):
    cfg = tmp_path / "config.json"
    flat = {
        "Have": "https://example.invalid/some-other-name.onnx",
        "Legacy": "https://example.invalid/model.hdf5?download=1",
    }
    cfg.write_text(json.dumps(flat))
    reg = mr.load_registry(str(cfg))
    assert reg.schema == 1
    # v1 keeps installer's key-based dest naming (<Key><ext-from-url>),
    # even when the URL basename differs.
    assert reg.entries["Have"].file == "Have.onnx"
    assert reg.entries["Legacy"].file == "Legacy.hdf5"
    # flat_map round-trips the original mapping.
    assert reg.flat_map() == flat


def test_load_registry_raises_like_legacy(tmp_path):
    with pytest.raises(FileNotFoundError):
        mr.load_registry(str(tmp_path / "nope.json"))
    empty = tmp_path / "config.json"
    empty.write_text("{}")
    with pytest.raises(ValueError):
        mr.load_registry(str(empty))


def test_reserved_custom(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "schema": 2,
        "models": {"Custom": {
            "url": "https://example.invalid/x.onnx", "file": "x.onnx"}},
    }))
    with pytest.raises(ValueError):
        mr.load_registry(str(cfg))


# --- integrity: no unverifiable download ------------------------------------

def test_every_entry_has_a_digest():
    """REGRESSION GUARD. Every downloadable entry in the shipped
    registry must carry a sha256 (zoo assets) or md5 (Zenodo hdf5), so
    no model can ever be installed without integrity verification.
    Adding an entry without one fails here."""
    reg = mr.load_registry(SHIPPED_CONFIG)
    unpinned = reg.unpinned()
    assert unpinned == [], (
        "unverifiable model download(s): "
        + ", ".join(f"{e.id} ({e.url})" for e in unpinned))
    # and the digests are real hex of the right width
    for e in reg.entries.values():
        algo, digest = mr._expected_digest(e)
        assert algo in ("sha256", "md5"), e.id
        assert len(digest) == (64 if algo == "sha256" else 32), e.id
        int(digest, 16)          # raises unless pure hex


def test_digest_is_actually_enforced_for_md5_entries(tmp_path):
    """md5 (not just sha256) must really be verified — the Zenodo
    originals are md5-only."""
    entry = mr.ModelEntry(
        id="H", label="hdf5", url="https://example.invalid/m.hdf5",
        file="m.hdf5", md5=hashlib.md5(b"GOOD").hexdigest())
    path = mr.ensure_model(entry, str(tmp_path), fetcher=_writer(b"GOOD"))
    assert open(path, "rb").read() == b"GOOD"
    os.remove(path)
    with pytest.raises(mr.ModelDownloadError) as exc:
        mr.ensure_model(entry, str(tmp_path), fetcher=_writer(b"BAD"))
    assert "checksum mismatch" in str(exc.value)
    assert "md5" in str(exc.value)
    assert not os.path.exists(path)          # nothing left behind


# --- recommended ranking / device-aware default ------------------------------

def test_recommended_ranking_shipped():
    """The default and its runner-up are explicit and ordered, not
    implied by dict order."""
    reg = mr.load_registry(SHIPPED_CONFIG)
    assert reg.recommended == ["Spruce_Deadwood_int8", "UNet_PT_int8"]
    assert reg.recommended[0] == reg.gui_default
    first, second = reg.recommended_entries()
    # 1st: INT8 Spruce + deadwood (SpecDS), from the models-v1 release
    assert first.id == "Spruce_Deadwood_int8"
    assert first.file == ("model_UNet_SpecDS_Spruce_Deadwood_512"
                          "_2024-12-19_194758_int8.onnx")
    assert first.precision == "int8"
    assert first.size_mb == 31.4
    assert first.sha256 == ZOO_SHA256[first.file]
    # 2nd: the PyTorch UNet w05 int8 — the real asset behind the
    # "SpecDS INT8 W05" shorthand; label/description must be explicit
    # that w05 is the PyTorch retrain, not a classic Keras variant.
    assert second.id == "UNet_PT_int8"
    assert second.file == ("model_UNet_SpecDS_Beech_512"
                           "_pytorch_w05_int8.onnx")
    assert second.size_mb == 7.9
    assert second.f1 == 0.76
    assert "w05" in second.label.lower()
    assert "specds" in second.description.lower()   # names the confusion
    assert second.family == "unet_pt" and first.family != second.family


def test_default_entry_is_device_aware(monkeypatch):
    """int8 on CPU, fp16 on GPU — the Spruce+Deadwood domain (the
    user's declared first choice) is preserved either way."""
    reg = mr.load_registry(SHIPPED_CONFIG)

    monkeypatch.setattr(mr, "detect_device", lambda: "cpu")
    e = reg.default_entry()                  # device="auto" -> stub
    assert e.id == "Spruce_Deadwood_int8"
    assert e.precision == "int8"
    assert e.family == "classic_spruce_deadwood"

    monkeypatch.setattr(mr, "detect_device", lambda: "gpu")
    e = reg.default_entry()
    assert e.id == "Spruce_Deadwood_fp16"
    assert e.precision == "fp16"
    assert e.family == "classic_spruce_deadwood"

    # explicit device argument wins over detection
    monkeypatch.setattr(mr, "detect_device", _boom)
    assert reg.default_entry(device="cpu").id == "Spruce_Deadwood_int8"
    assert reg.default_entry(device="gpu").id == "Spruce_Deadwood_fp16"


def test_default_entry_never_downloads_or_hits_network(monkeypatch,
                                                       tmp_path):
    """Resolving the default must not fetch anything, and must not
    shell out to nvidia-smi when the device is given."""
    reg = mr.load_registry(SHIPPED_CONFIG)
    monkeypatch.setattr(mr, "_DEFAULT_FETCHER", _boom)
    monkeypatch.setattr(subprocess, "run", _boom)
    for device in ("cpu", "gpu"):
        e = reg.default_entry(device=device)
        assert not os.path.exists(mr.local_path(e, str(tmp_path)))
    # WINMOL_DEVICE short-circuits the probe too (still no subprocess)
    monkeypatch.setenv("WINMOL_DEVICE", "gpu")
    mr._DEVICE_PROBE_CACHE.pop("probe", None)
    assert reg.default_entry().id == "Spruce_Deadwood_fp16"
    monkeypatch.setenv("WINMOL_DEVICE", "cpu")
    assert reg.default_entry().id == "Spruce_Deadwood_int8"


def test_explicit_selection_overrides_device_default(monkeypatch):
    """The device rule applies to the DEFAULT only: an explicit entry
    id or a forced variant is never rewritten."""
    reg = mr.load_registry(SHIPPED_CONFIG)
    monkeypatch.setattr(mr, "detect_device", lambda: "gpu")
    assert reg.default_entry().id == "Spruce_Deadwood_fp16"
    # user explicitly asks for the fp32 classic -> untouched
    assert reg.resolve("Spruce_Deadwood").id == "Spruce_Deadwood"
    # ...or for int8 on that GPU box -> honoured
    assert (reg.resolve("classic_spruce_deadwood", device="gpu",
                        variant="int8").id == "Spruce_Deadwood_int8")
    # ...or a different family entirely
    assert reg.resolve("HRNet_Beech").id == "HRNet_Beech"


def test_recommended_validation(tmp_path):
    """A registry whose ranking disagrees with gui_default, or names an
    unknown id, is rejected at load time."""
    base = json.load(open(SHIPPED_CONFIG))

    bad = dict(base, recommended=["UNet_PT_int8", "Spruce_Deadwood_int8"])
    cfg = tmp_path / "disagree.json"
    cfg.write_text(json.dumps(bad))
    with pytest.raises(ValueError) as exc:
        mr.load_registry(str(cfg))
    assert "gui_default" in str(exc.value)

    bad = dict(base, recommended=["NoSuchModel"], gui_default="NoSuchModel")
    cfg = tmp_path / "unknown.json"
    cfg.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        mr.load_registry(str(cfg))

    # absent 'recommended' -> gui_default is the ranking (back-compat)
    ok = {k: v for k, v in base.items() if k != "recommended"}
    cfg = tmp_path / "norec.json"
    cfg.write_text(json.dumps(ok))
    reg = mr.load_registry(str(cfg))
    assert reg.recommended == ["Spruce_Deadwood_int8"]
    assert reg.default_entry(device="cpu").id == "Spruce_Deadwood_int8"


# --- resolution --------------------------------------------------------------

def test_resolve_explicit_id_never_rewritten():
    reg = mr.load_registry(SHIPPED_CONFIG)
    for device in ("cpu", "gpu"):
        e = reg.resolve("General", device=device)
        assert e.id == "General"
        assert e.file == CLASSIC_ASSETS[0]
    # explicit variant ids resolve to themselves too
    assert reg.resolve("UNet_PT_int8", device="gpu").id == "UNet_PT_int8"


def test_resolve_family_auto_substitutes_only_lossless():
    reg = mr.load_registry(SHIPPED_CONFIG)
    # classic int8 is domain-calibrated (lossless: false) -> stays fp32
    assert reg.resolve("classic_general", device="cpu").id == "General"
    # classic fp16 is lossless -> substituted on gpu
    assert (reg.resolve("classic_general", device="gpu").id
            == "General_fp16")
    # PyTorch UNet int8 is certified lossless -> substituted on cpu
    assert reg.resolve("unet_pt", device="cpu").id == "UNet_PT_int8"
    assert reg.resolve("unet_pt", device="gpu").id == "UNet_PT_fp16"
    # deeplab has no cpu variant -> family default
    assert reg.resolve("deeplab", device="cpu").id == "DeepLab_Beech"
    assert reg.resolve("hrnet", device="gpu").id == "HRNet_Beech_fp16"


def test_resolve_forced_variant():
    reg = mr.load_registry(SHIPPED_CONFIG)
    assert (reg.resolve("classic_spruce", device="gpu", variant="fp32").id
            == "Spruce")
    assert (reg.resolve("classic_spruce", device="gpu", variant="int8").id
            == "Spruce_int8")
    with pytest.raises(ValueError):
        reg.resolve("deeplab", device="cpu", variant="int8")


def test_resolve_case_insensitive_and_errors():
    reg = mr.load_registry(SHIPPED_CONFIG)
    assert reg.get("spruce_deadwood").id == "Spruce_Deadwood"
    assert reg.get("unet_pt_int8").id == "UNet_PT_int8"
    # "unet_pt" (exact family id) auto-picks; a case-insensitive match
    # of the ENTRY id "UNet_PT" wins over the family and stays fp32.
    assert reg.resolve("unet_pt", device="cpu").id == "UNet_PT_int8"
    assert reg.resolve("UNET_PT", device="cpu").id == "UNet_PT"
    with pytest.raises(KeyError) as exc:
        reg.get("NoSuchModel")
    assert "UNet_PT" in str(exc.value)   # error names the visible ids


# --- download / verify -------------------------------------------------------

def test_download_atomic_and_verified(tmp_path):
    entry = _entry(tmp_path)
    seen = {}

    def fetch(url, tmp, progress, timeout):
        assert tmp.endswith(".part")
        seen["final_during_fetch"] = os.path.exists(tmp[:-5])
        with open(tmp, "wb") as f:
            f.write(b"DATA")

    path = mr.download_model(entry, str(tmp_path), fetcher=fetch)
    assert path == os.path.join(str(tmp_path), "x.onnx")
    with open(path, "rb") as f:
        assert f.read() == b"DATA"
    # never visible at the final name mid-fetch; no .part left behind
    assert seen["final_during_fetch"] is False
    assert not os.path.exists(path + ".part")
    # digest recorded in the verification cache
    with open(os.path.join(str(tmp_path), mr.VERIFIED_CACHE)) as f:
        cache = json.load(f)
    assert cache["x.onnx"]["digest"] == entry.sha256


def test_download_checksum_mismatch(tmp_path):
    entry = _entry(tmp_path, data=b"GOOD")
    with pytest.raises(mr.ModelDownloadError) as exc:
        mr.download_model(entry, str(tmp_path), fetcher=_writer(b"BAD"))
    assert "checksum mismatch" in str(exc.value)
    assert exc.value.model_id == "X"
    assert not os.path.exists(tmp_path / "x.onnx")
    assert not os.path.exists(tmp_path / "x.onnx.part")


def test_ensure_model_heals_stale_file(tmp_path):
    entry = _entry(tmp_path, data=b"FRESH")
    dest = tmp_path / "x.onnx"
    dest.write_bytes(b"stale-or-truncated")
    path = mr.ensure_model(entry, str(tmp_path), fetcher=_writer(b"FRESH"))
    assert path == str(dest)
    assert dest.read_bytes() == b"FRESH"    # size>0 hole is closed


def test_ensure_model_no_checksum_keeps_existing(tmp_path):
    entry = _entry(tmp_path, checksum=False)
    dest = tmp_path / "x.onnx"
    dest.write_bytes(b"whatever")
    # legacy behavior: no checksum -> any non-empty file is accepted
    path = mr.ensure_model(entry, str(tmp_path), fetcher=_boom)
    assert path == str(dest)
    assert dest.read_bytes() == b"whatever"


def test_ensure_model_no_download(tmp_path):
    entry = _entry(tmp_path, data=b"DATA")
    with pytest.raises(mr.ModelDownloadError):
        mr.ensure_model(entry, str(tmp_path), fetcher=_boom,
                        allow_download=False)
    # present + verified file returns immediately, fetcher never called
    (tmp_path / "x.onnx").write_bytes(b"DATA")
    path = mr.ensure_model(entry, str(tmp_path), fetcher=_boom,
                           allow_download=False)
    assert path == str(tmp_path / "x.onnx")


def test_verify_cache_avoids_rehashing(tmp_path, monkeypatch):
    entry = _entry(tmp_path, data=b"DATA")
    calls = []
    real = mr._hash_file

    def counting(path, algo):
        calls.append(os.path.basename(path))
        return real(path, algo)

    monkeypatch.setattr(mr, "_hash_file", counting)
    mr.ensure_model(entry, str(tmp_path), fetcher=_writer(b"DATA"))
    n_after_download = len(calls)
    assert n_after_download >= 1
    # second call: stat + cache lookup, no re-hash
    mr.ensure_model(entry, str(tmp_path), fetcher=_boom)
    assert len(calls) == n_after_download


def test_progress_callback(tmp_path):
    entry = _entry(tmp_path, checksum=False)
    got = []

    def fetch(url, tmp, progress, timeout):
        chunk = b"abc" * 100
        with open(tmp, "wb") as f:
            for i in range(3):
                f.write(chunk)
                progress((i + 1) * len(chunk), 3 * len(chunk))

    mr.download_model(entry, str(tmp_path), fetcher=fetch,
                      progress=lambda d, t, e: got.append((d, t, e)))
    assert len(got) == 3
    assert [d for d, _t, _e in got] == sorted(d for d, _t, _e in got)
    assert all(e is entry for _d, _t, e in got)
    assert all(t == 900 for _d, t, _e in got)
    # the CLI stderr printer copes with a missing total
    wb._stderr_progress(500, None, entry)
    wb._stderr_progress(500, 1000, entry)
    print(file=sys.stderr)


# --- installer integration ---------------------------------------------------

def test_installer_v2_no_startup_network(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "_DEFAULT_FETCHER", _boom)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "schema": 2,
        "preload": [],
        "models": {"UNet_PT_int8": {
            "label": "UNet PT int8",
            "url": "https://example.invalid/uw05_int8.onnx",
            "file": "uw05_int8.onnx",
        }},
    }))
    # preload=[] -> zero network I/O at startup, nothing missing
    assert inst.download_models(str(tmp_path), str(cfg)) == []

    cfg.write_text(json.dumps({
        "schema": 2,
        "preload": ["UNet_PT_int8"],
        "models": {"UNet_PT_int8": {
            "label": "UNet PT int8",
            "url": "https://example.invalid/uw05_int8.onnx",
            "file": "uw05_int8.onnx",
        }},
    }))
    # a failing fetch is reported, not raised (tolerant contract, v2 too)
    assert inst.download_models(str(tmp_path), str(cfg)) == ["UNet_PT_int8"]


# --- batch CLI ---------------------------------------------------------------

def test_batch_resolution_shipped_v2(tmp_path):
    paths = wb.load_model_paths(config_path=SHIPPED_CONFIG,
                                model_dir=str(tmp_path))
    # classic ids keep working; they now resolve to the models-v1
    # asset basenames they were repointed onto
    for name, asset in zip(CLASSIC, CLASSIC_ASSETS):
        assert paths[name] == str(tmp_path / asset)
    # zoo ids resolve to the release asset basenames
    assert (paths["UNet_PT_int8"]
            == str(tmp_path / "model_UNet_SpecDS_Beech_512"
                              "_pytorch_w05_int8.onnx"))
    # hdf5 originals are addressable from the CLI (TF users)
    assert (paths["Spruce_Deadwood_hdf5"]
            == str(tmp_path
                   / "model_UNet_SpecDS_Spruce_Deadwood_512"
                     "_2024-12-19_194758.hdf5"))


def test_batch_resolution_v1_byte_identical(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "A": "https://h.invalid/some_asset.onnx?download=1",
        "B": "https://h.invalid/sub/B.onnx",
    }))
    paths = wb.load_model_paths(config_path=str(cfg),
                                model_dir="/m")
    # v1 keeps the legacy URL-basename naming (url_to_filename)
    assert paths == {"A": os.path.join("/m", "some_asset.onnx"),
                     "B": os.path.join("/m", "B.onnx")}


def test_batch_no_download_missing_exits_2(tmp_path, capsys):
    rc = wb.main(["UNet_PT_int8", "--no-download",
                  "--model-dir", str(tmp_path),
                  "--input", str(tmp_path)])
    assert rc == 2
    out = capsys.readouterr().out
    # source-specific manual hint for a zoo asset
    assert "WINMOL_segmentor_pt" in out
    assert "model_UNet_SpecDS_Beech_512_pytorch_w05_int8.onnx" in out


def test_batch_lowercase_general_still_resolves(tmp_path, capsys):
    # Dockerfile compat: lowercase names resolve to the canonical id.
    rc = wb.main(["general", "--no-download",
                  "--model-dir", str(tmp_path),
                  "--input", str(tmp_path)])
    assert rc == 2          # file missing, but the NAME resolved
    out = capsys.readouterr().out
    assert "model_UNet_GenDS_512_2023-02-27_211141.onnx" in out
    assert "WINMOL_segmentor_pt" in out   # repointed onto the zoo release


def test_batch_download_on_demand(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mr, "_DEFAULT_FETCHER", _writer(b"not-a-model"))
    # checksum mismatch from the stub -> clean exit 2 with the cause
    rc = wb.main(["UNet_PT_int8",
                  "--model-dir", str(tmp_path),
                  "--input", str(tmp_path)])
    assert rc == 2
    assert "checksum mismatch" in capsys.readouterr().out


def test_batch_list_models(capsys):
    rc = wb.main(["--list-models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "UNet_PT_int8" in out
    assert "unet_pt" in out          # family ids listed too
    assert "General" in out


def test_batch_variant_forced_unavailable_exits_2(tmp_path, capsys):
    rc = wb.main(["deeplab", "--variant", "int8", "--no-download",
                  "--model-dir", str(tmp_path),
                  "--input", str(tmp_path)])
    assert rc == 2
    assert "int8" in capsys.readouterr().out


# --- import safety -----------------------------------------------------------

def test_registry_import_safety():
    code = (
        "import sys; sys.path.insert(0, {repo!r})\n"
        "import plugin_utils.model_registry\n"
        "bad = [m for m in sys.modules\n"
        "       if m.split('.')[0] in ('qgis', 'PyQt5', 'PyQt6')]\n"
        "print(','.join(bad))\n"
    ).format(repo=REPO)
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


# --- on-disk state, verification, removal (the Setup tab's model half) -------

def test_installed_state_missing_and_zero_byte(tmp_path):
    entry = _entry(tmp_path)
    assert mr.installed_state(entry, str(tmp_path)) == "missing"
    (tmp_path / entry.file).write_bytes(b"")
    assert mr.installed_state(entry, str(tmp_path)) == "missing"


def test_installed_state_present_then_verified(tmp_path):
    entry = _entry(tmp_path, data=b"DATA")
    (tmp_path / entry.file).write_bytes(b"DATA")
    # stat-only: a file nobody has hashed yet is 'present', never
    # 'verified' — the tree must not claim a guarantee it has not checked
    assert mr.installed_state(entry, str(tmp_path)) == "present"
    assert mr.verify_entry(entry, str(tmp_path)) is True
    assert mr.installed_state(entry, str(tmp_path)) == "verified"


def test_installed_state_unpinned_is_not_verified(tmp_path):
    entry = _entry(tmp_path, checksum=False)
    (tmp_path / entry.file).write_bytes(b"whatever")
    assert mr.installed_state(entry, str(tmp_path)) == "unpinned"
    # verify_file() would say True for it; the state must not
    assert mr.verify_file(str(tmp_path / entry.file)) is True


def test_installed_state_corrupt_after_a_failed_verify(tmp_path):
    entry = _entry(tmp_path, data=b"DATA")
    (tmp_path / entry.file).write_bytes(b"TRUNCATED")
    assert mr.verify_entry(entry, str(tmp_path)) is False
    assert mr.installed_state(entry, str(tmp_path)) == "corrupt"
    # and a failure memo can never be mistaken for a pass
    assert mr.verify_file(str(tmp_path / entry.file),
                          sha256=entry.sha256,
                          cache_dir=str(tmp_path)) is False


def test_installed_state_forgets_a_replaced_file(tmp_path):
    entry = _entry(tmp_path, data=b"DATA")
    (tmp_path / entry.file).write_bytes(b"TRUNCATED")
    mr.verify_entry(entry, str(tmp_path))
    assert mr.installed_state(entry, str(tmp_path)) == "corrupt"
    os.utime(str(tmp_path / entry.file), (1, 1))
    (tmp_path / entry.file).write_bytes(b"DATA")
    # different size/mtime -> the stale verdict must not stick
    assert mr.installed_state(entry, str(tmp_path)) == "present"


def test_verify_entry_reports_byte_progress(tmp_path):
    data = b"D" * (3 * mr._CHUNK_BYTES + 17)
    entry = _entry(tmp_path, data=data)
    (tmp_path / entry.file).write_bytes(data)
    seen = []
    assert mr.verify_entry(entry, str(tmp_path),
                           progress=lambda d, t: seen.append((d, t))) is True
    assert seen[-1] == (len(data), len(data))
    assert len(seen) >= 4 and all(t == len(data) for _d, t in seen)


def test_verify_entry_without_a_digest_is_vacuously_true(tmp_path):
    entry = _entry(tmp_path, checksum=False)
    (tmp_path / entry.file).write_bytes(b"x")
    assert mr.verify_entry(entry, str(tmp_path)) is True
    # ...and writes no memo that could later read as 'verified'
    assert mr.installed_state(entry, str(tmp_path)) == "unpinned"


def test_verify_entry_on_a_missing_file(tmp_path):
    assert mr.verify_entry(_entry(tmp_path), str(tmp_path)) is False


def test_remove_model_takes_the_part_file_and_the_memo(tmp_path):
    entry = _entry(tmp_path, data=b"DATA")
    (tmp_path / entry.file).write_bytes(b"DATA")
    (tmp_path / (entry.file + ".part")).write_bytes(b"XX")
    mr.verify_entry(entry, str(tmp_path))
    assert entry.file in mr._cache_load(str(tmp_path))

    freed = mr.remove_model(entry, str(tmp_path))
    assert freed == 6
    assert not (tmp_path / entry.file).exists()
    assert not (tmp_path / (entry.file + ".part")).exists()
    # nothing else in the tree prunes this cache; a re-download of a
    # same-sized file would otherwise inherit the old verdict
    assert entry.file not in mr._cache_load(str(tmp_path))
    assert mr.installed_state(entry, str(tmp_path)) == "missing"


def test_remove_model_prunes_the_memo_under_the_key_verify_file_wrote(
        tmp_path):
    """verify_file() keys the memo by basename(local_path); remove_model()
    keyed it by entry.file. Those coincide for every entry shipped today
    and diverge silently the moment an entry.file carries a subdirectory —
    leaving a stale "verified" verdict behind for the next download."""
    entry = _entry(tmp_path, data=b"DATA", file=os.path.join("sub",
                                                             "x.onnx"))
    (tmp_path / "sub").mkdir()
    (tmp_path / entry.file).write_bytes(b"DATA")
    mr.verify_entry(entry, str(tmp_path))
    assert "x.onnx" in mr._cache_load(str(tmp_path))

    mr.remove_model(entry, str(tmp_path))
    assert "x.onnx" not in mr._cache_load(str(tmp_path)), (
        "the memo survived the deletion under its real key")


def test_remove_model_absent_is_zero_not_an_error(tmp_path):
    assert mr.remove_model(_entry(tmp_path), str(tmp_path)) == 0


def test_remove_all(tmp_path):
    reg = mr.load_registry(SHIPPED_CONFIG)
    models = tmp_path / "models"
    models.mkdir()
    first, second = list(reg.entries.values())[:2]
    (models / first.file).write_bytes(b"a" * 100)
    (models / second.file).write_bytes(b"b" * 50)
    (models / (second.file + ".part")).write_bytes(b"b" * 5)
    stranger = models / "my_own_model.onnx"
    stranger.write_bytes(b"keep me")

    dry = mr.remove_all(reg, str(models), dry_run=True)
    assert dry["freed_bytes"] == 155
    assert dry["removed"] == []
    assert (models / first.file).exists()

    done = mr.remove_all(reg, str(models))
    assert done["freed_bytes"] == 155
    assert not (models / first.file).exists()
    assert not (models / (second.file + ".part")).exists()
    assert done["failed"] == []
    # a file the registry does not know about is never touched
    assert stranger.exists()
    assert not (models / mr.VERIFIED_CACHE).exists()
