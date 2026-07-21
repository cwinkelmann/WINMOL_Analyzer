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
# github.com/cwinkelmann/WINMOL_segmentor_pt (22 assets).
ZOO_SHA256 = {
    "unet_fp32.onnx":
        "e12653e519914dd4cea73517eb414c49061973589c271f0d47696702aea29cc6",
    "unet_w05_int8_cpu.onnx":
        "74c1bd1d8ed31988df5756c9ca7fc4c300de804a1733dc13df5a0bd8fceaaabd",
    "unet_w05_fp16_gpu.onnx":
        "71c47dda22b27c6918aaebd5edd06058e085f75e541c47ca1a60ebe4ec9e1825",
    "unet_rkeras_beech_512.onnx":
        "510bbd31e190d8b8836761f8b8b5af95acca73974585b0a27217ec3c47b5594e",
    "unet_rkeras_beech_512_fp16.onnx":
        "3529db039eb190113a1991fa699531bed48f1021be2efa764e212cdb3b9d5a4d",
    "unet_rkeras_beech_512_int8.onnx":
        "573bd0e09dac0aee0c48c4bf76b93b175d28a8b937e09c39d07e6b9f54559478",
    "deeplabv3plus_fp32.onnx":
        "6e37d38208d46b12d295f1c7c082b289a15455be19b1d85724a35e2a5d46ce4d",
    "deeplabv3plus_fp16_gpu.onnx":
        "af1c296d82b8f9b2bbce467b8030a3136733b304e13330182c1ade76fa339b15",
    "hrnet_fp32.onnx":
        "403b43d1ed793f0aabbad132c268104db4a7b574ebe7f4b39d6dc4b9c85976a9",
    "hrnet_fp16_gpu.onnx":
        "5e16ad145ab97642a4f0d981249fd3a2ca57fadee9a0fcbf23b784e4d4d1a752",
    "model_UNet_GenDS_512.onnx":
        "fa192f96eef750a9ebdc261983b74e0353b2b3f1f746a026655cabbe8f44e0f0",
    "model_UNet_GenDS_512_fp16.onnx":
        "8d8f2d303ff425983102d83dcac860f627c7a6898f0ab46804aa95cd2faf3dd5",
    "model_UNet_GenDS_512_int8.onnx":
        "507bb3da447de288d7a55a86c322fe6e547a28fdb7aca1cd3768745ef1e2060d",
    "model_UNet_SpecDS_Beech_512.onnx":
        "91497501aa4c303838569d17d62875dc1e6e1fa18dcb6086dfd9647e3dc34ca4",
    "model_UNet_SpecDS_Beech_512_fp16.onnx":
        "42b8e369769052c48e16e4d305be15fe7fa211ec6d09f5b0f4fcbdedf21eb310",
    "model_UNet_SpecDS_Beech_512_int8.onnx":
        "a19501ea98e04e695d89cad55590f99225c927c749b6b3df610ca95f02ab8289",
    "model_UNet_SpecDS_Spruce_512.onnx":
        "26546a66b64247961c8f2df07019d846297b1296f223755e405709c111c855fd",
    "model_UNet_SpecDS_Spruce_512_fp16.onnx":
        "2f25fdc70b23e23d87ddf56f8f3a481c8db3d4d046491a27d6e4ab2a2cb5824a",
    "model_UNet_SpecDS_Spruce_512_int8.onnx":
        "e6d22f398977b14b43e8ee5ae8468d378489d6c4d181468c9777eeece1cf733b",
    "model_UNet_SpecDS_Spruce_Deadwood_512.onnx":
        "88028ece8e24a64f7b08727267f933e3d62e12dfbf76d6667d35fa76cfaf83f1",
    "model_UNet_SpecDS_Spruce_Deadwood_512_fp16.onnx":
        "cadee59388d161e6bc64477cc7c91ef8211c3c6c5186b7747aca88dbc28f1e9d",
    "model_UNet_SpecDS_Spruce_Deadwood_512_int8.onnx":
        "e95e7876b38dd356faa789a0be1ae59280a42279db5bb507552771540cd13f46",
}

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
    assert reg.gui_default == "Spruce_Deadwood"

    # The four classic public ids are unchanged: same on-disk names, same
    # models-onnx-v1 release URLs — existing caches stay valid.
    for name in CLASSIC:
        e = reg.entries[name]
        assert e.file == f"{name}.onnx"
        assert "models-onnx-v1" in e.url
        assert e.url.endswith(f"/{name}.onnx")

    # Every family reference resolves to a real entry.
    for fam in reg.families.values():
        assert fam.default in reg.entries
        for ref in (fam.cpu, fam.gpu):
            assert ref is None or ref in reg.entries

    # Zoo assets carry the ground-truth sha256 from SHA256SUMS.
    zoo = [e for e in reg.entries.values()
           if "WINMOL_segmentor_pt" in e.url]
    assert len(zoo) >= 14
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


# --- resolution --------------------------------------------------------------

def test_resolve_explicit_id_never_rewritten():
    reg = mr.load_registry(SHIPPED_CONFIG)
    for device in ("cpu", "gpu"):
        e = reg.resolve("General", device=device)
        assert e.id == "General"
        assert e.file == "General.onnx"
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
            "url": "https://example.invalid/unet_w05_int8_cpu.onnx",
            "file": "unet_w05_int8_cpu.onnx",
        }},
    }))
    # preload=[] -> zero network I/O at startup, nothing missing
    assert inst.download_models(str(tmp_path), str(cfg)) == []

    cfg.write_text(json.dumps({
        "schema": 2,
        "preload": ["UNet_PT_int8"],
        "models": {"UNet_PT_int8": {
            "label": "UNet PT int8",
            "url": "https://example.invalid/unet_w05_int8_cpu.onnx",
            "file": "unet_w05_int8_cpu.onnx",
        }},
    }))
    # a failing fetch is reported, not raised (tolerant contract, v2 too)
    assert inst.download_models(str(tmp_path), str(cfg)) == ["UNet_PT_int8"]


# --- batch CLI ---------------------------------------------------------------

def test_batch_resolution_shipped_v2(tmp_path):
    paths = wb.load_model_paths(config_path=SHIPPED_CONFIG,
                                model_dir=str(tmp_path))
    # classic ids keep today's on-disk names
    for name in CLASSIC:
        assert paths[name] == str(tmp_path / f"{name}.onnx")
    # zoo ids resolve to the release asset basenames
    assert (paths["UNet_PT_int8"]
            == str(tmp_path / "unet_w05_int8_cpu.onnx"))
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
    assert "unet_w05_int8_cpu.onnx" in out


def test_batch_lowercase_general_still_resolves(tmp_path, capsys):
    # Dockerfile compat: lowercase names resolve to the canonical id.
    rc = wb.main(["general", "--no-download",
                  "--model-dir", str(tmp_path),
                  "--input", str(tmp_path)])
    assert rc == 2          # file missing, but the NAME resolved
    out = capsys.readouterr().out
    assert "General.onnx" in out
    assert "models-onnx-v1" in out   # classic hint, not the zoo hint


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
