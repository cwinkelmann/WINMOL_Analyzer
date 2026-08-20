"""Off-QGIS tests for plugin_utils.model_registry: schema-2 parsing,
device->variant resolution, checksum-verified atomic downloads, and the
v1-flat fallback. Mostly in-test fixtures — no network — plus a section
pinning the real, shipped config.json (schema-2, models-v1 release).
"""
import hashlib
import json
import os

import pytest

from plugin_utils import model_registry as mr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_CONFIG = os.path.join(REPO_ROOT, "config.json")

FIXTURE_V2 = {
    "schema": 2,
    "gui_default": "spruce_int8",
    "recommended": ["spruce_int8", "spruce_fp32"],
    "families": {
        "spruce": {"label": "Spruce", "default": "spruce_fp32"},
        "beech": {"label": "Beech", "default": "beech_fp32"},
    },
    "models": {
        "spruce_fp32": {
            "label": "Spruce (fp32)", "family": "spruce",
            "precision": "fp32",
            "url": "https://example.com/spruce_fp32.onnx",
            "file": "spruce_fp32.onnx", "sha256": "a" * 64,
            "size_mb": 12.3,
        },
        # int8: quantised, NOT certified lossless (like the shipped
        # Keras-family int8 builds) -> variant="auto" must refuse it.
        "spruce_int8": {
            "label": "Spruce (int8)", "family": "spruce",
            "precision": "int8",
            "url": "https://example.com/spruce_int8.onnx",
            "file": "spruce_int8.onnx", "sha256": "b" * 64,
            "size_mb": 3.1, "lossless": False,
        },
        # fp16: certified lossless -> variant="auto" may substitute it.
        "spruce_fp16": {
            "label": "Spruce (fp16)", "family": "spruce",
            "precision": "fp16",
            "url": "https://example.com/spruce_fp16.onnx",
            "file": "spruce_fp16.onnx", "sha256": "c" * 64,
            "size_mb": 6.2, "lossless": True,
        },
        # beech has ONLY the fp32 default -> cpu/gpu must fall back to it.
        "beech_fp32": {
            "label": "Beech (fp32)", "family": "beech",
            "precision": "fp32",
            "url": "https://example.com/beech_fp32.onnx",
            "file": "beech_fp32.onnx", "sha256": "d" * 64,
            "size_mb": 12.3,
        },
    },
}


def _registry():
    return mr._parse_v2(FIXTURE_V2, "<fixture>")


def assert_all_pinned(registry):
    """No-unverifiable-download property: every entry carries a sha256.
    Task 2 points this same helper at the real config.json."""
    unpinned = [e.id for e in registry.entries.values() if not e.sha256]
    assert not unpinned, f"unpinned (unverifiable) entries: {unpinned}"


# --- device -> variant -------------------------------------------------

def test_device_variant_mapping():
    reg = _registry()
    assert reg.resolve("spruce", device="cpu").id == "spruce_int8"
    assert reg.resolve("spruce", device="gpu").id == "spruce_fp16"
    assert reg.resolve("spruce", device="coreml").id == "spruce_fp32"


def test_device_variant_falls_back_to_family_default():
    reg = _registry()
    # beech declares no int8/fp16 variant -> every device lands on fp32.
    assert reg.resolve("beech", device="cpu").id == "beech_fp32"
    assert reg.resolve("beech", device="gpu").id == "beech_fp32"
    assert reg.resolve("beech", device="coreml").id == "beech_fp32"


def test_explicit_model_id_never_rewritten_by_device():
    reg = _registry()
    for device in ("cpu", "gpu", "coreml"):
        assert reg.resolve("spruce_fp32", device=device).id == "spruce_fp32"
    # case-insensitive entry lookup is still an explicit id, not a family.
    assert reg.resolve("SPRUCE_FP32", device="cpu").id == "spruce_fp32"


def test_default_entry_uses_gui_default_and_device():
    reg = _registry()
    assert reg.default_entry(device="cpu").id == "spruce_int8"
    assert reg.default_entry(device="gpu").id == "spruce_fp16"
    assert reg.default_entry(device="coreml").id == "spruce_fp32"


def test_unknown_name_raises_key_error():
    reg = _registry()
    with pytest.raises(KeyError):
        reg.resolve("no-such-model")


# --- the variant kwarg ---------------------------------------------------

def test_variant_explicit_precision_ignores_device():
    reg = _registry()
    for device in ("cpu", "gpu", "coreml"):
        assert reg.resolve("spruce", device=device,
                           variant="int8").id == "spruce_int8"
        assert reg.resolve("spruce", device=device,
                           variant="fp16").id == "spruce_fp16"
        assert reg.resolve("spruce", device=device,
                           variant="fp32").id == "spruce_fp32"


def test_variant_explicit_precision_missing_raises_clear_key_error():
    reg = _registry()
    # beech ships only the fp32 default.
    for missing in ("int8", "fp16"):
        with pytest.raises(KeyError, match="beech.*has no"):
            reg.resolve("beech", device="cpu", variant=missing)
    assert reg.resolve("beech", device="cpu", variant="fp32").id \
        == "beech_fp32"


def test_variant_default_and_none_keep_device_rule():
    reg = _registry()
    for variant in (None, "default"):
        assert reg.resolve("spruce", device="cpu",
                           variant=variant).id == "spruce_int8"
        assert reg.resolve("spruce", device="gpu",
                           variant=variant).id == "spruce_fp16"
        # missing device precision falls back to the family default.
        assert reg.resolve("beech", device="cpu",
                           variant=variant).id == "beech_fp32"


def test_variant_auto_is_lossless_only():
    reg = _registry()
    # cpu wants int8, which is NOT lossless -> stay on the fp32 default.
    assert reg.resolve("spruce", device="cpu",
                       variant="auto").id == "spruce_fp32"
    # gpu wants fp16, which IS lossless -> substituted.
    assert reg.resolve("spruce", device="gpu",
                       variant="auto").id == "spruce_fp16"
    assert reg.resolve("spruce", device="coreml",
                       variant="auto").id == "spruce_fp32"
    # no variants at all -> the default, on every device.
    assert reg.resolve("beech", device="cpu",
                       variant="auto").id == "beech_fp32"


def test_variant_never_rewrites_explicit_entry_id():
    reg = _registry()
    assert reg.resolve("spruce_int8", device="gpu",
                       variant="fp32").id == "spruce_int8"


def test_variant_unknown_value_raises_value_error():
    reg = _registry()
    with pytest.raises(ValueError, match="unknown variant"):
        reg.resolve("spruce", device="cpu", variant="int4")


def test_lossless_defaults_true_and_hidden_defaults_false():
    reg = _registry()
    # fixture fp32 entries carry no flags at all.
    assert reg.entries["spruce_fp32"].lossless is True
    assert reg.entries["spruce_fp32"].hidden is False
    assert reg.entries["spruce_int8"].lossless is False


def test_hidden_entries_dropped_from_visible():
    raw = json.loads(json.dumps(FIXTURE_V2))   # deep copy
    raw["models"]["spruce_fp16"]["hidden"] = True
    reg = mr._parse_v2(raw, "<fixture>")
    ids = [e.id for e in reg.visible()]
    assert "spruce_fp16" not in ids
    assert "spruce_fp32" in ids
    # hidden filters choosers, never resolution.
    assert reg.resolve("spruce", device="gpu").id == "spruce_fp16"


# --- fixture-registry pinning property ----------------------------------

def test_fixture_registry_every_entry_has_sha256():
    assert_all_pinned(_registry())


def test_assert_all_pinned_catches_unpinned_entry():
    raw = json.loads(json.dumps(FIXTURE_V2))   # deep copy
    raw["models"]["spruce_fp32"]["sha256"] = None
    reg = mr._parse_v2(raw, "<fixture>")
    with pytest.raises(AssertionError):
        assert_all_pinned(reg)


# --- download atomicity -------------------------------------------------

_PAYLOAD = b"totally-a-model-payload"
_DIGEST = hashlib.sha256(_PAYLOAD).hexdigest()


def _entry(sha256=_DIGEST, file="m.onnx"):
    return mr.ModelEntry(id="m", label="M", family="", precision="fp32",
                         url="https://example.com/m.onnx", file=file,
                         sha256=sha256)


def test_download_success_writes_final_file_and_no_part(tmp_path):
    def fetcher(url, tmp_path_, progress, timeout):
        with open(tmp_path_, "wb") as f:
            f.write(_PAYLOAD)
        return None   # exercise the re-hash fallback path too

    dest = mr.download_model(_entry(), str(tmp_path), fetcher=fetcher)
    assert dest == str(tmp_path / "m.onnx")
    assert (tmp_path / "m.onnx").read_bytes() == _PAYLOAD
    assert not (tmp_path / "m.onnx.part").exists()


def test_download_checksum_mismatch_leaves_no_files(tmp_path):
    def fetcher(url, tmp_path_, progress, timeout):
        with open(tmp_path_, "wb") as f:
            f.write(b"wrong bytes entirely")

    with pytest.raises(mr.ModelDownloadError):
        mr.download_model(_entry(), str(tmp_path), fetcher=fetcher)
    assert not (tmp_path / "m.onnx").exists()
    assert not (tmp_path / "m.onnx.part").exists()


def test_download_fetcher_failure_leaves_no_files(tmp_path):
    def fetcher(url, tmp_path_, progress, timeout):
        with open(tmp_path_, "wb") as f:
            f.write(b"partial")
        raise OSError("connection reset")

    with pytest.raises(mr.ModelDownloadError):
        mr.download_model(_entry(), str(tmp_path), fetcher=fetcher)
    assert not (tmp_path / "m.onnx").exists()
    assert not (tmp_path / "m.onnx.part").exists()


# --- ensure_model ---------------------------------------------------------

def test_ensure_model_short_circuits_on_verified_existing(tmp_path):
    (tmp_path / "m.onnx").write_bytes(_PAYLOAD)
    calls = []

    def fetcher(url, tmp_path_, progress, timeout):
        calls.append(url)

    path = mr.ensure_model(_entry(), str(tmp_path), fetcher=fetcher)
    assert path == str(tmp_path / "m.onnx")
    assert calls == []   # fetcher must not run


def test_ensure_model_refetches_stale_file(tmp_path):
    (tmp_path / "m.onnx").write_bytes(b"stale garbage")

    def fetcher(url, tmp_path_, progress, timeout):
        with open(tmp_path_, "wb") as f:
            f.write(_PAYLOAD)

    path = mr.ensure_model(_entry(), str(tmp_path), fetcher=fetcher)
    assert (tmp_path / "m.onnx").read_bytes() == _PAYLOAD
    assert path == str(tmp_path / "m.onnx")


def test_ensure_model_no_download_refuses_with_clear_error(tmp_path):
    calls = []

    def fetcher(url, tmp_path_, progress, timeout):
        calls.append(url)

    with pytest.raises(mr.ModelDownloadError, match="downloads disabled"):
        mr.ensure_model(_entry(), str(tmp_path), fetcher=fetcher,
                        no_download=True)
    assert calls == []


# --- v1-flat fallback -----------------------------------------------------

def test_v1_flat_fallback_maps_names_to_entries(tmp_path):
    raw = {
        "Spruce": "https://example.com/spruce.onnx",
        "Beech": "https://example.com/beech.onnx?token=abc",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))

    reg = mr.load_registry(str(config_path))
    assert reg.schema == 1
    spruce = reg.get("Spruce")
    assert spruce.url == "https://example.com/spruce.onnx"
    assert spruce.file == "Spruce.onnx"
    # case-insensitive resolve, and no crash on a querystring URL.
    assert reg.resolve("spruce").id == "Spruce"
    assert reg.resolve("beech").file == "Beech.onnx"


def test_load_registry_v2_from_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(FIXTURE_V2))
    reg = mr.load_registry(str(config_path))
    assert reg.schema == 2
    assert reg.default_entry(device="cpu").id == "spruce_int8"


def test_load_registry_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mr.load_registry(str(tmp_path / "nope.json"))


# --- real, shipped config.json (schema-2, pinned to the models-v1 release) -

CLASSIC_FAMILIES = ("Spruce", "Beech", "Spruce_Deadwood", "General")


def _shipped_registry():
    return mr.load_registry(SHIPPED_CONFIG)


def test_shipped_config_parses_as_schema_2():
    reg = _shipped_registry()
    assert reg.schema == 2
    assert reg.entries


def test_shipped_config_every_entry_has_sha256():
    assert_all_pinned(_shipped_registry())


def test_shipped_config_every_entry_has_a_downloadable_url():
    reg = _shipped_registry()
    bad = [e.id for e in reg.entries.values()
           if not e.url or not e.url.lower().startswith("http")]
    assert not bad, f"entries without an http(s) url: {bad}"


def test_shipped_config_gui_default_resolves():
    reg = _shipped_registry()
    assert reg.gui_default is not None
    assert reg.gui_default in reg.entries
    assert reg.get(reg.gui_default).sha256


def test_shipped_config_recommended_all_resolve():
    reg = _shipped_registry()
    assert reg.recommended, "expected a non-empty recommended list"
    for mid in reg.recommended:
        assert mid in reg.entries
    # design decision: recommended[0] is the gui_default.
    assert reg.recommended[0] == reg.gui_default


def test_shipped_config_classic_family_names_present():
    reg = _shipped_registry()
    assert set(CLASSIC_FAMILIES) <= set(reg.families)


@pytest.mark.parametrize("family", CLASSIC_FAMILIES)
@pytest.mark.parametrize("device", ["cpu", "gpu", "coreml"])
def test_shipped_config_classic_families_resolve_on_every_device(
        family, device):
    reg = _shipped_registry()
    entry = reg.resolve(family, device=device)
    assert entry.sha256
    assert entry.url.lower().startswith("http")


def test_shipped_config_default_entry_resolves_on_cpu():
    reg = _shipped_registry()
    entry = reg.default_entry(device="cpu")
    assert entry.id == reg.gui_default
    assert entry.sha256


def test_shipped_config_lossless_flags_match_release_metadata():
    """fp32 references and the fp16 builds are certified lossless; the
    Keras-family int8 builds are not; the PyTorch w05 int8 build is the
    one certified int8 (mirrors the models-v1 release metadata)."""
    reg = _shipped_registry()
    for entry in reg.entries.values():
        if entry.precision in ("fp32", "fp16"):
            assert entry.lossless, entry.id
        elif entry.id == "beech_pytorch_int8":
            assert entry.lossless, entry.id
        else:
            assert not entry.lossless, entry.id


@pytest.mark.parametrize("family", CLASSIC_FAMILIES)
def test_shipped_config_auto_variant_refuses_uncertified_int8(family):
    reg = _shipped_registry()
    # cpu device rule wants int8, which is not certified -> fp32.
    assert reg.resolve(family, device="cpu",
                       variant="auto").precision == "fp32"
    # gpu wants fp16, which is certified -> substituted.
    assert reg.resolve(family, device="gpu",
                       variant="auto").precision == "fp16"


def test_shipped_config_has_no_hidden_entries():
    reg = _shipped_registry()
    assert [e.id for e in reg.visible()] == list(reg.entries)
