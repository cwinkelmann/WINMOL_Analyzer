"""plugin_utils/model_status.scan() — the model tree's data source.

scan() is called on the GUI thread every time the Setup tab is shown, so
it must be stat()-only: no hashing, no network, no subprocess beyond the
device probe the caller supplies. It is also where the device-aware
default becomes visible, which is the one-click first-run action.
"""

import hashlib
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plugin_utils import model_registry as mr     # noqa: E402
from plugin_utils import model_status             # noqa: E402

SHIPPED_CONFIG = os.path.join(REPO, "config.json")

FIXTURE = {
    "schema": 2,
    "tile_px": 512,
    "gui_default": "Alpha_int8",
    "recommended": ["Alpha_int8"],
    "preload": [],
    "families": {
        "alpha": {"label": "Alpha", "default": "Alpha",
                  "cpu": "Alpha_int8", "gpu": "Alpha_fp16"},
        "beta": {"label": "Beta", "default": "Beta"},
    },
    "models": {
        "Alpha": {"label": "Alpha fp32", "family": "alpha",
                  "precision": "fp32", "size_mb": 92.0,
                  "url": "https://example.invalid/a.onnx", "file": "a.onnx",
                  "sha256": hashlib.sha256(b"AAAA").hexdigest()},
        "Alpha_int8": {"label": "Alpha int8", "family": "alpha",
                       "precision": "int8", "backend": "cpu",
                       "url": "https://example.invalid/ai.onnx",
                       "file": "ai.onnx",
                       "sha256": hashlib.sha256(b"BBBB").hexdigest()},
        "Alpha_fp16": {"label": "Alpha fp16", "family": "alpha",
                       "precision": "fp16", "backend": "gpu",
                       "url": "https://example.invalid/af.onnx",
                       "file": "af.onnx",
                       "sha256": hashlib.sha256(b"CCCC").hexdigest()},
        "Beta": {"label": "Beta fp32", "family": "beta",
                 "url": "https://example.invalid/b.onnx", "file": "b.onnx"},
        "Beta_hdf5": {"label": "Beta original", "family": "beta",
                      "format": "hdf5", "hidden": True,
                      "url": "https://example.invalid/b.hdf5",
                      "file": "b.hdf5", "md5": "0" * 32},
    },
}


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(FIXTURE))
    return mr.load_registry(str(path))


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return str(d)


def _rows_by_id(rows):
    return {row.entry_id: row for row in rows}


def test_scan_covers_every_entry_and_excludes_reserved(registry, models_dir):
    rows = model_status.scan(registry, models_dir, device="cpu")
    assert set(_rows_by_id(rows)) == set(FIXTURE["models"])
    assert "Custom" not in _rows_by_id(rows)
    assert mr.RESERVED_IDS == ("Custom",)


def test_hidden_entries_are_flagged_not_dropped(registry, models_dir):
    rows = _rows_by_id(model_status.scan(registry, models_dir, device="cpu"))
    assert rows["Beta_hdf5"].hidden is True
    assert all(not rows[k].hidden for k in
               ("Alpha", "Alpha_int8", "Alpha_fp16", "Beta"))
    grouped = model_status.group_by_family(
        model_status.scan(registry, models_dir, device="cpu"))
    shown = {r.entry_id for _f, _l, members in grouped for r in members}
    assert "Beta_hdf5" not in shown


def test_pinned_is_false_exactly_where_no_digest_is_published(registry,
                                                              models_dir):
    rows = _rows_by_id(model_status.scan(registry, models_dir, device="cpu"))
    unpinned = {k for k, row in rows.items() if not row.pinned}
    assert unpinned == {"Beta"}


def test_shipped_registry_pins_every_downloadable_model():
    """Guards the property the Setup tab's 'unpinned' state exists for:
    it must stay reachable in code (legacy v1 configs) but must not be
    reachable from what we ship."""
    reg = mr.load_registry(SHIPPED_CONFIG)
    assert reg.unpinned() == []


@pytest.mark.parametrize("device,expected", [("cpu", "Alpha_int8"),
                                             ("gpu", "Alpha_fp16")])
def test_recommended_follows_the_device(registry, models_dir, device,
                                        expected):
    rows = model_status.scan(registry, models_dir, device=device,
                             family_id="alpha")
    flagged = [row.entry_id for row in rows if row.recommended]
    assert flagged == [expected]


def test_recommended_follows_the_selected_family(registry, models_dir):
    rows = model_status.scan(registry, models_dir, device="cpu",
                             family_id="beta")
    assert [r.entry_id for r in rows if r.recommended] == ["Beta"]


def test_recommended_defaults_to_the_registry_gui_default(registry,
                                                          models_dir):
    rows = model_status.scan(registry, models_dir, device="cpu")
    assert [r.entry_id for r in rows if r.recommended] == ["Alpha_int8"]


def test_an_unknown_family_flags_nothing_rather_than_raising(registry,
                                                             models_dir):
    rows = model_status.scan(registry, models_dir, device="cpu",
                             family_id="gamma")
    assert [r.entry_id for r in rows if r.recommended] == []


def test_states_and_sizes_come_from_disk(registry, models_dir):
    with open(os.path.join(models_dir, "ai.onnx"), "wb") as f:
        f.write(b"BBBB")
    with open(os.path.join(models_dir, "a.onnx"), "wb") as f:
        f.write(b"WRONG")
    with open(os.path.join(models_dir, "b.onnx"), "wb") as f:
        f.write(b"anything")
    rows = _rows_by_id(model_status.scan(registry, models_dir, device="cpu"))

    assert rows["Alpha_fp16"].state == "missing"
    assert rows["Alpha_fp16"].present is False
    assert rows["Alpha_fp16"].bytes_on_disk == 0
    # present but not yet hashed — never reported as verified
    assert rows["Alpha_int8"].state == "present"
    assert rows["Alpha_int8"].present is True
    assert rows["Alpha_int8"].bytes_on_disk == 4
    assert rows["Alpha_int8"].verified is False
    # no digest published: its own state, distinct from verified
    assert rows["Beta"].state == "unpinned"
    assert rows["Beta"].pinned is False


def test_verified_state_after_a_verify_pass(registry, models_dir):
    entry = registry.entries["Alpha_int8"]
    with open(os.path.join(models_dir, "ai.onnx"), "wb") as f:
        f.write(b"BBBB")
    assert mr.verify_entry(entry, models_dir) is True
    rows = _rows_by_id(model_status.scan(registry, models_dir, device="cpu"))
    assert rows["Alpha_int8"].state == "verified"
    assert rows["Alpha_int8"].verified is True


def test_corrupt_state_after_a_failed_verify(registry, models_dir):
    entry = registry.entries["Alpha"]
    with open(os.path.join(models_dir, "a.onnx"), "wb") as f:
        f.write(b"NOT-THE-MODEL")
    assert mr.verify_entry(entry, models_dir) is False
    rows = _rows_by_id(model_status.scan(registry, models_dir, device="cpu"))
    assert rows["Alpha"].state == "corrupt"
    assert rows["Alpha"].verified is False


def test_scan_never_hashes(registry, models_dir, monkeypatch):
    """A repaint must stay cheap: scan() runs on the GUI thread."""
    with open(os.path.join(models_dir, "a.onnx"), "wb") as f:
        f.write(b"AAAA")

    def _boom(*_a, **_k):
        raise AssertionError("scan() must not hash a model file")
    monkeypatch.setattr(mr, "_hash_file", _boom)
    monkeypatch.setattr(mr, "detect_device", _boom)
    rows = model_status.scan(registry, models_dir, device="cpu")
    assert _rows_by_id(rows)["Alpha"].state == "present"


def test_grouping_is_by_family_in_registry_order(registry, models_dir):
    rows = model_status.scan(registry, models_dir, device="cpu")
    grouped = model_status.group_by_family(rows, show_all_variants=True)
    assert [fid for fid, _label, _members in grouped] == ["alpha", "beta"]
    assert grouped[0][1] == "Alpha"
    assert len(grouped[0][2]) == 3


def test_default_grouping_hides_unused_variants(registry, models_dir):
    rows = model_status.scan(registry, models_dir, device="cpu",
                             family_id="alpha")
    grouped = dict((fid, members)
                   for fid, _l, members in model_status.group_by_family(rows))
    shown = {row.entry_id for row in grouped["alpha"]}
    # the fp32 reference and the device-recommended int8; not the fp16
    assert shown == {"Alpha", "Alpha_int8"}


def test_family_summary_text(registry, models_dir):
    with open(os.path.join(models_dir, "ai.onnx"), "wb") as f:
        f.write(b"x" * (96 * 1024 * 1024))
    rows = [r for r in model_status.scan(registry, models_dir, device="cpu")
            if r.family_id == "alpha"]
    assert model_status.family_summary(rows) == "1 of 3 on disk, 96 MB"


def test_scan_against_the_shipped_registry():
    """The real config.json must produce a sane tree with no models on
    disk (a fresh install) — 8 families, the four Zenodo originals
    hidden, nothing pretending to be verified."""
    reg = mr.load_registry(SHIPPED_CONFIG)
    rows = model_status.scan(reg, os.path.join(REPO, "no-such-models"),
                             device="cpu")
    assert len(rows) == len(reg.entries)
    assert all(row.state == "missing" for row in rows)
    hidden = sorted(row.entry_id for row in rows if row.hidden)
    assert hidden and all(h.endswith("_hdf5") for h in hidden)
    grouped = model_status.group_by_family(rows)
    assert len(grouped) == len(reg.families)
    # exactly what a run would load, resolver and all — including
    # resolve()'s lossless gate, which keeps 'auto' on the fp32
    # reference for a family whose int8 variant is not certified
    family = reg.families[reg.entries[reg.gui_default].family]
    expected = reg.resolve(family.id, device="cpu", variant="auto").id
    assert [r.entry_id for r in rows if r.recommended] == [expected]


def test_the_declared_default_is_flagged_when_its_variant_is_selected():
    """The dialog preselects the precision the registry calls the
    effective default, so the highlighted row must follow the variant
    selector rather than second-guess it."""
    reg = mr.load_registry(SHIPPED_CONFIG)
    family = reg.families[reg.entries[reg.gui_default].family]
    precision = reg.entries[reg.gui_default].precision
    rows = model_status.scan(reg, os.path.join(REPO, "no-such-models"),
                             device="cpu", family_id=family.id,
                             variant=precision)
    assert [r.entry_id for r in rows if r.recommended] == [reg.gui_default]
