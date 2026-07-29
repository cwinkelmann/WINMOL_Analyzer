"""Setup-tab backend: env removal safety, state texts, model rows.

The safety property under test: ``installer.remove_environment`` must
never delete anything outside the managed tree, and ``deletion_plan``
must classify a bring-your-own interpreter as untouchable.
"""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from plugin_utils import installer, model_status, setup_state  # noqa: E402


def _make_venv(plugin_dir, payload=b"x" * 1024):
    """A fake managed venv (off QGIS, managed_root == plugin_dir)."""
    venv = Path(installer.venv_location(str(plugin_dir)))
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_bytes(payload)
    (venv / "lib.bin").write_bytes(payload)
    return venv


# --- human_bytes ------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (0, "0 bytes"),
    (512, "512 bytes"),
    (2048, "2 KB"),
    (63 * 1024 ** 2, "63 MB"),
    (2 * 1024 ** 3, "2.0 GB"),
    (-5, "0 bytes"),
    (None, "0 bytes"),
])
def test_human_bytes(n, expected):
    assert setup_state.human_bytes(n) == expected


# --- deletion_plan ----------------------------------------------------------

def test_deletion_plan_byo_refuses_outside_interpreter(tmp_path):
    outside = tmp_path / "conda" / "bin" / "python"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"#!")
    plan = setup_state.deletion_plan(str(tmp_path / "plugin"),
                                     configured_exe=str(outside))
    assert plan == {"kind": "byo", "paths": [], "clears_setting": False}


def test_deletion_plan_empty_when_nothing_managed(tmp_path):
    plan = setup_state.deletion_plan(str(tmp_path))
    assert plan == {"kind": "none", "paths": [], "clears_setting": False}


def test_deletion_plan_managed_venv(tmp_path):
    venv = _make_venv(tmp_path)
    exe = str(venv / "bin" / "python")
    plan = setup_state.deletion_plan(str(tmp_path), configured_exe=exe)
    assert plan["kind"] == "managed"
    assert str(venv) in plan["paths"]
    assert plan["clears_setting"] is True


# --- remove_environment -----------------------------------------------------

def test_remove_dry_run_prices_without_deleting(tmp_path):
    venv = _make_venv(tmp_path)
    result = installer.remove_environment(str(tmp_path), dry_run=True)
    assert venv.exists()
    assert result["removed"] == []
    assert result["freed_bytes"] == 2048
    assert [p for p, _s in result["planned"]] == [str(venv)]


def test_remove_deletes_only_inside_managed_tree(tmp_path):
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    venv = _make_venv(plugin_dir)
    bystander = tmp_path / "keep.bin"
    bystander.write_bytes(b"y" * 64)
    result = installer.remove_environment(str(plugin_dir))
    assert not venv.exists()
    assert bystander.exists()
    assert result["removed"] == [str(venv)]
    assert result["freed_bytes"] == 2048
    assert result["failed"] == []


def test_remove_refuses_path_outside_managed_tree(tmp_path, monkeypatch):
    """THE safety test: a venv location outside the managed tree is
    refused, and nothing under it is deleted."""
    victim = tmp_path / "elsewhere" / "victim_venv"
    victim.mkdir(parents=True)
    (victim / "precious.txt").write_bytes(b"do not delete")
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    monkeypatch.setattr(installer, "venv_location",
                        lambda _pd: str(victim))
    result = installer.remove_environment(str(plugin_dir))
    assert (victim / "precious.txt").exists()
    assert result["removed"] == []
    assert result["failed"] == [
        (str(victim), "refused: outside the managed tree")]


def test_remove_clear_setting_only_for_managed_exe(tmp_path):
    venv = _make_venv(tmp_path)
    managed = str(venv / "bin" / "python")
    byo = str(tmp_path / ".." / "other" / "python")
    inside = installer.remove_environment(
        str(tmp_path), configured_exe=managed, dry_run=True)
    outside = installer.remove_environment(
        str(tmp_path), configured_exe=byo, dry_run=True)
    assert inside["clear_setting"] is True
    assert outside["clear_setting"] is False


def _evil_registry(plugin_dir, file_value):
    """A schema-2 registry with one entry whose ``file`` is hostile."""
    config = {"schema": 2, "models": {"evil": {
        "label": "evil", "url": "https://example.org/x.onnx",
        "file": file_value}}}
    (plugin_dir / "config.json").write_text(json.dumps(config))


def test_remove_models_refuses_traversal_entry(tmp_path):
    """A registry ``file`` of "../../x" must be refused per victim —
    the outside file survives and shows up under failed, not removed."""
    victim = tmp_path / "escape.txt"
    victim.write_bytes(b"precious")
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    models_dir = Path(installer.models_location(str(plugin_dir)))
    models_dir.mkdir(parents=True)
    _evil_registry(plugin_dir, os.path.relpath(victim, models_dir))
    result = installer.remove_environment(
        str(plugin_dir), remove_venv=False, remove_models=True)
    assert victim.exists()
    assert result["removed"] == []
    assert result["freed_bytes"] == 0
    assert result["failed"]
    assert all(msg == "refused: outside the models directory"
               for _p, msg in result["failed"])


def test_remove_models_refuses_absolute_entry(tmp_path):
    """An absolute registry ``file`` escapes os.path.join entirely;
    it must be refused, never deleted."""
    victim = tmp_path / "abs_victim.onnx"
    victim.write_bytes(b"precious")
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    Path(installer.models_location(str(plugin_dir))).mkdir(parents=True)
    _evil_registry(plugin_dir, str(victim))
    result = installer.remove_environment(
        str(plugin_dir), remove_venv=False, remove_models=True)
    assert victim.exists()
    assert result["removed"] == []
    assert (str(victim), "refused: outside the models directory") \
        in result["failed"]


def test_remove_refuses_symlinked_venv_root(tmp_path):
    """When the venv path IS a symlink, removal is refused upfront:
    the link survives, the target (contents AND permissions) is
    untouched, and nothing is falsely reported as removed."""
    target = tmp_path / "target"
    target.mkdir()
    keep = target / "important.txt"
    keep.write_bytes(b"precious")
    mode_before = stat.S_IMODE(os.stat(target).st_mode)
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    venv = installer.venv_location(str(plugin_dir))
    os.symlink(str(target), venv)
    result = installer.remove_environment(str(plugin_dir))
    assert os.path.islink(venv)                     # link survives
    assert keep.exists()                            # target intact
    assert stat.S_IMODE(os.stat(target).st_mode) == mode_before
    assert result["removed"] == []
    assert result["freed_bytes"] == 0
    assert result["failed"] == [
        (venv, "refused: the path is a symlink")]


def test_remove_environment_none_refuses_without_raising():
    """The never-raises contract holds for None/'' plugin_dir: a
    refused-empty result, no TypeError."""
    for bogus in (None, ""):
        result = installer.remove_environment(bogus)
        assert result["planned"] == []
        assert result["removed"] == []
        assert result["freed_bytes"] == 0
        assert result["clear_setting"] is False
        assert result["failed"] == [
            ("", "refused: no plugin directory")]


# --- invalidate_marker ------------------------------------------------------

def test_invalidate_marker_forces_not_ready(tmp_path, monkeypatch):
    venv = _make_venv(tmp_path)
    monkeypatch.setattr(installer, "_python_version",
                        lambda _exe: (3, 11))
    installer._write_marker(str(venv))
    assert installer.is_ready(str(venv)) is True
    assert installer.invalidate_marker(str(venv)) is True
    assert installer.is_ready(str(venv)) is False
    assert installer.invalidate_marker(str(venv)) is False


# --- model_status.scan ------------------------------------------------------

def _write_registry(tmp_path, int8_payload=b"quantized-weights"):
    digest = hashlib.sha256(int8_payload).hexdigest()
    config = {
        "schema": 2,
        "gui_default": "spruce_fp32",
        "recommended": ["spruce_fp32"],
        "families": {"spruce": {"label": "Spruce",
                                "default": "spruce_fp32"}},
        "models": {
            "spruce_fp32": {"label": "Spruce fp32", "family": "spruce",
                            "precision": "fp32", "size_mb": 124.0,
                            "url": "https://example.org/s32.onnx",
                            "file": "spruce_fp32.onnx"},
            "spruce_int8": {"label": "Spruce int8", "family": "spruce",
                            "precision": "int8", "size_mb": 32.0,
                            "sha256": digest,
                            "url": "https://example.org/s8.onnx",
                            "file": "spruce_int8.onnx"},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "spruce_int8.onnx").write_bytes(int8_payload)
    return str(config_path), str(models_dir)


def test_scan_flags_device_default_and_installed(tmp_path):
    config_path, models_dir = _write_registry(tmp_path)
    rows = model_status.scan(config_path, models_dir, device="cpu")
    by_id = {row.entry_id: row for row in rows}
    assert set(by_id) == {"spruce_fp32", "spruce_int8"}
    assert by_id["spruce_int8"].is_default is True      # cpu -> int8
    assert by_id["spruce_fp32"].is_default is False
    assert by_id["spruce_int8"].installed is True
    assert by_id["spruce_int8"].bytes_on_disk == len(b"quantized-weights")
    assert by_id["spruce_fp32"].installed is False


# --- env_info + texts -------------------------------------------------------

def test_env_info_managed_ready(tmp_path):
    venv = _make_venv(tmp_path)
    installer._write_marker(str(venv))
    exe = str(venv / "bin" / "python")

    def resolve_fn(_plugin_dir, build=False):
        return {"status": "ready", "python": exe,
                "venv_path": str(venv), "message": "ok"}

    info = setup_state.env_info(str(tmp_path), resolve_fn=resolve_fn)
    assert info.managed is True
    assert info.variant == "cpu"
    assert info.venv_bytes >= 2048     # payload + the ready marker
    assert setup_state.env_ready(info) is True
    assert setup_state.blocking_reason(info) is None
    assert setup_state.env_state_text(info) == \
        setup_state.TXT_ENV_READY_MANAGED.format(variant="CPU")


def test_env_info_needs_setup_blocks_run(tmp_path):
    def resolve_fn(_plugin_dir, build=False):
        return {"status": "needs_setup", "python": None,
                "venv_path": str(tmp_path / "winmol_venv"),
                "message": "not set up"}

    info = setup_state.env_info(str(tmp_path), resolve_fn=resolve_fn)
    assert setup_state.env_ready(info) is False
    assert setup_state.env_state_text(info) == setup_state.TXT_ENV_NONE
    assert (setup_state.blocking_reason(info)
            == setup_state.TXT_BLOCK_NO_ENV)
    assert (setup_state.blocking_reason(info, busy=True)
            == setup_state.TXT_BLOCK_BUSY)


def test_button_states_interlock(tmp_path):
    config_path, models_dir = _write_registry(tmp_path)
    rows = model_status.scan(config_path, models_dir, device="cpu")

    def resolve_fn(_plugin_dir, build=False):
        return {"status": "needs_setup", "python": None,
                "venv_path": str(tmp_path / "winmol_venv"), "message": ""}

    info = setup_state.env_info(str(tmp_path), resolve_fn=resolve_fn)
    states = setup_state.button_states(info, rows,
                                       selected_entry_id="spruce_fp32")
    assert states["env_create_button"] is True
    assert states["run_button"] is False
    assert states["models_download_button"] is True     # fp32 missing
    assert states["models_delete_button"] is False
    busy = setup_state.button_states(info, rows, busy=True)
    assert all(v is False for v in busy.values())


def test_models_summary_text(tmp_path):
    config_path, models_dir = _write_registry(tmp_path)
    rows = model_status.scan(config_path, models_dir, device="cpu")
    text = setup_state.models_summary_text(rows)
    assert text == setup_state.TXT_MODELS_SUMMARY.format(
        have=1, total=2, size=setup_state.human_bytes(
            len(b"quantized-weights")))
