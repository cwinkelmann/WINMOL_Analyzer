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
    py = Path(installer.get_venv_python_path(str(venv)))
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_bytes(payload)
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


def test_scan_variant_follows_selection(tmp_path):
    """The is_default flag follows the SELECTED family + variant, not
    just the device rule: fp32 chosen on a CPU box highlights fp32."""
    config_path, models_dir = _write_registry(tmp_path)
    rows = model_status.scan(config_path, models_dir, device="cpu",
                             family_id="spruce", variant="fp32")
    by_id = {row.entry_id: row for row in rows}
    assert by_id["spruce_fp32"].is_default is True
    assert by_id["spruce_int8"].is_default is False
    # An unresolvable variant flags nothing rather than guessing.
    none_flagged = model_status.scan(config_path, models_dir,
                                     device="cpu", family_id="spruce",
                                     variant="fp16")
    assert not any(row.is_default for row in none_flagged)


def test_scan_read_fallback_detects_legacy_install(tmp_path):
    """A file only in the legacy dir (rr6-era install next to the
    plugin) shows installed with its REAL path; a missing file keeps
    the managed dir as the write target."""
    config_path, models_dir = _write_registry(tmp_path)
    legacy = tmp_path / "legacy_models"
    legacy.mkdir()
    (legacy / "spruce_fp32.onnx").write_bytes(b"fp32-weights")
    rows = model_status.scan(config_path, models_dir, device="cpu",
                             fallback_dirs=(str(legacy),))
    by_id = {row.entry_id: row for row in rows}
    assert by_id["spruce_fp32"].installed is True
    assert by_id["spruce_fp32"].path == str(legacy / "spruce_fp32.onnx")
    assert by_id["spruce_fp32"].bytes_on_disk == len(b"fp32-weights")
    # int8 lives in the managed dir: the fallback never wins over it.
    assert by_id["spruce_int8"].path == os.path.join(
        models_dir, "spruce_int8.onnx")


def test_group_by_family_and_summary(tmp_path):
    config_path, models_dir = _write_registry(tmp_path)
    rows = model_status.scan(config_path, models_dir, device="cpu")
    grouped = model_status.group_by_family(rows)
    assert [(fam_id, label) for fam_id, label, _r in grouped] == \
        [("spruce", "Spruce")]
    fam_rows = grouped[0][2]
    assert {row.entry_id for row in fam_rows} == \
        {"spruce_fp32", "spruce_int8"}
    assert model_status.family_summary(fam_rows) == \
        "1 of 2 on disk, " + setup_state.human_bytes(
            len(b"quantized-weights"))


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


def test_button_states_delete_reachable_for_byo(tmp_path):
    """A bring-your-own interpreter with no managed venv on disk must
    keep env_delete_button live — it is the only path to the 'Forget
    this interpreter' offer."""
    info = setup_state.EnvInfo(
        status="byo", python=str(tmp_path / "conda" / "bin" / "python"),
        venv_path=str(tmp_path / "plugin" / "winmol_venv"),
        managed=False, variant=None, venv_bytes=0, message="")
    states = setup_state.button_states(info, [])
    assert states["env_delete_button"] is True


def test_models_summary_text(tmp_path):
    config_path, models_dir = _write_registry(tmp_path)
    rows = model_status.scan(config_path, models_dir, device="cpu")
    text = setup_state.models_summary_text(rows)
    assert text == setup_state.TXT_MODELS_SUMMARY.format(
        have=1, total=2, size=setup_state.human_bytes(
            len(b"quantized-weights")))


def test_busy_guarded_slots_take_no_required_signal_parameters():
    """_refuse_if_busy's wrapper drops Qt's POSITIONAL signal args and
    calls ``method(self, **kwargs)`` (PyQt's own truncation does this
    for undecorated bound methods; a naive forwarding wrapper once broke
    it: clicked(bool) -> TypeError). Pin the contract: a decorated slot
    must have no REQUIRED extra positional parameter and no ``*args``/
    ``**kwargs`` (Qt could not satisfy them). Optional params WITH
    defaults are allowed — Qt's positional args never reach them, and an
    internal caller may pass them by keyword (e.g. confirmed=True)."""
    import ast as _ast
    src = (REPO / "winmol_analyzer_dialog.py").read_text()
    tree = _ast.parse(src)
    offenders = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.FunctionDef):
            continue
        if not any(isinstance(d, _ast.Name) and d.id == "_refuse_if_busy"
                   for d in node.decorator_list):
            continue
        a = node.args
        # positional params after self, minus those with defaults
        positional = a.args[1:]
        n_required = len(positional) - len(a.defaults)
        required = [p.arg for p in positional[:max(0, n_required)]]
        kwonly_required = [p.arg for p, d in
                           zip(a.kwonlyargs, a.kw_defaults) if d is None]
        bad = required + kwonly_required
        if bad or a.vararg or a.kwarg:
            offenders.append("%s(%s)" % (node.name, ", ".join(
                ["self"] + bad
                + (["*" + a.vararg.arg] if a.vararg else [])
                + (["**" + a.kwarg.arg] if a.kwarg else []))))
    assert not offenders, (
        "busy-guarded slots must not require a positional signal "
        "parameter (the wrapper calls method(self, **kwargs)): "
        + ", ".join(offenders))
