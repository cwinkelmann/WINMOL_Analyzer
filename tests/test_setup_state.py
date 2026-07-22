"""The Setup tab's brain, tested without a single widget.

plugin_utils/setup_state.py exists precisely so this file can exist: the
repo has no live QGIS in CI, so any decision left inside a Qt slot is a
decision nobody tests. Everything asserted here — why Run is blocked,
which button is live, what a deletion would take — is what the dialog
displays verbatim.
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plugin_utils import installer as inst        # noqa: E402
from plugin_utils import setup_state as ss        # noqa: E402


def _info(**kw):
    kw.setdefault("exe", "/env/bin/python")
    kw.setdefault("exists", True)
    kw.setdefault("version", (3, 11))
    kw.setdefault("deps_ok", True)
    kw.setdefault("managed", True)
    kw.setdefault("venv_path", "/env")
    kw.setdefault("runtime_path", "/rt")
    kw.setdefault("marker_ok", True)
    return ss.EnvInfo(**kw)


def _row(**kw):
    kw.setdefault("entry_id", "M")
    kw.setdefault("family_id", "fam")
    kw.setdefault("family_label", "Fam")
    kw.setdefault("label", "Model")
    kw.setdefault("precision", "fp32")
    kw.setdefault("backend", "any")
    kw.setdefault("file", "m.onnx")
    kw.setdefault("path", "/models/m.onnx")
    kw.setdefault("size_expected_mb", 10.0)
    kw.setdefault("bytes_on_disk", 0)
    kw.setdefault("present", False)
    kw.setdefault("pinned", True)
    kw.setdefault("verified", False)
    kw.setdefault("hidden", False)
    kw.setdefault("recommended", False)
    kw.setdefault("state", "missing")
    return ss.ModelRow(**kw)


# --- blocking_reason --------------------------------------------------------

def test_blocking_reason_no_interpreter_configured():
    info = _info(exe=None, exists=False, version=None, deps_ok=None,
                 managed=False, marker_ok=False)
    assert ss.blocking_reason(info) == (
        "No Python environment yet — open the Setup tab and create one.")


def test_blocking_reason_interpreter_path_gone():
    """A remembered interpreter that no longer exists on disk reads the
    same as none at all — it is equally unusable."""
    info = _info(exists=False, version=None, deps_ok=None)
    assert ss.blocking_reason(info) == ss.TXT_BLOCK_NO_ENV


@pytest.mark.parametrize("version,shown", [((3, 10), "3.10"),
                                           ((3, 12), "3.12"),
                                           ((3, 9, 6), "3.9.6")])
def test_blocking_reason_unsupported_python(version, shown):
    info = _info(version=version)
    assert ss.blocking_reason(info) == (
        f"The selected interpreter is Python {shown}; WINMOL needs "
        "3.11. Choose another in Setup.")


def test_blocking_reason_missing_dependencies():
    assert ss.blocking_reason(_info(deps_ok=False)) == (
        "The environment is missing onnxruntime/rasterio/geopandas. "
        "Reinstall dependencies in Setup.")


def test_blocking_reason_incomplete_managed_venv():
    assert ss.blocking_reason(_info(marker_ok=False)) == (
        "The environment is incomplete — reinstall dependencies in Setup.")


def test_a_stale_marker_does_not_block_a_byo_interpreter():
    """The marker describes the MANAGED venv. A user's own conda env has
    none and must not be reported as incomplete because of it."""
    info = _info(managed=False, marker_ok=False)
    assert ss.blocking_reason(info) is None


@pytest.mark.parametrize("kind", ["env", "download"])
def test_blocking_reason_busy(kind):
    assert ss.blocking_reason(_info(), kind) == (
        "Setup is busy — wait for the current job to finish.")


def test_busy_wins_over_a_broken_environment():
    info = _info(exists=False, version=None, deps_ok=None)
    assert ss.blocking_reason(info, "env") == ss.TXT_BLOCK_BUSY


def test_a_run_in_progress_is_not_a_setup_blocking_reason():
    """Run is already disabled by _run_active; claiming setup is busy
    would be a second, wrong explanation."""
    assert ss.blocking_reason(_info(), "run") is None


def test_ready_environment_has_no_blocking_reason():
    assert ss.blocking_reason(_info(), None) is None


def test_a_missing_model_is_never_a_blocking_reason():
    """Deliberate: a missing model is recoverable at Run time by the
    download pre-flight. Blocking on it strands a user who never opened
    the Setup tab."""
    rows = [_row(state="missing", present=False, recommended=True)]
    assert ss.blocking_reason(_info(), None) is None
    assert ss.models_summary_text(rows).startswith("Models on disk: 0 of 1")


# --- button_states ----------------------------------------------------------

ALL_BUTTONS = ("env_create_button", "env_choose_button", "env_repair_button",
               "env_delete_button", "models_download_button",
               "models_download_default_button", "models_verify_button",
               "models_delete_button", "models_refresh_button")


@pytest.mark.parametrize("busy", [{"run_active": True}, {"env_busy": True},
                                  {"dl_busy": True}])
def test_everything_is_disabled_while_anything_runs(busy):
    rows = [_row(state="verified", present=True, bytes_on_disk=10)]
    states = ss.button_states(_info(), rows, "M", **busy)
    for name in ALL_BUTTONS + ("run_button",):
        assert states[name] is False, name


@pytest.mark.parametrize("state,expected", [("missing", True),
                                            ("corrupt", True),
                                            ("present", False),
                                            ("verified", False),
                                            ("unpinned", False)])
def test_download_is_offered_for_missing_and_corrupt_only(state, expected):
    rows = [_row(state=state, present=state != "missing")]
    assert ss.button_states(_info(), rows, "M")[
        "models_download_button"] is expected


def test_verify_is_disabled_for_an_unpinned_row():
    """verify_file() returns True unconditionally without a published
    digest; offering Verify there would fake a guarantee."""
    rows = [_row(state="unpinned", present=True, pinned=False)]
    assert ss.button_states(_info(), rows, "M")[
        "models_verify_button"] is False


def test_verify_is_offered_for_a_pinned_file_on_disk():
    for state in ("present", "verified", "corrupt"):
        rows = [_row(state=state, present=True, pinned=True)]
        assert ss.button_states(_info(), rows, "M")[
            "models_verify_button"] is True, state


def test_nothing_selected_disables_the_row_actions():
    rows = [_row(state="verified", present=True)]
    states = ss.button_states(_info(), rows, None)
    assert states["models_download_button"] is False
    assert states["models_verify_button"] is False
    assert states["models_delete_button"] is False
    # but the folder-wide actions stay live
    assert states["models_refresh_button"] is True


def test_delete_needs_a_file_on_disk():
    assert ss.button_states(_info(), [_row(state="missing")], "M")[
        "models_delete_button"] is False
    assert ss.button_states(
        _info(), [_row(state="verified", present=True)], "M")[
        "models_delete_button"] is True


def test_download_recommended_tracks_the_recommended_row():
    on_disk = [_row(state="verified", present=True, recommended=True)]
    assert ss.button_states(_info(), on_disk, None)[
        "models_download_default_button"] is False
    absent = [_row(state="missing", recommended=True)]
    assert ss.button_states(_info(), absent, None)[
        "models_download_default_button"] is True


def test_create_is_offered_only_while_the_environment_is_not_ready():
    assert ss.button_states(_info(), [], None)[
        "env_create_button"] is False
    broken = _info(exists=False, version=None, deps_ok=None)
    assert ss.button_states(broken, [], None)[
        "env_create_button"] is True
    # repair and choose stay available either way
    assert ss.button_states(_info(), [], None)["env_repair_button"] is True
    assert ss.button_states(_info(), [], None)["env_choose_button"] is True


def test_delete_environment_needs_something_to_delete():
    nothing = _info(exe=None, exists=False, version=None, deps_ok=None,
                    managed=False, marker_ok=False)
    assert ss.button_states(nothing, [], None)[
        "env_delete_button"] is False
    assert ss.button_states(_info(), [], None)[
        "env_delete_button"] is True


def test_run_button_follows_the_environment():
    assert ss.button_states(_info(), [], None)["run_button"] is True
    assert ss.button_states(_info(deps_ok=False), [], None)[
        "run_button"] is False


# --- env_info ---------------------------------------------------------------

def test_env_info_managed_venv(tmp_path, monkeypatch):
    plugin_dir = str(tmp_path)
    venv = inst.venv_location(plugin_dir)
    exe = inst.get_venv_python_path(venv)
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    with open(exe, "wb") as f:
        f.write(b"#!")
    monkeypatch.setattr(inst, "_python_version", lambda _e: (3, 11))
    monkeypatch.setattr(inst, "_has_compute_deps", lambda _e: True)
    monkeypatch.setattr(inst, "is_ready", lambda _p, version=None: True)

    info = ss.env_info(plugin_dir, None)
    assert info.exe == exe and info.exists is True
    assert info.managed is True and info.marker_ok is True
    assert info.version == (3, 11) and info.deps_ok is True
    assert info.venv_path == venv
    assert info.runtime_path.endswith("py311")
    assert ss.env_state_text(info) == "Ready — Python 3.11"
    assert ss.blocking_reason(info) is None


def test_env_info_bring_your_own_interpreter(tmp_path, monkeypatch):
    plugin_dir = str(tmp_path)
    byo = tmp_path / "conda" / "bin" / "python"
    byo.parent.mkdir(parents=True)
    byo.write_bytes(b"#!")
    monkeypatch.setattr(inst, "_python_version", lambda _e: (3, 11))
    monkeypatch.setattr(inst, "_has_compute_deps", lambda _e: True)
    monkeypatch.setattr(inst, "is_ready", lambda _p, version=None: False)

    info = ss.env_info(plugin_dir, str(byo))
    assert info.managed is False and info.exists is True
    assert ss.blocking_reason(info) is None
    assert ss.env_detail_text(info) == (
        "your own interpreter · dependencies present")


def test_env_info_nothing_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(inst, "is_ready", lambda _p, version=None: False)
    monkeypatch.setattr(inst, "_python_version", _never_probed)
    monkeypatch.setattr(inst, "_has_compute_deps", _never_probed)
    info = ss.env_info(str(tmp_path), None)
    assert info.exe is None and info.exists is False
    assert info.version is None and info.deps_ok is None
    assert ss.env_state_text(info) == "Not set up"


def _never_probed(_exe):
    raise AssertionError("must not probe an interpreter that does not exist")


def test_env_info_configured_path_that_vanished(tmp_path, monkeypatch):
    monkeypatch.setattr(inst, "is_ready", lambda _p, version=None: False)
    monkeypatch.setattr(inst, "_python_version", _never_probed)
    monkeypatch.setattr(inst, "_has_compute_deps", _never_probed)
    info = ss.env_info(str(tmp_path), str(tmp_path / "deleted" / "python"))
    assert info.exists is False
    assert ss.blocking_reason(info) == ss.TXT_BLOCK_NO_ENV


def test_env_info_stale_marker(tmp_path, monkeypatch):
    plugin_dir = str(tmp_path)
    exe = inst.get_venv_python_path(inst.venv_location(plugin_dir))
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    with open(exe, "wb") as f:
        f.write(b"#!")
    monkeypatch.setattr(inst, "_python_version", lambda _e: (3, 11))
    monkeypatch.setattr(inst, "_has_compute_deps", lambda _e: True)
    monkeypatch.setattr(inst, "is_ready", lambda _p, version=None: False)
    info = ss.env_info(plugin_dir, None)
    assert info.marker_ok is False
    assert ss.env_state_text(info) == "Incomplete — reinstall dependencies"
    assert ss.blocking_reason(info) == ss.TXT_BLOCK_INCOMPLETE


def test_env_info_hands_the_measured_version_to_is_ready(tmp_path,
                                                         monkeypatch):
    """is_ready() re-spawned the venv's interpreter to re-learn a version
    env_info had just measured — two subprocess launches for one number,
    on a path that used to run on the GUI thread."""
    plugin_dir = str(tmp_path)
    venv = inst.venv_location(plugin_dir)
    exe = inst.get_venv_python_path(venv)
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    with open(exe, "wb") as f:
        f.write(b"#!")
    calls = []

    def _version(exe_path):
        calls.append(exe_path)
        return (3, 11)

    monkeypatch.setattr(inst, "_python_version", _version)
    monkeypatch.setattr(inst, "_has_compute_deps", lambda _e: True)
    monkeypatch.setattr(inst, "marker_matches", lambda _p: True)

    info = ss.env_info(plugin_dir, None)
    assert info.marker_ok is True
    assert calls == [exe], (
        f"the interpreter was probed {len(calls)} times for one version")


# --- env_seed: the probe-free first paint -----------------------------------

def _no_subprocess(_exe):
    raise AssertionError("env_seed must not spawn an interpreter")


def test_env_seed_never_probes(tmp_path, monkeypatch):
    plugin_dir = str(tmp_path)
    exe = inst.get_venv_python_path(inst.venv_location(plugin_dir))
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    with open(exe, "wb") as f:
        f.write(b"#!")
    monkeypatch.setattr(inst, "_python_version", _no_subprocess)
    monkeypatch.setattr(inst, "_has_compute_deps", _no_subprocess)
    monkeypatch.setattr(
        inst, "is_ready",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("env_seed must not call is_ready")))

    info = ss.env_seed(plugin_dir, None)
    assert info.probed is False
    assert info.exists is True and info.managed is True
    assert info.version is None


def test_an_unprobed_seed_is_read_optimistically(tmp_path, monkeypatch):
    """Claiming "unsupported Python" for a second on every open would be a
    lie the user acts on; an interpreter on disk is assumed usable until
    the worker's probe lands."""
    plugin_dir = str(tmp_path)
    exe = inst.get_venv_python_path(inst.venv_location(plugin_dir))
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    with open(exe, "wb") as f:
        f.write(b"#!")
    info = ss.env_seed(plugin_dir, None)
    assert ss.env_ready(info) is True
    assert ss.blocking_reason(info) is None
    assert ss.env_state_text(info) == ss.TXT_ENV_CHECKING
    assert ss.button_states(info, [], None)["run_button"] is True


def test_an_absent_interpreter_is_still_blocked_without_probing(tmp_path):
    info = ss.env_seed(str(tmp_path), None)
    assert info.probed is False and info.exists is False
    assert ss.env_ready(info) is False
    assert ss.blocking_reason(info) == ss.TXT_BLOCK_NO_ENV
    assert ss.env_state_text(info) == ss.TXT_ENV_NONE


def test_env_seed_carries_the_resolved_dependency_verdict(tmp_path):
    """installer.resolve_environment already paid for the dependency probe
    at plugin load; reusing its verdict keeps a configured user's detail
    line from flickering through "dependencies not checked"."""
    plugin_dir = str(tmp_path)
    exe = inst.get_venv_python_path(inst.venv_location(plugin_dir))
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    with open(exe, "wb") as f:
        f.write(b"#!")
    seeded = ss.env_seed(plugin_dir, None,
                         {"python": exe, "status": "ready"})
    assert seeded.deps_ok is True
    # a verdict about a DIFFERENT interpreter is never carried over
    other = ss.env_seed(plugin_dir, None,
                        {"python": str(tmp_path / "elsewhere"),
                         "status": "byo"})
    assert other.deps_ok is None


def test_env_detail_text_survives_a_missing_usage_map(tmp_path):
    """The sizes come from the probe worker, so the first paint has none —
    that must read as "no size yet", not as an error."""
    info = ss.env_seed(str(tmp_path), None)
    assert "·" in ss.env_detail_text(info, {})
    assert "GB" not in ss.env_detail_text(info, {})


# --- text -------------------------------------------------------------------

@pytest.mark.parametrize("n,text", [
    (0, "0 bytes"),
    (1, "1 bytes"),
    (1023, "1023 bytes"),
    (1024, "1 KB"),
    (66_060_288, "63 MB"),
    (163_577_856, "156 MB"),
    (2_040_109_466, "1.9 GB"),
])
def test_human_bytes(n, text):
    assert ss.human_bytes(n) == text


def test_human_bytes_is_defensive():
    assert ss.human_bytes(None) == "0 bytes"
    assert ss.human_bytes(-5) == "0 bytes"


def test_env_state_text_unsupported_and_missing_deps():
    assert ss.env_state_text(_info(version=(3, 12))) == (
        "Python 3.12 — unsupported")
    assert ss.env_state_text(_info(deps_ok=False)) == "Dependencies missing"
    assert ss.env_state_text(_info(version=(3, 11, 15))) == (
        "Ready — Python 3.11.15")


def test_env_location_text_names_the_folder_and_the_uninstall_caveat(
        tmp_path):
    """The user's report: '<profile>/winmol stays untouched after
    deinstalling'. QGIS has no uninstall hook, so the only honest
    remedy is to say where it is and how to get rid of it."""
    root = inst.managed_root(str(tmp_path))
    text = ss.env_location_text(str(tmp_path))
    assert root in text
    assert "does NOT remove it" in text
    assert "Delete environment" in text
    # the size is appended only once the probe worker has measured it
    assert "GB" not in text
    sized = ss.env_location_text(
        str(tmp_path), {"venv": 2_040_109_466, "runtime": 66_060_288})
    assert "(2.0 GB)" in sized


def test_env_location_text_is_read_only(tmp_path):
    """It must not create the folder it describes."""
    before = sorted(os.listdir(tmp_path))
    ss.env_location_text(str(tmp_path), {"venv": 1})
    assert sorted(os.listdir(tmp_path)) == before


def test_env_detail_text_with_usage():
    info = _info()
    text = ss.env_detail_text(info, {"venv": 2_040_109_466,
                                     "runtime": 66_060_288})
    assert text == ("managed venv · dependencies present · 1.9 GB · "
                    "runtime 63 MB")


def test_models_summary_text():
    rows = [_row(entry_id=f"m{i}") for i in range(22)]
    rows[0].present = True
    rows[0].bytes_on_disk = 100_000_000
    rows[1].present = True
    rows[1].bytes_on_disk = 63_577_856
    assert ss.models_summary_text(rows) == (
        "Models on disk: 2 of 22 — 156 MB")


def test_models_summary_ignores_hidden_rows():
    rows = [_row(entry_id="a"), _row(entry_id="b", hidden=True)]
    assert ss.models_summary_text(rows) == "Models on disk: 0 of 1 — 0 bytes"


def test_state_text_never_calls_an_unpinned_file_verified():
    unpinned = ss.state_text(_row(state="unpinned"))
    assert unpinned == "on disk — no checksum published"
    assert "verified" not in unpinned
    assert ss.state_text(_row(state="corrupt")) == "CORRUPT — re-download"


# --- deletion_plan ----------------------------------------------------------

def test_deletion_plan_managed(tmp_path):
    plugin_dir = str(tmp_path)
    venv = inst.venv_location(plugin_dir)
    os.makedirs(venv)
    exe = os.path.join(venv, "bin", "python")
    os.makedirs(os.path.dirname(exe))
    with open(exe, "wb") as f:
        f.write(b"#!")
    plan = ss.deletion_plan(plugin_dir, exe, remove_venv=True)
    assert plan["kind"] == "managed"
    assert plan["paths"] == [venv]
    assert plan["clears_setting"] is True


def test_deletion_plan_byo_keeps_everything(tmp_path):
    plugin_dir = str(tmp_path)
    os.makedirs(inst.venv_location(plugin_dir))
    byo = tmp_path / "conda" / "bin" / "python"
    byo.parent.mkdir(parents=True)
    byo.write_bytes(b"#!")
    plan = ss.deletion_plan(plugin_dir, str(byo), remove_venv=True)
    assert plan["kind"] == "byo"
    assert plan["paths"] == []
    assert plan["clears_setting"] is False


def test_deletion_plan_none_when_nothing_is_configured(tmp_path):
    plan = ss.deletion_plan(str(tmp_path), None, remove_venv=True,
                            remove_runtime=True, remove_models=True)
    assert plan["kind"] == "none"
    assert plan["paths"] == []
    assert plan["clears_setting"] is False


def test_deletion_plan_lists_only_what_exists(tmp_path):
    plugin_dir = str(tmp_path)
    os.makedirs(inst.venv_location(plugin_dir))
    os.makedirs(os.path.join(plugin_dir, inst.MODELS_PATH))
    plan = ss.deletion_plan(plugin_dir, None, remove_venv=True,
                            remove_runtime=True, remove_models=True)
    assert plan["kind"] == "managed"
    assert plan["paths"] == [inst.venv_location(plugin_dir),
                             os.path.join(plugin_dir, inst.MODELS_PATH)]
    assert plan["clears_setting"] is False


@pytest.mark.parametrize("module", ["setup_state.py", "model_status.py"])
def test_the_decision_modules_import_nothing_from_qt(module):
    """They must stay usable on a machine with no QGIS — that is the
    whole reason the Setup tab's logic lives outside the dialog."""
    import ast
    src = open(os.path.join(REPO, "plugin_utils", module)).read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = [name for name in imported
              if name.split(".")[0] in ("qgis", "PyQt5", "PyQt6", "PyQt")]
    assert not banned, f"{module} imports Qt/QGIS: {banned}"
