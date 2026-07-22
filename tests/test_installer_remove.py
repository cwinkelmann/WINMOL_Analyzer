"""Deleting the WINMOL environment — the destructive half of the Setup tab.

Nothing in the tree deleted a venv, a runtime or a model before this, so
every guarantee here is net-new and worth pinning hard:

* a dry run reports exactly what a real run would remove, and removes
  nothing (the confirmation dialog's "frees N GB" comes from this);
* only paths strictly inside the managed tree are ever passed to rmtree;
* a bring-your-own interpreter is never deleted and never silently
  un-configured;
* the ready marker dies BEFORE the first rmtree, so a half-deleted venv
  degrades to "needs rebuild" instead of being blessed by is_ready();
* nothing raises — failures are collected and reported honestly.
"""

import os
import shutil
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plugin_utils import installer as inst        # noqa: E402


def _managed_tree(tmp_path, venv_bytes=2048, runtime_bytes=1024):
    """A fake plugin dir with a venv, a py311 runtime and a models dir."""
    plugin_dir = str(tmp_path)
    venv = inst.venv_location(plugin_dir)
    os.makedirs(os.path.join(venv, "lib"), exist_ok=True)
    with open(os.path.join(venv, "lib", "big.so"), "wb") as f:
        f.write(b"\0" * venv_bytes)
    inst._write_marker(venv)
    runtime = os.path.join(inst.managed_root(plugin_dir), "py311")
    os.makedirs(runtime, exist_ok=True)
    with open(os.path.join(runtime, "python"), "wb") as f:
        f.write(b"\0" * runtime_bytes)
    models = os.path.join(plugin_dir, inst.MODELS_PATH)
    os.makedirs(models, exist_ok=True)
    return plugin_dir, venv, runtime, models


# --- helpers ----------------------------------------------------------------

def test_directory_size_counts_files_and_tolerates_absence(tmp_path):
    assert inst.directory_size(str(tmp_path / "nope")) == 0
    d = tmp_path / "d" / "sub"
    d.mkdir(parents=True)
    (d / "a").write_bytes(b"x" * 100)
    (tmp_path / "d" / "b").write_bytes(b"y" * 23)
    assert inst.directory_size(str(tmp_path / "d")) == 123


def test_invalidate_marker_makes_is_ready_false(tmp_path, monkeypatch):
    venv = str(tmp_path / "venv")
    os.makedirs(venv)
    inst._write_marker(venv)
    monkeypatch.setattr(inst, "get_venv_python_path", lambda _p: __file__)
    monkeypatch.setattr(inst, "_python_version", lambda _e: inst.MIN_PY)
    assert inst.is_ready(venv) is True
    assert inst.invalidate_marker(venv) is True
    assert inst.is_ready(venv) is False
    # idempotent: a second call is a no-op, not an error
    assert inst.invalidate_marker(venv) is False


def test_path_is_inside_uses_realpath_and_normcase(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    assert inst.path_is_inside(str(root / "sub"), str(root))
    assert inst.path_is_inside(str(root), str(root))
    assert not inst.path_is_inside(str(other), str(root))
    # a sibling whose name merely starts with the root's name
    sibling = tmp_path / "root_backup"
    sibling.mkdir()
    assert not inst.path_is_inside(str(sibling), str(root))
    if hasattr(os, "symlink"):
        link = tmp_path / "link"
        try:
            os.symlink(str(root / "sub"), str(link))
        except (OSError, NotImplementedError):      # pragma: no cover
            pytest.skip("symlinks unavailable")
        assert inst.path_is_inside(str(link), str(root))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_a_symlink_leaf_inside_the_root_counts_as_inside(tmp_path):
    """The exact shape of a POSIX venv's bin/python: an entry INSIDE the
    root whose target is an interpreter outside it. Realpath'ing that
    leaf is what made WINMOL classify its own managed venv as
    bring-your-own."""
    root = tmp_path / "venv"
    (root / "bin").mkdir(parents=True)
    outside = tmp_path / "base" / "bin"
    outside.mkdir(parents=True)
    (outside / "python3.11").write_text("#!/bin/sh\n")
    leaf = root / "bin" / "python"
    try:
        os.symlink(str(outside / "python3.11"), str(leaf))
    except (OSError, NotImplementedError):          # pragma: no cover
        pytest.skip("symlinks unavailable")
    assert inst.path_is_inside(str(leaf), str(root))
    # ...and the target itself is still outside: a bring-your-own
    # interpreter must never become deletable.
    assert not inst.path_is_inside(str(outside / "python3.11"), str(root))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_a_symlink_pointing_out_of_the_root_is_still_outside(tmp_path):
    """A link that LIVES outside and points in is not inside."""
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    escape = outside_dir / "python"
    try:
        os.symlink(str(outside_dir / "real"), str(escape))
    except (OSError, NotImplementedError):          # pragma: no cover
        pytest.skip("symlinks unavailable")
    assert not inst.path_is_inside(str(escape), str(root))


def test_a_real_venv_is_recognised_as_managed(tmp_path):
    """The end-to-end regression: build an actual venv and assert the
    Setup tab would offer to delete it.

    Before this fix, on every macOS and Linux install deletion_plan()
    returned kind='byo' — so "Delete environment…" routed to "forget the
    interpreter" and removed nothing, and the leftover
    <profile>/winmol the user complained about could not be cleaned up
    from inside the plugin at all.
    """
    import venv as venv_mod

    from plugin_utils import setup_state

    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    venv_dir = inst.venv_location(str(plugin_dir))
    try:
        venv_mod.EnvBuilder(with_pip=False, symlinks=True).create(venv_dir)
    except Exception as exc:                        # pragma: no cover
        pytest.skip(f"cannot build a venv here: {exc}")
    exe = inst.get_venv_python_path(venv_dir)
    assert os.path.exists(exe)
    # the shape that broke it: an absolute symlink out of the venv
    assert os.path.islink(exe) or sys.platform == "win32"

    assert inst.path_is_inside(exe, venv_dir) is True
    plan = setup_state.deletion_plan(str(plugin_dir), exe)
    assert plan["kind"] == "managed"
    assert venv_dir in plan["paths"]
    assert plan["clears_setting"] is True
    assert setup_state.env_seed(str(plugin_dir), exe).managed is True

    # ...and a bring-your-own interpreter is still bring-your-own.
    byo = setup_state.deletion_plan(str(plugin_dir), sys.executable)
    assert byo["kind"] == "byo"


# --- dry run ----------------------------------------------------------------

def test_dry_run_reports_paths_and_bytes_and_deletes_nothing(tmp_path):
    plugin_dir, venv, runtime, _models = _managed_tree(tmp_path)
    result = inst.remove_environment(plugin_dir, remove_venv=True,
                                     remove_runtime=True, dry_run=True)
    planned = dict(result["planned"])
    assert set(planned) == {venv, runtime}
    # the venv also carries the .winmol_ready marker; sizes are the real
    # on-disk totals, not the payload we wrote
    assert planned[venv] == inst.directory_size(venv) > 2048
    assert planned[runtime] == 1024
    assert result["freed_bytes"] == planned[venv] + planned[runtime]
    assert result["removed"] == [] and result["failed"] == []
    assert os.path.isdir(venv) and os.path.isdir(runtime)
    # and the marker survives a dry run
    assert os.path.exists(inst._marker_path(venv))


def test_dry_run_and_real_run_agree(tmp_path):
    plugin_dir, _venv, _runtime, _models = _managed_tree(tmp_path)
    dry = inst.remove_environment(plugin_dir, remove_venv=True,
                                  remove_runtime=True, dry_run=True)
    real = inst.remove_environment(plugin_dir, remove_venv=True,
                                   remove_runtime=True)
    assert real["freed_bytes"] == dry["freed_bytes"]
    assert sorted(real["removed"]) == sorted(p for p, _s in dry["planned"])


# --- selective removal ------------------------------------------------------

def test_remove_venv_leaves_models_and_runtime(tmp_path):
    plugin_dir, venv, runtime, models = _managed_tree(tmp_path)
    (tmp_path / "models" / "keep.onnx").write_bytes(b"m" * 10)
    result = inst.remove_environment(plugin_dir, remove_venv=True)
    assert not os.path.exists(venv)
    assert os.path.isdir(runtime)
    assert os.path.exists(os.path.join(models, "keep.onnx"))
    assert result["failed"] == []


def test_remove_models_leaves_the_venv(tmp_path, monkeypatch):
    plugin_dir, venv, _runtime, models = _managed_tree(tmp_path)
    known = os.path.join(models, "known.onnx")
    with open(known, "wb") as f:
        f.write(b"k" * 64)
    with open(known + ".part", "wb") as f:
        f.write(b"k" * 4)
    stranger = os.path.join(models, "mine.onnx")
    with open(stranger, "wb") as f:
        f.write(b"s" * 8)
    monkeypatch.setattr(inst, "_plan_models", _fake_plan_models(known))

    result = inst.remove_environment(plugin_dir, remove_venv=False,
                                     remove_models=True)
    assert os.path.isdir(venv)
    assert not os.path.exists(known)
    # a file the registry does not know about is never touched
    assert os.path.exists(stranger)
    assert result["freed_bytes"] == 64


def _fake_plan_models(path):
    """Stand in for the registry-backed model removal with one known file
    (the real thing is covered by tests/test_model_registry.py)."""
    def _plan(_plugin_dir, _models_dir, dry_run, progress=None):
        try:
            size = os.path.getsize(path)
        except OSError:
            return {"planned": [], "removed": [], "failed": [],
                    "freed_bytes": 0}
        if dry_run:
            return {"planned": [(path, size)], "removed": [], "failed": [],
                    "freed_bytes": size}
        os.remove(path)
        return {"planned": [(path, size)], "removed": [path], "failed": [],
                "freed_bytes": size}
    return _plan


# --- safety -----------------------------------------------------------------

def test_refuses_a_path_outside_the_managed_tree(tmp_path, monkeypatch):
    plugin_dir = str(tmp_path / "plugin")
    os.makedirs(plugin_dir)
    outside = tmp_path / "elsewhere" / "winmol_venv"
    outside.mkdir(parents=True)
    (outside / "f").write_bytes(b"x")
    monkeypatch.setattr(inst, "venv_location", lambda _p: str(outside))
    monkeypatch.setattr(shutil, "rmtree", _never_called)

    result = inst.remove_environment(plugin_dir, remove_venv=True)
    assert result["removed"] == []
    assert result["planned"] == []
    assert [p for p, _m in result["failed"]] == [str(outside)]
    assert "refused" in result["failed"][0][1]
    assert os.path.isdir(outside)


def test_refuses_the_managed_root_itself(tmp_path, monkeypatch):
    plugin_dir = str(tmp_path)
    monkeypatch.setattr(inst, "venv_location", lambda p: p)
    monkeypatch.setattr(shutil, "rmtree", _never_called)
    result = inst.remove_environment(plugin_dir, remove_venv=True)
    assert result["removed"] == []
    assert "refused" in result["failed"][0][1]


def _never_called(*_a, **_k):
    raise AssertionError("rmtree must not be called for a refused path")


def test_rmtree_failure_is_collected_not_raised(tmp_path, monkeypatch):
    plugin_dir, venv, _runtime, _models = _managed_tree(tmp_path)

    def _boom(*_a, **_k):
        raise OSError(32, "The process cannot access the file")
    monkeypatch.setattr(shutil, "rmtree", _boom)

    result = inst.remove_environment(plugin_dir, remove_venv=True)
    assert result["removed"] == []
    assert result["freed_bytes"] == 0
    assert [p for p, _m in result["failed"]] == [venv]
    assert os.path.isdir(venv)


def test_marker_is_invalidated_before_any_rmtree(tmp_path, monkeypatch):
    plugin_dir, venv, _runtime, _models = _managed_tree(tmp_path)
    monkeypatch.setattr(inst, "get_venv_python_path", lambda _p: __file__)
    monkeypatch.setattr(inst, "_python_version", lambda _e: inst.MIN_PY)
    assert inst.is_ready(venv) is True
    monkeypatch.setattr(shutil, "rmtree", lambda *_a, **_k: None)

    inst.remove_environment(plugin_dir, remove_venv=True)
    # rmtree did nothing, yet the environment must no longer claim ready
    assert os.path.isdir(venv)
    assert inst.is_ready(venv) is False


# --- the bring-your-own guard ----------------------------------------------

def test_clear_setting_true_for_an_interpreter_inside_the_venv(tmp_path):
    plugin_dir, venv, _runtime, _models = _managed_tree(tmp_path)
    exe = os.path.join(venv, "bin", "python")
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    with open(exe, "wb") as f:
        f.write(b"#!")
    result = inst.remove_environment(plugin_dir, remove_venv=True,
                                     configured_exe=exe, dry_run=True)
    assert result["clear_setting"] is True


def test_clear_setting_false_for_a_byo_interpreter(tmp_path):
    plugin_dir, _venv, _runtime, _models = _managed_tree(tmp_path)
    byo = tmp_path / "conda" / "envs" / "gis" / "bin" / "python"
    byo.parent.mkdir(parents=True)
    byo.write_bytes(b"#!")
    result = inst.remove_environment(plugin_dir, remove_venv=True,
                                     configured_exe=str(byo), dry_run=True)
    assert result["clear_setting"] is False


def test_clear_setting_false_when_the_venv_is_kept(tmp_path):
    plugin_dir, venv, _runtime, _models = _managed_tree(tmp_path)
    exe = os.path.join(venv, "bin", "python")
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    with open(exe, "wb") as f:
        f.write(b"#!")
    result = inst.remove_environment(plugin_dir, remove_venv=False,
                                     remove_runtime=True,
                                     configured_exe=exe, dry_run=True)
    assert result["clear_setting"] is False


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_clear_setting_sees_through_a_symlinked_interpreter(tmp_path):
    plugin_dir, venv, _runtime, _models = _managed_tree(tmp_path)
    real = os.path.join(venv, "bin", "python3.11")
    os.makedirs(os.path.dirname(real), exist_ok=True)
    with open(real, "wb") as f:
        f.write(b"#!")
    link = str(tmp_path / "python-link")
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):          # pragma: no cover
        pytest.skip("symlinks unavailable")
    result = inst.remove_environment(plugin_dir, remove_venv=True,
                                     configured_exe=link, dry_run=True)
    assert result["clear_setting"] is True


def test_never_raises_on_a_nonexistent_tree(tmp_path):
    plugin_dir = str(tmp_path / "gone")
    result = inst.remove_environment(plugin_dir, remove_venv=True,
                                     remove_runtime=True)
    assert result["removed"] == [] and result["failed"] == []
    assert result["freed_bytes"] == 0
