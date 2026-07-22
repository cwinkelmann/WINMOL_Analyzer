"""STATIC LINT RULES for the dialog — not behavioural tests.

Every assertion in this file parses source (the dialog's AST, the .ui
XML) instead of executing it. That is a bad way to test software and it
is done here for exactly one reason: ``winmol_analyzer_dialog.py``
imports ``qgis`` and ``PyQt5``, neither of which can be instantiated in
this environment, so the only alternative to a static rule is no check
at all.

They are therefore held to a much higher bar than a normal test: each of
the five rules below encodes a *specific bug that shipped*, in a failure
mode that costs a live QGIS session to rediscover. A rule that merely
restates the code's shape (this name exists, this call comes before that
one, this label is worded so) is not worth the refactors it blocks and
does not belong here.

* a hardcoded tab index shipped a bug when a tab was inserted;
* blocking work on the GUI thread froze QGIS twice;
* environment deletion on the GUI thread, and deletion from ``unload()``
  which QGIS also fires on disable/reload/quit, both erase a multi-GB
  venv;
* a wire naming a widget or a slot that does not exist raises only when
  the dialog is opened by a human.

Anything that can be exercised by running code belongs in a real test,
and anything needing a live QGIS belongs in the manual checklist.
"""

import ast
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_FILE = os.path.join(REPO, "winmol_analyzer_dialog_base.ui")
DIALOG_FILE = os.path.join(REPO, "winmol_analyzer_dialog.py")


def _module():
    return ast.parse(open(DIALOG_FILE, encoding="utf-8").read())


def _function_node(name):
    for node in ast.walk(_module()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{name} not found in winmol_analyzer_dialog.py")


def _function_names():
    return {node.name for node in ast.walk(_module())
            if isinstance(node, ast.FunctionDef)}


def _ui_names():
    """Every object name declared in the .ui."""
    import xml.etree.ElementTree as ET
    root = ET.parse(UI_FILE).getroot()
    return {element.get("name") for element in root.iter()
            if element.tag in ("widget", "layout", "spacer")
            and element.get("name")}


# --- 1. the tab index that shipped a bug ------------------------------------

def test_no_integer_literal_reaches_set_current_index():
    """setCurrentIndex(1) used to mean "the Log tab". Setup was inserted
    at index 1 and has since moved to index 0, so that literal would mean
    the Setup tab and then the Detection tab; it is banned outright and
    _show_tab(page) resolves the index from the widget."""
    tree = _module()
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if getattr(func, "attr", None) != "setCurrentIndex":
            continue
        # only the TAB widget: a combo box legitimately indexes by number
        if getattr(func.value, "attr", None) != "log_widget":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                offenders.append((getattr(node, "lineno", "?"), arg.value))
    assert not offenders, (
        "hardcoded tab index passed to setCurrentIndex at "
        f"{offenders}; go through _show_tab(page) instead")


# --- 2. the two freezes -----------------------------------------------------

GUI_THREAD_FUNCTIONS = ("_refresh_setup_state", "_refresh_model_tree",
                        "_refresh_setup_actions", "_env_snapshot",
                        "_env_usage", "_apply_blocking_reason",
                        "_scan_models", "__init__",
                        # The proactive GPU offer. run_process is in here
                        # because the pre-run question happens INSIDE it:
                        # a probe called from there would freeze the GUI
                        # at the exact moment the user pressed Run, which
                        # is the worst possible moment for it. Every one
                        # of these reads the cache EnvProbeWorker filled.
                        "run_process", "_gpu_offer_pre_flight",
                        "_apply_accel_nudge", "_gpu_prompt_dismissed",
                        "_remember_gpu_prompt", "_accel_status",
                        "_confirm_gpu_runtime")

BANNED_CALLS = (
    "run", "Popen", "check_call", "check_output",      # subprocess
    "rmtree", "remove_model", "verify_entry", "ensure_model",
    "setup_environment", "install_requirements",
    "install_requirements_into", "download_model", "download_models",
    # ...and the ones that let the freeze back in through a side door.
    # BANNED_CALLS used to be a list of names nobody had measured: the
    # Setup tab's repaint called env_info -> _python_version x2 +
    # _has_compute_deps (an `import onnxruntime, rasterio, geopandas`,
    # 1.16 s warm and bounded only by a 60 s timeout) and directory_size
    # (1.95 s over a 2.17 GB venv) — about 4 s of blocked GUI thread on
    # every dialog open and every Rescan, none of which this test could
    # see. Everything below belongs on EnvProbeWorker.
    "env_info", "directory_size", "_python_version", "_has_compute_deps",
    "is_ready", "resolve_environment", "remove_environment",
    "remove_all", "verify_file",
    # The accelerator verdict: detect_gpu shells out to nvidia-smi (which
    # blocks in an uninterruptible ioctl on a wedged driver) and
    # probe_runtime launches the child interpreter to import onnxruntime.
    # accelerator_STATUS is pure and stays allowed; accelerator_FROM_MACHINE
    # is the measuring one and belongs on EnvProbeWorker.
    "accelerator_from_machine", "detect_gpu", "probe_runtime",
    "verify_gpu_runtime", "distribution_installed",
)


def _gui_thread_nodes():
    for node in ast.walk(_module()):
        if isinstance(node, ast.FunctionDef) and (
                node.name.startswith("_setup_")
                or node.name in GUI_THREAD_FUNCTIONS):
            yield node


def test_nothing_blocking_runs_on_the_gui_thread():
    """pip on the GUI thread froze QGIS for a whole multi-minute install.
    It cannot come back through a Setup slot."""
    offenders = []
    for func in _gui_thread_nodes():
        for call in ast.walk(func):
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "attr", None) or \
                getattr(call.func, "id", None)
            if name in BANNED_CALLS:
                offenders.append((func.name, name, call.lineno))
    assert not offenders, (
        f"blocking work reachable from a GUI slot: {offenders}")


def test_env_removal_never_runs_in_the_dialog_at_all():
    """Not even the dry run. Pricing a deletion is a full directory walk
    of the venv — the same cost class as performing it — so both the
    pricing pass and the removal go through EnvRemoveWorker."""
    offenders = [node.lineno for node in ast.walk(_module())
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "attr", None) == "remove_environment"]
    assert not offenders, (
        f"remove_environment called on the GUI thread at {offenders}; hand "
        "it to EnvRemoveWorker (dry_run=True prices it) instead")
    # ...and the pricing pass really is a dry run, not a deletion.
    priced = [node for node in ast.walk(_function_node(
        "_start_deletion_pricing"))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "EnvRemoveWorker"]
    assert priced, "_start_deletion_pricing does not build an EnvRemoveWorker"
    flags = {kw.arg: kw.value for kw in priced[0].keywords}
    assert "dry_run" in flags and flags["dry_run"].value is True


# --- 3. the environment QGIS leaves behind ----------------------------------

def test_unload_never_deletes_the_environment():
    """QGIS has no uninstall hook: pyplugin_installer's uninstallPlugin is
    unloadPlugin() + removeDir(<plugin dir>), and unloadPlugin calls the
    plugin's unload() — the SAME callback fired on disable, on reload and
    at application shutdown. Deleting from there would wipe a multi-GB
    venv every time QGIS closes."""
    plugin_file = os.path.join(REPO, "winmol_analyzer.py")
    module = ast.parse(open(plugin_file, encoding="utf-8").read())
    unload = None
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == "unload":
            unload = node
    assert unload is not None
    destructive = {"rmtree", "remove", "unlink", "rmdir",
                   "remove_environment", "removeDir"}
    offenders = [call.lineno for call in ast.walk(unload)
                 if isinstance(call, ast.Call)
                 and getattr(call.func, "attr", None) in destructive]
    assert not offenders, (
        f"unload() deletes something at {offenders}; it cannot tell an "
        "uninstall from a disable/reload/quit")


# --- 4. .ui-vs-Python name consistency --------------------------------------

def _connection_table():
    """The SETUP_CONNECTIONS class constant, as (widget, signal, slot)."""
    for node in ast.walk(_module()):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "SETUP_CONNECTIONS" not in targets:
            continue
        return [tuple(el.value for el in item.elts)
                for item in node.value.elts]
    pytest.fail("SETUP_CONNECTIONS not found")


def test_every_connected_widget_and_slot_really_exists():
    """A wire naming a widget the .ui does not declare, or a slot the
    dialog does not define, raises AttributeError the first time a human
    opens the dialog and never before."""
    ui_names = _ui_names()
    functions = _function_names()
    for widget, _signal, slot in _connection_table():
        assert widget in ui_names, f"{widget} is not in the .ui"
        assert slot in functions, f"{slot} is not defined on the dialog"
