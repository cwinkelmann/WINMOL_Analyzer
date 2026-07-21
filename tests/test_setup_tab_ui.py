"""The Setup tab, pinned statically.

The dialog cannot be instantiated here (no qgis/PyQt in the test env), so
this file asserts the two things that ARE statically knowable and that
silently rot otherwise:

* the .ui XML — the widget tree, the tab order, and the user's actual
  requirement ("env setup, deletion and model download leave the
  'Detect stems from UAV images' tab") expressed as a regression test;
* the dialog's AST — that every wire named in set_connections exists,
  that no integer literal is ever handed to log_widget.setCurrentIndex
  again, and that nothing blocking runs on the GUI thread.

Anything needing a live QGIS is in the PR's manual checklist, not here.
"""

import ast
import os
import xml.etree.ElementTree as ET

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_FILE = os.path.join(REPO, "winmol_analyzer_dialog_base.ui")
DIALOG_FILE = os.path.join(REPO, "winmol_analyzer_dialog.py")


# --- .ui helpers ------------------------------------------------------------

def _root():
    return ET.parse(UI_FILE).getroot()


def _parents(root):
    return {child: parent for parent in root.iter() for child in parent}


def _named(root):
    """{name: [element, ...]} over widgets, layouts, spacers, actions."""
    out = {}
    for element in root.iter():
        if element.tag not in ("widget", "layout", "spacer"):
            continue
        name = element.get("name")
        if name:
            out.setdefault(name, []).append(element)
    return out


def _subtree_names(element):
    names = set()
    for node in element.iter():
        if node.tag in ("widget", "layout", "spacer") and node.get("name"):
            names.add(node.get("name"))
    return names


def _tab_pages():
    root = _root()
    tabs = [w for w in root.iter("widget") if w.get("class") == "QTabWidget"]
    assert len(tabs) == 1, "expected exactly one QTabWidget (log_widget)"
    assert tabs[0].get("name") == "log_widget"
    return [child for child in tabs[0] if child.tag == "widget"]


# --- tab order (the regression that _show_tab protects) ---------------------

def test_the_tab_widget_has_exactly_three_pages_in_order():
    pages = _tab_pages()
    assert [p.get("name") for p in pages] == ["tab", "tab_setup", "tab_2"]
    titles = [p.find("attribute[@name='title']/string").text for p in pages]
    assert titles == ["Detection", "Setup", "Log"]


def test_no_integer_literal_reaches_set_current_index():
    """setCurrentIndex(1) used to mean "the Log tab". With Setup inserted
    at index 1 it would silently mean "the Setup tab", so the literal is
    banned outright and _show_tab(page) resolves the index."""
    tree = ast.parse(open(DIALOG_FILE, encoding="utf-8").read())
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


def test_show_tab_resolves_the_index_from_the_page():
    node = _function_node("_show_tab")
    called = {c.func.attr for c in ast.walk(node)
              if isinstance(c, ast.Call) and hasattr(c.func, "attr")}
    assert "indexOf" in called and "setCurrentIndex" in called


# --- the widget tree --------------------------------------------------------

SETUP_WIDGETS = (
    "env_state_label", "env_path_label", "env_detail_label",
    "env_create_button", "env_choose_button", "env_repair_button",
    "env_delete_button", "models_summary_label", "models_dir_label",
    "models_variants_checkBox", "models_refresh_button",
    "models_treeWidget", "models_download_button",
    "models_download_default_button", "models_verify_button",
    "models_delete_button", "models_open_folder_button",
    "setup_intro_label", "setup_ready_label", "setup_go_detect_button",
    "setup_status_label", "setup_progress_bar", "setup_detail_log",
    "setup_open_log_button",
)
DETECTION_ADDITIONS = (
    "setup_banner_widget", "setup_banner_label", "setup_banner_button",
    "model_variant_widget", "horizontalLayout_variant",
)


@pytest.mark.parametrize("name", SETUP_WIDGETS + DETECTION_ADDITIONS)
def test_every_wired_object_name_exists_exactly_once(name):
    found = _named(_root()).get(name, [])
    assert len(found) == 1, (
        f"{name} appears {len(found)} times in the .ui; the Python wiring "
        "binds it by name")


def test_the_setup_page_carries_no_detection_parameter():
    """The user's requirement, executable: per-run parameters do not move
    to Setup."""
    page = [p for p in _tab_pages() if p.get("name") == "tab_setup"][0]
    forbidden = ("uav_", "output_", "minlength_", "maxdistance_",
                 "tolerance_", "maxtree_", "tileside_", "image_")
    strays = sorted(n for n in _subtree_names(page)
                    if n.startswith(forbidden)
                    or n in ("model_comboBox", "model_lineEdit"))
    assert not strays, f"detection parameters leaked onto Setup: {strays}"


def test_the_detection_page_carries_no_setup_control():
    """The other half of the same requirement: environment setup,
    environment deletion and model download leave the detection tab. Only
    the navigation banner and the variant placeholder stay."""
    page = [p for p in _tab_pages() if p.get("name") == "tab"][0]
    allowed = set(DETECTION_ADDITIONS) | {
        "horizontalLayout_setup_banner", "horizontalSpacer_setup_banner"}
    strays = sorted(n for n in _subtree_names(page)
                    if n.startswith(("env_", "models_", "setup_"))
                    and n not in allowed)
    assert not strays, f"setup controls still on the detection tab: {strays}"


def test_the_banner_starts_hidden():
    widget = _named(_root())["setup_banner_widget"][0]
    visible = widget.find("property[@name='visible']/bool")
    assert visible is not None and visible.text == "false"


def test_the_detail_log_is_capped():
    """A verbose pip run streams thousands of lines through it."""
    widget = _named(_root())["setup_detail_log"][0]
    assert widget.find("property[@name='maximumBlockCount']/number") \
        is not None
    read_only = widget.find("property[@name='readOnly']/bool")
    assert read_only is not None and read_only.text == "true"


def test_the_model_tree_is_four_columns_and_single_selection():
    widget = _named(_root())["models_treeWidget"][0]
    assert len(widget.findall("column")) == 4
    headers = [c.find("property[@name='text']/string").text
               for c in widget.findall("column")]
    assert headers == ["Model", "Precision", "Size", "Status"]
    mode = widget.find("property[@name='selectionMode']/enum")
    assert mode is not None
    assert mode.text == "QAbstractItemView::SingleSelection", (
        "multi-select would put gigabytes behind one Yes/No")


@pytest.mark.parametrize("name", ["env_delete_button", "models_delete_button"])
def test_destructive_buttons_announce_a_confirmation(name):
    widget = _named(_root())[name][0]
    text = widget.find("property[@name='text']/string").text
    assert text.endswith("…"), (
        f"{name} deletes data; its label must carry the ellipsis "
        "convention for 'this opens a dialog'")


def test_the_setup_page_has_its_own_progress_bar():
    """A model download must never animate the inference run's bar."""
    names = _named(_root())
    assert "setup_progress_bar" in names and "progress_bar" in names
    page = [p for p in _tab_pages() if p.get("name") == "tab_setup"][0]
    assert "setup_progress_bar" in _subtree_names(page)
    assert "progress_bar" not in _subtree_names(page)


def test_the_output_defaults_are_untouched_and_still_on_detection():
    page = [p for p in _tab_pages() if p.get("name") == "tab"][0]
    names = _subtree_names(page)
    for box in ("output_checkBox_stem", "output_checkBox_trees",
                "output_checkBox_nodes"):
        assert box in names
        widget = _named(_root())[box][0]
        checked = widget.find("property[@name='checked']/bool")
        assert checked is not None and checked.text == "true"


def test_the_shared_button_row_stayed_outside_the_tabs():
    root = _root()
    parents = _parents(root)
    grid = _named(root)["gridLayout_3"][0]
    ancestors = []
    node = parents.get(grid)
    while node is not None:
        ancestors.append(node.get("name"))
        node = parents.get(node)
    assert "tab" not in ancestors and "tab_setup" not in ancestors
    for name in ("progress_bar", "run_button", "cancel_button",
                 "close_button", "help_button"):
        assert name in _subtree_names(grid)


# --- AST: the wiring --------------------------------------------------------

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


NEW_SLOTS = (
    "_setup_create_env", "_setup_repair_env", "_setup_delete_env",
    "_setup_download_selected", "_setup_download_recommended",
    "_setup_verify_selected", "_setup_delete_selected",
    "_refresh_model_tree", "_refresh_setup_actions", "_on_tab_changed",
    "_open_models_dir", "_refresh_setup_state", "_enter_first_run",
    "_on_env_removed", "_on_models_changed", "_start_env_job",
    "_start_dl_job", "_show_tab",
)


@pytest.mark.parametrize("name", NEW_SLOTS)
def test_every_new_slot_exists(name):
    assert name in _function_names()


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


def test_set_connections_wires_the_setup_tab():
    node = _function_node("set_connections")
    called = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    assert "_connect_setup_tab" in called


@pytest.mark.parametrize("slot", [
    "_setup_create_env", "_setup_repair_env", "_setup_delete_env",
    "_setup_download_selected", "_setup_download_recommended",
    "_setup_verify_selected", "_setup_delete_selected",
    "_refresh_model_tree", "_refresh_setup_actions", "_on_tab_changed",
    "_open_models_dir",
])
def test_each_slot_is_actually_connected(slot):
    wired = {row[2] for row in _connection_table()}
    assert slot in wired, f"{slot} exists but nothing triggers it"


def test_every_connected_widget_and_slot_really_exists():
    ui_names = set(_named(_root()))
    functions = _function_names()
    for widget, _signal, slot in _connection_table():
        assert widget in ui_names, f"{widget} is not in the .ui"
        assert slot in functions, f"{slot} is not defined on the dialog"


def test_setup_buttons_constant_matches_the_ui():
    module = _module()
    for node in ast.walk(module):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SETUP_BUTTONS"
                for t in node.targets):
            names = [el.value for el in node.value.elts]
            break
    else:                                          # pragma: no cover
        pytest.fail("SETUP_BUTTONS constant not found")
    ui_names = set(_named(_root()))
    missing = [n for n in names if n not in ui_names]
    assert not missing, f"SETUP_BUTTONS names absent from the .ui: {missing}"
    assert len(names) == 9


def test_the_old_env_button_is_gone():
    """It was replaced by env_create_button + env_choose_button on the
    Setup tab; no getattr-guarded dead reference may survive."""
    for node in ast.walk(_module()):
        if isinstance(node, ast.Constant) and node.value == "env_button":
            pytest.fail(
                f"'env_button' still referenced at line {node.lineno}")
    assert "_refresh_env_status" not in _function_names()


# --- AST: the anti-freeze contract -----------------------------------------

GUI_THREAD_FUNCTIONS = ("_refresh_setup_state", "_refresh_model_tree",
                        "_refresh_setup_actions")

BANNED_CALLS = (
    "run", "Popen", "check_call", "check_output",      # subprocess
    "rmtree", "remove_model", "verify_entry", "ensure_model",
    "setup_environment", "install_requirements",
    "install_requirements_into", "download_model", "download_models",
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


def test_env_removal_from_a_gui_slot_is_only_ever_a_dry_run():
    """The real deletion goes through EnvRemoveWorker; the only thing a
    slot may do inline is price it."""
    for node in ast.walk(_module()):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "remove_environment":
            continue
        flags = {kw.arg: kw.value for kw in node.keywords}
        assert "dry_run" in flags and flags["dry_run"].value is True, (
            f"remove_environment at line {node.lineno} deletes on the GUI "
            "thread; hand it to EnvRemoveWorker instead")


@pytest.mark.parametrize("starter", ["_start_env_job", "_start_dl_job"])
def test_the_job_starters_keep_the_thread_discipline(starter):
    node = _function_node(starter)
    # thread.quit / worker.deleteLater are passed as slots, not called
    referenced = {n.attr for n in ast.walk(node)
                  if isinstance(n, ast.Attribute)}
    for required in ("moveToThread", "quit", "deleteLater", "start",
                     "finished"):
        assert required in referenced, f"{starter} never uses {required}"


def test_terminal_slots_are_passed_per_worker_not_per_pair():
    """A successful DELETE handled as a successful BUILD would call
    _set_python() and re-point the plugin at the tree it just erased."""
    for starter in ("_start_env_job", "_start_dl_job"):
        args = [a.arg for a in _function_node(starter).args.args]
        assert "on_done" in args and "on_failed" in args
    # ...and the remove path names its own terminal slots
    node = _function_node("_setup_delete_env")
    referenced = {n.attr for n in ast.walk(node)
                  if isinstance(n, ast.Attribute)}
    assert "_on_env_removed" in referenced
    assert "_on_env_ready" not in referenced


def test_shutdown_still_reaps_both_pairs():
    node = _function_node("_shutdown_threads")
    source = ast.dump(node)
    assert "_env_thread" in source and "_dl_thread" in source
    reaps = [c for c in ast.walk(node)
             if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "_reap_thread"]
    assert len(reaps) >= 3, (
        "the inference, env and download pairs must all be reaped; a "
        "missed one qFatals QGIS on close")


def test_the_download_on_run_rerun_is_gated():
    """Downloading a model from the Setup tab must not start an
    inference run."""
    node = _function_node("_on_model_downloaded")
    guarded = [n for n in ast.walk(node)
               if isinstance(n, ast.If)
               and "_dl_then_run" in ast.dump(n.test)]
    assert guarded, "_on_model_downloaded re-runs unconditionally"
    reruns = [c for c in ast.walk(node)
              if isinstance(c, ast.Call)
              and getattr(c.func, "attr", None) == "singleShot"]
    assert reruns and all(
        any(r is c for r in ast.walk(guarded[0])) for c in reruns)


def test_the_setup_progress_bar_is_the_only_one_a_setup_job_drives():
    node = _function_node("_on_setup_progress")
    names = {n.value for n in ast.walk(node)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "setup_progress_bar" in names
    assert "progress_bar" not in names


def test_the_heartbeat_exists():
    """The literal cure for 'it looks frozen': a status line that is
    never empty and a 1 Hz tick that visibly moves."""
    assert "_tick_setup_status" in _function_names()
    node = _function_node("_tick_setup_status")
    assert any(isinstance(n, ast.Call)
               and getattr(n.func, "attr", None) == "setText"
               for n in ast.walk(node))
    init = _function_node("__init__")
    assert "_setup_ticker" in ast.dump(init)


def test_no_modal_is_popped_at_construction():
    """The on-open QMessageBox is exactly what BUGS.md complains about."""
    init = _function_node("__init__")
    deferred = [c for c in ast.walk(init)
                if isinstance(c, ast.Call)
                and getattr(c.func, "attr", None) == "singleShot"]
    assert deferred, "__init__ no longer schedules the first-run entry"
    targets = {getattr(a, "attr", None)
               for c in deferred for a in c.args} - {None}
    assert targets == {"_enter_first_run"}
    assert "choose_environment" not in targets
