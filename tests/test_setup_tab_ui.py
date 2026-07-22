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
    """Setup is leftmost — the user's requirement, pinned. Every index
    that used to be written down as a number is now resolved from these
    names, so this assertion is the only place the order lives."""
    pages = _tab_pages()
    assert [p.get("name") for p in pages] == ["tab_setup", "tab", "tab_2"]
    titles = [p.find("attribute[@name='title']/string").text for p in pages]
    assert titles == ["Setup", "Detection", "Log"]


def test_the_dialog_opens_on_detection_not_on_setup():
    """The .ui's designer default is the one index a widget lookup cannot
    replace, so it is checked against the page order rather than a
    number. _enter_first_run may still send a fresh install to Setup."""
    root = _root()
    tabs = [w for w in root.iter("widget") if w.get("class") == "QTabWidget"]
    current = tabs[0].find("property[@name='currentIndex']/number")
    assert current is not None
    pages = [p.get("name") for p in _tab_pages()]
    assert pages[int(current.text)] == "tab"


def test_no_integer_literal_reaches_set_current_index():
    """setCurrentIndex(1) used to mean "the Log tab". Setup was inserted
    at index 1 and has since moved to index 0, so that literal would mean
    the Setup tab and then the Detection tab; it is banned outright and
    _show_tab(page) resolves the index from the widget."""
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


def _titles_by_object_name():
    return {p.get("name"): p.find("attribute[@name='title']/string").text
            for p in _tab_pages()}


@pytest.mark.parametrize("helper,title", [
    ("_go_to_detection", "Detection"),
    ("_go_to_log", "Log"),
    ("_go_to_setup", "Setup"),
])
def test_each_navigation_helper_opens_the_tab_it_is_named_after(helper, title):
    """Reordering tabs cannot be caught by "does it switch?" — only by
    "does it switch to the RIGHT one?". Each helper is read for the
    object name it hands to _show_tab, and that name is looked up in the
    .ui to confirm the page really carries the matching title."""
    node = _function_node(helper)
    targets = [arg.value
               for call in ast.walk(node) if isinstance(call, ast.Call)
               for arg in call.args
               if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
    assert len(targets) == 1, (
        f"{helper} must name exactly one page widget, found {targets}")
    titles = _titles_by_object_name()
    assert targets[0] in titles, f"{helper} targets a page not in the .ui"
    assert titles[targets[0]] == title, (
        f"{helper} opens the tab titled {titles[targets[0]]!r}, "
        f"not {title!r} — the tab order moved underneath it")


# --- the widget tree --------------------------------------------------------

SETUP_WIDGETS = (
    "env_state_label", "env_path_label", "env_detail_label",
    "env_create_button", "env_choose_button", "env_repair_button",
    "env_delete_button", "models_summary_label", "models_dir_label",
    "models_variants_checkBox", "models_refresh_button",
    "models_treeWidget", "models_download_button",
    "models_download_default_button", "models_verify_button",
    "models_delete_button", "models_open_folder_button",
    "env_location_label", "env_open_folder_button",
    "setup_intro_label", "setup_ready_label", "setup_go_detect_button",
    "setup_status_label", "setup_progress_bar", "setup_detail_log",
    "setup_open_log_button",
)
DETECTION_ADDITIONS = (
    "setup_banner_widget", "setup_banner_label", "setup_banner_button",
    "model_variant_widget", "horizontalLayout_variant",
    # The idle-GPU warning. It is a NAVIGATION banner like its sibling
    # above — a sentence and a button that opens Setup — not a setup
    # control, so it does not violate "setup leaves the detection tab".
    "accel_banner_widget", "accel_banner_label", "accel_banner_button",
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
        "horizontalLayout_setup_banner", "horizontalSpacer_setup_banner",
        "horizontalLayout_accel_banner", "horizontalSpacer_accel_banner"}
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


def _ancestor_names(root, element):
    parents = _parents(root)
    names, node = [], parents.get(element)
    while node is not None:
        names.append(node.get("name"))
        node = parents.get(node)
    return names


def test_the_shared_button_row_stayed_outside_the_tabs():
    root = _root()
    grid = _named(root)["gridLayout_3"][0]
    ancestors = _ancestor_names(root, grid)
    assert "tab" not in ancestors and "tab_setup" not in ancestors
    for name in ("progress_bar", "export_button", "cancel_button",
                 "close_button", "help_button"):
        assert name in _subtree_names(grid)


# --- Run at the top, Export at the bottom (BUGS.md "Button positions") ------

def _outer_items(root):
    """The direct children of the dialog's outermost QVBoxLayout, top to
    bottom, as the name of the single widget/layout each <item> holds."""
    outer = _named(root)["verticalLayout_7"][0]
    out = []
    for item in outer.findall("item"):
        for child in item:
            if child.get("name"):
                out.append(child.get("name"))
    return out


def test_run_sits_above_the_tabs_and_export_below_them():
    """The user's words: "Run should be at the top / Export at the
    bottom". Positions are asserted relative to the tab widget, so a
    later insertion above or below cannot quietly invert them."""
    root = _root()
    order = _outer_items(root)
    assert order.index("horizontalLayout_run") < order.index("log_widget"), \
        "the Run row must come before the tabs"
    assert order.index("gridLayout_3") > order.index("log_widget"), \
        "the shared button row must come after the tabs"
    assert "run_button" in _subtree_names(
        _named(root)["horizontalLayout_run"][0])
    assert "run_button" not in _subtree_names(_named(root)["gridLayout_3"][0])
    assert "export_button" not in _subtree_names(
        _named(root)["horizontalLayout_run"][0])


def test_run_and_export_are_outside_the_tabs():
    """Both must stay reachable from every tab."""
    root = _root()
    for name in ("run_button", "export_button"):
        ancestors = _ancestor_names(root, _named(root)[name][0])
        assert "tab" not in ancestors and "tab_setup" not in ancestors, \
            f"{name} is trapped inside a tab page"


def test_export_is_declared_disabled_and_wired_once():
    """It used to be built in Python and inserted at the top; it is a
    declared widget now, so the initial disabled state lives in the .ui
    and the click goes through set_connections like every sibling."""
    widget = _named(_root())["export_button"][0]
    enabled = widget.find("property[@name='enabled']/bool")
    assert enabled is not None and enabled.text == "false", (
        "Export must start dead — there is nothing to export before a run")
    assert widget.find("property[@name='text']/string").text.endswith("…"), \
        "Export opens a folder chooser; keep the ellipsis convention"
    source = open(DIALOG_FILE, encoding="utf-8").read()
    assert source.count("self.export_button.clicked.connect") == 1
    assert "QtWidgets.QPushButton(\"Export" not in source, (
        "the hand-built Export bar is gone; the .ui owns the button")


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
    "_start_dl_job", "_show_tab", "_open_env_dir",
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
    assert len(names) == 10


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


# --- AST: mutual exclusion --------------------------------------------------

def _setup_action_slots():
    return sorted(name for name in _function_names()
                  if name.startswith("_setup_"))


def _first_statement(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and \
            isinstance(body[0].value, ast.Constant):
        body = body[1:]          # skip the docstring
    return body[0] if body else None


@pytest.mark.parametrize("slot", _setup_action_slots())
def test_every_setup_action_slot_opens_with_the_busy_guard(slot):
    """models_treeWidget.itemDoubleClicked reaches _setup_download_selected
    without going anywhere near a button, and the tree used to stay
    enabled while every button was dead. A double-click during a live
    detection then started a download that can replace, on disk, the
    .onnx the running child has open. The interlock therefore also lives
    at the top of every slot, not only in the enabled states."""
    first = _first_statement(_function_node(slot))
    assert isinstance(first, ast.If), (
        f"{slot} does not start with a guard")
    assert "_busy" in ast.dump(first.test), (
        f"{slot}'s first statement does not test self._busy()")
    body = ast.dump(ast.Module(body=first.body, type_ignores=[]))
    assert "_refuse_while_busy" in body and "Return" in body, (
        f"{slot} tests _busy() but does not refuse and return")


def test_the_model_tree_itself_is_disabled_while_busy():
    """Not only the buttons: the tree carries itemDoubleClicked ->
    download, so a live tree is a signal path around the interlock."""
    module = _module()
    for node in ast.walk(module):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SETUP_INPUTS"
                for t in node.targets):
            inputs = [el.value for el in node.value.elts]
            break
    else:                                          # pragma: no cover
        pytest.fail("SETUP_INPUTS constant not found")
    assert "models_treeWidget" in inputs
    assert set(inputs) <= set(_named(_root())), (
        "SETUP_INPUTS names something the .ui does not declare")
    source = ast.dump(_function_node("_set_busy_ui"))
    assert "SETUP_INPUTS" in source, (
        "_set_busy_ui disables the buttons but leaves the tree live")
    assert "SETUP_INPUTS" in ast.dump(_function_node("_refresh_setup_actions"))


@pytest.mark.parametrize("starter", ["_start_env_job", "_start_dl_job"])
def test_the_starters_refuse_a_job_of_another_kind(starter):
    """The per-pair guard alone let a download start during a detection:
    _start_dl_job only ever looked at _dl_thread."""
    source = ast.dump(_function_node(starter))
    assert "_claim_job" in source, (
        f"{starter} only guards its own pair; a run or the other pair can "
        "still be live")


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
    # ...and every step of the remove path names its own terminal slots
    referenced = set()
    for name in ("_setup_delete_env", "_start_deletion_pricing",
                 "_run_pending_deletion"):
        referenced |= {n.attr for n in ast.walk(_function_node(name))
                       if isinstance(n, ast.Attribute)}
    assert "_on_env_removed" in referenced
    assert "_on_env_ready" not in referenced


def test_shutdown_still_reaps_every_pair():
    node = _function_node("_shutdown_threads")
    source = ast.dump(node)
    for pair in ("_env_thread", "_dl_thread", "_probe_thread"):
        assert pair in source, f"{pair} is never reaped"
    reaps = [c for c in ast.walk(node)
             if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "_reap_thread"]
    assert len(reaps) >= 4, (
        "the inference, env, download and probe pairs must all be reaped; "
        "a missed one qFatals QGIS on close")


def test_the_environment_is_measured_on_a_worker():
    """The probe (three interpreter launches plus two directory walks) is
    what froze the dialog on open; it belongs on a thread of its own."""
    node = _function_node("_start_env_probe")
    referenced = {n.attr for n in ast.walk(node)
                  if isinstance(n, ast.Attribute)}
    for required in ("moveToThread", "quit", "deleteLater", "start"):
        assert required in referenced, \
            f"_start_env_probe never uses {required}"
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    assert "EnvProbeWorker" in names
    # ...and it is read-only, so it must NOT join the interlock: a probe
    # that disabled Run would block a detection for no reason at all.
    assert "_probe_thread" not in ast.dump(_function_node("_busy_kind"))


def test_the_first_paint_is_seeded_not_measured():
    node = _function_node("_env_snapshot")
    called = {c.func.attr for c in ast.walk(node)
              if isinstance(c, ast.Call) and hasattr(c.func, "attr")}
    assert "env_seed" in called, (
        "_env_snapshot measures on the GUI thread again")
    assert "env_info" not in called


def test_the_download_on_run_rerun_is_gated():
    """Downloading a model from the Setup tab must not start an
    inference run."""
    node = _function_node("_on_model_downloaded")
    guarded = [n for n in ast.walk(node)
               if isinstance(n, ast.If)
               and "_dl_then_run" in ast.dump(n.test)]
    assert guarded, "_on_model_downloaded re-runs unconditionally"
    arms = [n for n in ast.walk(guarded[0])
            if isinstance(n, ast.Attribute)
            and n.attr == "_rerun_after_download"]
    assert arms, "the gated branch must arm the rerun flag"


def _singleshot_targets(node):
    """The slots handed to QTimer.singleShot inside ``node``."""
    out = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if getattr(call.func, "attr", None) != "singleShot":
            continue
        for arg in call.args:
            name = getattr(arg, "attr", None)
            if name:
                out.add(name)
    return out


def test_the_rerun_is_fired_from_the_thread_cleared_slot():
    """worker.done is delivered BEFORE thread.finished nulls _dl_thread.
    A zero-timer posted from the done slot therefore reaches run_process
    while _busy_kind() is still 'download': the pre-flight computes
    TXT_BLOCK_BUSY, bounces to the Setup tab and — _dl_then_run having
    already been cleared — never retries. So the rerun may only be
    scheduled from _clear_dl_thread, and only after the pair is free."""
    done = _function_node("_on_model_downloaded")
    assert "run_process" not in _singleshot_targets(done), (
        "_on_model_downloaded must arm the rerun, not post it: the "
        "download thread is still live when done fires")

    clear = _function_node("_clear_dl_thread")
    assert "run_process" in _singleshot_targets(clear)
    # ...and strictly after _dl_thread is dropped, or run_process still
    # sees a busy dialog.
    body = clear.body
    cleared_at = next(
        i for i, stmt in enumerate(body)
        if isinstance(stmt, ast.Assign)
        and any(getattr(t, "attr", None) == "_dl_thread"
                for t in stmt.targets))
    fired_at = next(i for i, stmt in enumerate(body)
                    if "singleShot" in ast.dump(stmt))
    assert fired_at > cleared_at


def test_the_deletion_confirmation_is_also_fired_after_the_pair_is_free():
    """Same race, same cure: the pricing job's done slot only arms the
    confirmation; _clear_env_thread runs it once the pair is free."""
    priced = _function_node("_on_deletion_priced")
    assert not _singleshot_targets(priced)
    assert "_pending_deletion" in ast.dump(priced)
    clear = _function_node("_clear_env_thread")
    assert "_run_pending_deletion" in _singleshot_targets(clear)


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


# --- window size + scrolling (BUGS.md "The GUI window is too high") ---------
#
# The report: "Currently I can[not] press run because it is just outside my
# screen." The dialog declared a fixed 640x750 with no QScrollArea anywhere,
# so on a 768/800 px laptop panel the bottom of the window — Run included —
# sat below the desktop and the plugin could not be used at all. Three
# things keep that from coming back: a default height a laptop can show, a
# scroll area around each form page, and a hard rule about which controls
# may never end up inside one.

# A 800 px panel minus the macOS menu bar and dock (~100 px) minus the
# window frame. The whole window has to live inside this.
LAPTOP_SAFE_HEIGHT = 700


def _form():
    """The top-level <widget> of the form — the dialog itself."""
    return _root().find("widget")


def _declared_size(root, prop):
    node = root.find("property[@name='%s']" % prop)
    if node is None:
        return None
    size = node.find("rect")
    if size is None:
        size = node.find("size")
    if size is None:
        return None
    return (int(size.find("width").text), int(size.find("height").text))


def test_the_dialog_opens_short_enough_for_a_laptop():
    width, height = _declared_size(_form(), "geometry")
    assert height <= LAPTOP_SAFE_HEIGHT, (
        f"the dialog declares {width}x{height}; anything taller than "
        f"{LAPTOP_SAFE_HEIGHT} px puts the button row off a laptop screen")


def test_the_dialog_may_be_shrunk_further():
    """A minimum taller than the default would make the default a floor
    and hand the bug straight back."""
    minimum = _declared_size(_form(), "minimumSize")
    assert minimum is not None, (
        "no minimumSize: the dialog's minimum is then whatever its "
        "content demands, which is exactly what nobody can control")
    _, default_height = _declared_size(_form(), "geometry")
    assert minimum[1] < default_height
    assert minimum[1] <= 400 and minimum[0] <= 560, (
        f"minimumSize {minimum} is too large to survive a short screen")


def test_the_dialog_declares_no_maximum_size():
    """It still has to be usable on a big monitor."""
    assert _form().find("property[@name='maximumSize']") is None


def _scroll_areas(element):
    return [w for w in element.iter("widget")
            if w.get("class") == "QScrollArea"]


@pytest.mark.parametrize("page_name", ["tab_setup", "tab"])
def test_each_form_page_scrolls(page_name):
    """The form pages are the tall ones. Their content goes in a scroll
    area so a short window clips nothing."""
    page = [p for p in _tab_pages() if p.get("name") == page_name][0]
    areas = _scroll_areas(page)
    assert len(areas) == 1, (
        f"{page_name} must hold exactly one QScrollArea, found "
        f"{[a.get('name') for a in areas]}")
    area = areas[0]
    resizable = area.find("property[@name='widgetResizable']/bool")
    assert resizable is not None and resizable.text == "true", (
        "without widgetResizable the content keeps its size hint and the "
        "scroll area is decoration")
    # and it really WRAPS the page: the page's own controls are inside it
    inner = _subtree_names(area)
    outer = _subtree_names(page) - inner - {area.get("name")}
    assert not [n for n in outer
                if n.endswith(("_button", "_comboBox", "_lineEdit",
                               "_checkBox", "_spinBox"))], (
        f"{page_name} has controls outside its scroll area: "
        f"{sorted(outer)}")


def test_the_log_page_scrolls_on_its_own():
    """A QPlainTextEdit already scrolls; nesting it in a scroll area
    would give it two scrollbars and an unbounded height."""
    page = [p for p in _tab_pages() if p.get("name") == "tab_2"][0]
    classes = {w.get("class") for w in page.iter("widget")}
    assert "QScrollArea" not in classes
    assert "QPlainTextEdit" in classes


# The controls the report is actually about. If any of these can scroll,
# the user can once again fail to reach Run.
NEVER_SCROLLABLE = ("run_button", "cancel_button", "close_button",
                    "help_button", "export_button", "progress_bar")


@pytest.mark.parametrize("name", NEVER_SCROLLABLE)
def test_the_always_reachable_controls_never_scroll(name):
    """An ancestry walk over the parsed tree, not a grep: the whole
    failure mode is a widget being *nested* somewhere it scrolls away."""
    root = _root()
    parents = {child: parent for parent in root.iter() for child in parent}
    matches = _named(root)[name]
    assert len(matches) == 1, f"{name} is declared {len(matches)} times"
    node, chain = parents.get(matches[0]), []
    while node is not None:
        chain.append((node.get("class"), node.get("name")))
        node = parents.get(node)
    scrolled = [c for c in chain if c[0] == "QScrollArea"]
    assert not scrolled, (
        f"{name} sits inside {scrolled}; it must stay pinned outside "
        "every scroll area so it is reachable at any window height")
    assert ("QTabWidget", "log_widget") not in chain, (
        f"{name} is trapped inside the tabs")


def test_the_dialog_clamps_itself_to_the_screen_at_construction():
    """The declared default is still too tall for a netbook. The runtime
    clamp is the backstop — and it must be guarded, because the test
    suite and any headless QGIS have no QScreen at all."""
    init = _function_node("__init__")
    called = {getattr(c.func, "attr", None) for c in ast.walk(init)
              if isinstance(c, ast.Call)}
    assert "_fit_to_available_screen" in called

    fit = _function_node("_fit_to_available_screen")
    names = {getattr(c.func, "attr", None) for c in ast.walk(fit)
             if isinstance(c, ast.Call)}
    assert "availableGeometry" in names, "it must consult the real screen"
    assert "resize" in names
    assert any(isinstance(n, ast.Try) for n in ast.walk(fit)), (
        "an absent QScreen must not break the constructor")


def test_only_the_constructor_resizes_the_dialog():
    """A resize from a slot would undo whatever the user dragged the
    window to."""
    offenders = []
    for func in ast.walk(_module()):
        if not isinstance(func, ast.FunctionDef):
            continue
        if func.name == "_fit_to_available_screen":
            continue
        for call in ast.walk(func):
            if not isinstance(call, ast.Call):
                continue
            if getattr(call.func, "attr", None) != "resize":
                continue
            if getattr(getattr(call.func, "value", None), "id", None) == \
                    "self":
                offenders.append((func.name, call.lineno))
    assert not offenders, (
        f"the dialog resizes itself outside construction: {offenders}")


# --- AST: the precision selector -------------------------------------------

def _variant_items():
    """The (label, value) pairs of the module-level VARIANT_ITEMS."""
    for node in ast.walk(_module()):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "VARIANT_ITEMS"
                   for t in node.targets):
            continue
        return [tuple(c.value for c in elt.elts)
                for elt in node.value.elts]
    pytest.fail("VARIANT_ITEMS not found in winmol_analyzer_dialog.py")


def test_the_variant_selector_opens_on_the_machines_default():
    """Item 0 is what an untouched dialog runs with. It must be
    'default' — the registry's own device answer (int8 on a CPU-only
    box) — not 'auto', whose lossless-only gate refuses the shipped int8
    default and lands a CPU-only machine on the 124 MB fp32 model."""
    items = _variant_items()
    assert items[0][1] == "default"
    values = [value for _label, value in items]
    assert values == ["default", "auto", "fp32", "int8", "fp16"]
    # every escape hatch stays one click away
    assert "fp32" in values and "int8" in values


def test_the_variant_preset_lives_where_the_combo_exists():
    """The construction-order defect, pinned.

    populate_model_combo_box() runs BEFORE _add_custom_controls()
    creates variant_comboBox, so a preset written into the former is
    dead code on every single launch — which is exactly how a CPU-only
    machine kept opening on 'auto'. The preset therefore belongs to
    _add_custom_controls (or anything it calls), and nowhere else.
    """
    src = open(DIALOG_FILE, encoding="utf-8").read()
    module = ast.parse(src)

    init = _function_node("__init__")
    order = [call.func.attr for call in ast.walk(init)
             if isinstance(call, ast.Call)
             and getattr(call.func, "attr", None) in
             ("populate_model_combo_box", "_add_custom_controls")]
    assert order == ["populate_model_combo_box", "_add_custom_controls"], (
        "construction order changed; the preset assumption below must be "
        "re-derived")

    callers = set()
    for func in ast.walk(module):
        if not isinstance(func, ast.FunctionDef):
            continue
        for call in ast.walk(func):
            if isinstance(call, ast.Call) and getattr(
                    call.func, "attr", None) == "_default_variant_value":
                callers.add(func.name)
    assert "populate_model_combo_box" not in callers, (
        "the variant preset is back in populate_model_combo_box, where "
        "variant_comboBox does not exist yet — it silently does nothing")
    assert callers, "nothing presets the variant selector any more"

    adder = _function_node("_add_custom_controls")
    called = {c.func.attr for c in ast.walk(adder)
              if isinstance(c, ast.Call) and hasattr(c.func, "attr")}
    assert "_preset_default_variant" in called


def test_the_variant_tooltip_does_not_oversell_int8():
    """int8 for the Keras-derived families is domain-calibrated, not
    certified lossless. The UI has to say so at the point of choice."""
    src = open(DIALOG_FILE, encoding="utf-8").read()
    start = src.index("VARIANT_TOOLTIP")
    tip = src[start:src.index("\n)\n", start)].lower()
    assert "not certified lossless" in tip
    assert "domain-calibrated" in tip
    assert "reference (fp32)" in tip


def test_the_variant_labels_carry_the_download_size():
    """31 MB against 124 MB is the reason to prefer int8 on a CPU box;
    it must be visible before the download, not after."""
    node = _function_node("_variant_item_text")
    src = ast.unparse(node)
    assert "size_mb" in src and "MB" in src
    refresher = ast.unparse(_function_node("_refresh_variant_controls"))
    assert "_variant_item_text" in refresher and "setItemText" in refresher


# --- AST: the environment QGIS leaves behind --------------------------------

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
    # ...and the reason is written down where the next reader will look.
    doc = ast.get_docstring(unload) or ""
    assert "uninstall" in doc.lower()


def test_the_setup_tab_names_the_folder_uninstall_leaves_behind():
    """The user's report: '<profile>/winmol stays untouched after
    deinstalling'. It must at least be visible and openable."""
    assert "_open_env_dir" in _function_names()
    node = _function_node("_open_env_dir")
    src = ast.unparse(node)
    assert "managed_root" in src and "openUrl" in src
    # and nothing destructive hides behind an "open folder" button
    assert "rmtree" not in src and "remove_environment" not in src

    refresh = ast.unparse(_function_node("_refresh_setup_state"))
    assert "env_location_text" in refresh
    scope = ast.unparse(_function_node("_ask_deletion_scope"))
    assert "env_location_text" in scope, (
        "the deletion dialog must say what uninstalling does not remove")


def test_deleting_the_environment_still_needs_an_explicit_confirmation():
    """Making the leftover removable must not make it removable by
    accident: the scope dialog and the itemised confirmation both stay."""
    delete = ast.unparse(_function_node("_setup_delete_env"))
    assert "_ask_deletion_scope" in delete
    confirm = ast.unparse(_function_node("_confirm_deletion"))
    assert "Cancel" in confirm and "cannot be undone" in confirm
