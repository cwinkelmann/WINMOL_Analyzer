"""The three plugin output products are enabled by default.

The QGIS dialog cannot be instantiated here (no ``qgis``/``PyQt5`` in the
test environment), so the default state is pinned statically:

* the ``.ui`` file is the source of truth for the checkbox defaults, and
* ``set_connections`` must not undo those defaults with hardcoded literals.

The selection logic itself lives in :mod:`plugin_utils.output_selection`,
which is Qt-free and therefore directly unit-testable.
"""

import ast
import os
import xml.etree.ElementTree as ET

import pytest

from plugin_utils.output_selection import (
    gpkg_layers_for,
    process_type_for,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_FILE = os.path.join(REPO, "winmol_analyzer_dialog_base.ui")
DIALOG_FILE = os.path.join(REPO, "winmol_analyzer_dialog.py")

OUTPUT_CHECKBOXES = (
    "output_checkBox_stem",
    "output_checkBox_trees",
    "output_checkBox_nodes",
)


def _checkbox_widgets():
    root = ET.parse(UI_FILE).getroot()
    found = {}
    for widget in root.iter("widget"):
        if widget.get("class") != "QCheckBox":
            continue
        name = widget.get("name")
        if name in OUTPUT_CHECKBOXES:
            found[name] = widget
    return found


def test_all_three_output_checkboxes_default_checked():
    widgets = _checkbox_widgets()
    missing = sorted(set(OUTPUT_CHECKBOXES) - set(widgets))
    assert not missing, f"output checkboxes missing from the .ui: {missing}"

    for name in OUTPUT_CHECKBOXES:
        checked = widgets[name].find("property[@name='checked']/bool")
        assert checked is not None, f"{name} has no 'checked' property"
        assert checked.text == "true", f"{name} does not default to checked"


def _set_connections_node():
    tree = ast.parse(open(DIALOG_FILE, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "set_connections":
            return node
    pytest.fail("set_connections not found in winmol_analyzer_dialog.py")


HANDLERS = (
    "checkbox_changed_stem",
    "checkbox_changed_trees",
    "checkbox_changed_nodes",
)


def _called_name(call):
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", None)


def test_priming_does_not_hardcode_unchecked():
    """The handlers must not be primed with a literal 'unchecked' state.

    ``checkbox_changed_trees(1)`` used to force the nodes checkbox off and
    disabled at construction time, silently overriding the .ui defaults.
    The priming state must be derived from the widgets instead.
    """
    node = _set_connections_node()

    # A rename of the handlers must fail loudly rather than pass vacuously:
    # all three must still be referenced (called or bound) here.
    referenced = {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }
    missing = sorted(set(HANDLERS) - referenced)
    assert not missing, (
        f"handlers no longer wired in set_connections: {missing}")

    # '_handler' is the loop variable the priming calls go through.
    priming_targets = set(HANDLERS) | {"_handler"}
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if _called_name(call) not in priming_targets:
            continue
        for arg in call.args:
            assert not isinstance(arg, ast.Constant), (
                f"{_called_name(call)} is primed with the literal "
                f"{arg.value!r}; derive the state from the widget instead")


def _function_node(name):
    tree = ast.parse(open(DIALOG_FILE, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{name} not found in winmol_analyzer_dialog.py")


def test_layer_loading_goes_through_one_computed_list():
    """No second, independent branch may re-add the 'stems' layer.

    With trees+nodes both checked (the default) two non-exclusive ``if``
    branches used to add 'stems' to the QGIS project twice.
    """
    node = _function_node("load_layers_to_session")
    calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
    assert any(_called_name(c) == "gpkg_layers_for" for c in calls), (
        "load_layers_to_session must derive its layer list from "
        "gpkg_layers_for()")
    loads = [c for c in calls if _called_name(c) == "load_gpkg_layers"]
    assert len(loads) == 1, (
        f"expected exactly one load_gpkg_layers() call, found {len(loads)}")


@pytest.mark.parametrize(
    "stem,trees,nodes,expected",
    [
        (True, True, True, "Nodes"),
        (True, True, False, "Trees"),
        (True, False, False, "Stems"),
        (False, False, False, "Stems"),
        (True, False, True, "Nodes"),
    ],
)
def test_process_type_for(stem, trees, nodes, expected):
    assert process_type_for(stem, trees, nodes) == expected


@pytest.mark.parametrize(
    "trees,nodes,expected",
    [
        # The exact list, so a layer added twice (which used to put
        # 'stems' into the QGIS project twice) fails here.
        (True, True, ["stems", "vectors", "nodes"]),
        (False, True, ["stems", "vectors", "nodes"]),
        (True, False, ["stems"]),
        (False, False, []),
    ],
)
def test_gpkg_layers_for(trees, nodes, expected):
    assert gpkg_layers_for(trees, nodes) == expected
