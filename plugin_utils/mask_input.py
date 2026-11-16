"""Plugin side of the mask input: build and check the ``winmol_from_labels.py`` call.

With "Existing stem mask" chosen, the dialog skips prediction and runs the downstream stages (vectorising, or
measuring along given centre lines) on a stem mask the user already has: drawn by hand, or predicted by any model.
Qt-free for unit testing; the dialog only collects the paths and starts the command.
"""
import os

FROM_LABELS = "winmol_from_labels.py"


def split_layer_source(source):
    """(path, layer name or None) from a QGIS layer source such as ``/d/a.gpkg|layername=stems``."""
    path, _, rest = source.partition("|")
    layer = None
    for part in rest.split("|") if rest else []:
        key, _, value = part.partition("=")
        if key == "layername" and value:
            layer = value
    return path, layer


def mask_command(python_exe, plugin_dir, mask, out_prefix, mask_layer=None, lines=None, lines_layer=None,
                 id_field="ID"):
    """The child command; ``out_prefix`` gets ``.gpkg`` (stems, vectors, nodes) and ``_summary.json``."""
    cmd = [python_exe, "-u", os.path.join(plugin_dir, FROM_LABELS), "--mask", mask, "--out", out_prefix]
    if mask_layer:
        cmd += ["--mask-layer", mask_layer]
    if lines:
        cmd += ["--lines", lines]
        if lines_layer:
            cmd += ["--lines-layer", lines_layer]
        cmd += ["--id-field", id_field]
    return cmd


def validate_mask_input(mask, lines=None):
    """None when the inputs exist, else the message to show."""
    if not mask:
        return "No stem mask selected. Pick a raster or polygon layer or file."
    if not os.path.exists(mask):
        return f"Stem mask was not found:\n{mask}"
    if lines and not os.path.exists(lines):
        return f"Centre-line file was not found:\n{lines}"
    return None
