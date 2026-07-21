"""Map the plugin's three output checkboxes onto one pipeline run.

The dialog offers three "products": the semantic stem map (raster), the
detected wind-thrown trees (line layer) and the measuring nodes (point
layer). They are *not* three independent runs: a single ``winmol_run.py``
invocation with ``process_type == "Nodes"`` writes the stem-map raster and
a GeoPackage holding the ``stems``, ``vectors`` and ``nodes`` layers.

This module is deliberately Qt-free so the selection logic can be unit
tested without a live QGIS.
"""

STEMS_LAYER = "stems"
VECTORS_LAYER = "vectors"
NODES_LAYER = "nodes"


def process_type_for(stem: bool, trees: bool, nodes: bool) -> str:
    """Return the ``winmol_run.py`` process type for a checkbox selection.

    The three checkboxes behave as a ladder: each product implies the ones
    below it, so the highest selected product wins. ``stem`` only matters
    as the (implied) baseline -- with nothing selected we still fall back
    to the cheapest mode.
    """
    if nodes:
        return "Nodes"
    if trees:
        return "Trees"
    _ = stem
    return "Stems"


def gpkg_layers_for(trees: bool, nodes: bool) -> list:
    """Return the GeoPackage layers to load, in load order, without dupes.

    ``Nodes`` runs produce all three vector layers; ``Trees`` runs produce
    only ``stems``; a stem-map-only run produces no GeoPackage at all.
    """
    if nodes:
        return [STEMS_LAYER, VECTORS_LAYER, NODES_LAYER]
    if trees:
        return [STEMS_LAYER]
    return []
