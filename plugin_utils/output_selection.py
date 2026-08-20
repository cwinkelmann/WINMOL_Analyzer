"""Map the plugin's three output checkboxes onto one pipeline run.

The dialog offers three "products" (stem map, trees, nodes), but they are
not independent runs -- one ``winmol_run.py`` invocation with the highest
selected ``process_type`` produces all of them. Qt-free for unit testing.
"""

STEMS_LAYER = "stems"
VECTORS_LAYER = "vectors"
NODES_LAYER = "nodes"


def process_type_for(stem: bool, trees: bool, nodes: bool) -> str:
    """Return the ``winmol_run.py`` process type for a checkbox selection.

    The three checkboxes are a ladder: each product implies the ones below
    it, so the highest selected product wins; nothing selected falls back
    to the cheapest mode.
    """
    if nodes:
        return "Nodes"
    if trees:
        return "Trees"
    return "Stems"


def gpkg_layers_for(trees: bool, nodes: bool) -> list:
    """Return the GeoPackage layers to load, in load order, without dupes.

    ``Nodes`` runs produce all three layers, ``Trees`` only ``stems``, and
    a stem-map-only run produces no GeoPackage at all.
    """
    if nodes:
        return [STEMS_LAYER, VECTORS_LAYER, NODES_LAYER]
    if trees:
        return [STEMS_LAYER]
    return []
