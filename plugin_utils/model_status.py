"""What the model registry looks like ON THIS MACHINE, right now.

:func:`scan` turns a parsed ``Registry`` plus a models directory into the
flat list of :class:`~plugin_utils.setup_state.ModelRow` the Setup tab's
tree renders. It is stat()-only — it never hashes a file and never
touches the network — so it is safe to call on the GUI thread every time
the Setup tab becomes visible.

Qt-free and QGIS-free, like everything else the Setup tab decides with.
"""

import os

from . import model_registry
from .setup_state import ModelRow


def _family_label(registry, entry):
    family = registry.families.get(entry.family)
    if family is not None:
        return family.id, family.label
    # Hand-edited registries may hold family-less entries; keep them
    # visible under their own name rather than silently dropping them.
    return entry.family or entry.id, entry.family or entry.label


def _recommended_id(registry, family_id, device, variant="auto"):
    """The entry the registry would actually load for ``family_id`` on
    this device, or None when it cannot be resolved."""
    if not family_id:
        return None
    try:
        return registry.resolve(family_id, device=device,
                                variant=variant).id
    except (KeyError, ValueError):
        return None


def scan(registry, models_dir, device="auto", family_id=None,
         variant="auto") -> list:
    """One :class:`ModelRow` per selectable registry entry.

    ``family_id`` and ``variant`` are the family and precision currently
    chosen on the detection tab; the single entry that
    ``registry.resolve(family_id, device=device, variant=variant)``
    returns is flagged ``recommended``. That is deliberately the SAME
    call the run itself makes, so the highlighted row is the file that
    will actually be loaded — not a second opinion about it.
    ``family_id`` defaults to the registry's own ``gui_default`` family
    so a caller with no selection yet still gets a sensible highlight.

    Reserved ids (the GUI's "Custom" escape hatch) are excluded. Hidden
    entries (the TensorFlow-only Zenodo originals) are INCLUDED but
    flagged, so the tree can suppress them without this module deciding
    presentation.
    """
    if device == "auto":
        device = model_registry.detect_device()
    if family_id is None:
        family_id = _default_family_id(registry)
    recommended = _recommended_id(registry, family_id, device, variant)

    rows = []
    for entry in registry.entries.values():
        if entry.id in model_registry.RESERVED_IDS:
            continue
        path = model_registry.local_path(entry, models_dir)
        state = model_registry.installed_state(entry, models_dir)
        try:
            on_disk = os.path.getsize(path)
        except OSError:
            on_disk = 0
        fam_id, fam_label = _family_label(registry, entry)
        rows.append(ModelRow(
            entry_id=entry.id,
            family_id=fam_id,
            family_label=fam_label,
            label=entry.label,
            precision=entry.precision,
            backend=entry.backend,
            file=entry.file,
            path=path,
            size_expected_mb=entry.size_mb,
            bytes_on_disk=on_disk,
            present=state != model_registry.STATE_MISSING,
            pinned=bool(entry.sha256 or entry.md5),
            verified=state == model_registry.STATE_VERIFIED,
            hidden=bool(entry.hidden),
            recommended=entry.id == recommended,
            state=state,
            description=entry.description,
            f1=entry.f1,
        ))
    return rows


def _default_family_id(registry):
    mid = registry.gui_default
    entry = registry.entries.get(mid) if mid else None
    if entry is None:
        return None
    family = registry.families.get(entry.family)
    return family.id if family is not None else None


def group_by_family(rows, show_all_variants=False, include_hidden=False):
    """``[(family_id, family_label, [rows])]`` in registry order.

    Without ``show_all_variants`` a family shows its default entry plus
    anything already on disk plus the recommended row — enough to act on,
    without a 22-row wall of precisions on first open.
    """
    order = []
    grouped = {}
    for row in rows:
        if row.hidden and not include_hidden:
            continue
        if row.family_id not in grouped:
            grouped[row.family_id] = []
            order.append((row.family_id, row.family_label))
        grouped[row.family_id].append(row)

    out = []
    for fam_id, fam_label in order:
        members = grouped[fam_id]
        if show_all_variants:
            shown = members
        else:
            shown = [r for r in members
                     if r.present or r.recommended or r.precision == "fp32"]
            shown = shown or members[:1]
        out.append((fam_id, fam_label, shown))
    return out


def family_summary(rows) -> str:
    """'1 of 3 on disk, 92 MB' for a family's rows."""
    from .setup_state import human_bytes
    present = [row for row in rows if row.present]
    size = human_bytes(sum(row.bytes_on_disk for row in present))
    return f"{len(present)} of {len(rows)} on disk, {size}"
