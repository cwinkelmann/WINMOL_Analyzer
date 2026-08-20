"""What the model registry looks like ON THIS MACHINE, right now.

:func:`scan` turns ``config.json`` plus a models directory into the
flat list of :class:`ModelRow` the Setup tab's tree renders. It is
stat()-only — it never hashes a file and never touches the network —
so it is safe to call every time the Setup tab becomes visible.
(Checksum verification is worker-thread territory and goes through
tasks_threads.ModelMaintenanceWorker, not this scan.)

Qt-free and QGIS-free, like everything else the Setup tab decides with.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from . import model_registry


@dataclass
class ModelRow:
    """One registry model, as the Setup tree needs to render it.

    ``is_default`` flags the single entry the registry resolves for
    the CURRENT selection (family + variant + device) — the file a run
    with that selection would load. ``path`` is where the file actually
    lives when installed (managed dir, or a read-fallback dir for
    rr6-era installs); for a missing file it is the managed WRITE
    target a download would create.
    """

    entry_id: str
    label: str
    precision: str
    file: str
    path: str
    size_mb: Optional[float]
    bytes_on_disk: int
    installed: bool
    pinned: bool
    is_default: bool
    family_id: str = ""
    family_label: str = ""
    hidden: bool = False


def _family_of(registry, entry):
    """(family_id, family_label) for ``entry``. Family-less entries
    (legacy v1 registries, the PyTorch retrain trio) stay visible
    under their own name rather than being silently dropped."""
    family = registry.families.get(entry.family)
    if family is not None:
        return family.id, family.label
    return entry.family or entry.id, entry.family or entry.label


def _default_id(registry, device, family_id, variant):
    """The entry a run with the current selection would load, or None.

    With a ``family_id`` this is deliberately the SAME
    ``registry.resolve(family, device=…, variant=…)`` call the run
    itself makes, so the highlighted row is the file that will
    actually be loaded — not a second opinion about it. Without one
    (nothing selected yet, Custom, legacy registry) it falls back to
    the machine-wide ``default_entry``.
    """
    if family_id:
        try:
            return registry.resolve(
                family_id, device=device, variant=variant).id
        except (KeyError, ValueError):
            return None
    try:
        return registry.default_entry(device).id
    except KeyError:
        return None


def _locate(entry, models_dir, fallback_dirs):
    """(path, bytes) — the managed location when the file is there,
    else the first fallback dir that has it (rr6-era installs kept
    models next to the plugin), else the managed WRITE target with 0
    bytes. stat()-only."""
    candidates = [models_dir] + [d for d in fallback_dirs if d]
    for directory in candidates:
        path = model_registry.local_path(entry, directory)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        if size > 0:
            return path, size
    return model_registry.local_path(entry, models_dir), 0


def scan(config_path, models_dir, device=None, family_id=None,
         variant=None, fallback_dirs=()) -> List[ModelRow]:
    """One :class:`ModelRow` per registry entry.

    ``family_id`` and ``variant`` are the family and precision
    currently chosen on the Detection tab; the entry their resolution
    lands on carries ``is_default``. ``device`` (``cpu`` | ``gpu`` |
    ``coreml``, default: detect) fills the device rule where the
    variant delegates to it. ``fallback_dirs`` are extra READ-only
    locations (legacy ``<plugin_dir>/models``) checked when the
    managed dir lacks the file.
    """
    registry = model_registry.load_registry(config_path)
    default_id = _default_id(
        registry, device or "auto", family_id, variant)

    rows = []
    for entry in registry.entries.values():
        path, on_disk = _locate(entry, models_dir, fallback_dirs)
        fam_id, fam_label = _family_of(registry, entry)
        rows.append(ModelRow(
            entry_id=entry.id,
            label=entry.label,
            precision=entry.precision,
            file=entry.file,
            path=path,
            size_mb=entry.size_mb,
            bytes_on_disk=on_disk,
            installed=on_disk > 0,
            pinned=bool(entry.sha256),
            is_default=entry.id == default_id,
            family_id=fam_id,
            family_label=fam_label,
            hidden=bool(entry.hidden),
        ))
    return rows


def group_by_family(rows, include_hidden=False):
    """``[(family_id, family_label, [rows])]`` in registry order, for
    the tree's family parents. Hidden entries (non-runnable formats)
    are dropped unless asked for; which VISIBLE rows to show is the
    caller's filter — this only groups what it is given."""
    order = []
    grouped = {}
    for row in rows:
        if row.hidden and not include_hidden:
            continue
        if row.family_id not in grouped:
            grouped[row.family_id] = []
            order.append((row.family_id, row.family_label))
        grouped[row.family_id].append(row)
    return [(fam_id, fam_label, grouped[fam_id])
            for fam_id, fam_label in order]


def family_summary(rows) -> str:
    """'1 of 3 on disk, 92 MB' for a family's rows."""
    from .setup_state import human_bytes
    present = [row for row in rows if row.installed]
    size = human_bytes(sum(row.bytes_on_disk for row in present))
    return f"{len(present)} of {len(rows)} on disk, {size}"
