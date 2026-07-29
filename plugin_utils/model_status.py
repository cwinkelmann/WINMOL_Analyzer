"""What the model registry looks like ON THIS MACHINE, right now.

:func:`scan` turns ``config.json`` plus a models directory into the
flat list of :class:`ModelRow` the Setup tab's tree renders. By default
it is stat()-only — it never hashes a file and never touches the
network — so it is safe to call every time the Setup tab becomes
visible; ``verify=True`` (worker-thread territory) additionally checks
pinned checksums.

Qt-free and QGIS-free, like everything else the Setup tab decides with.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from . import model_registry


@dataclass
class ModelRow:
    """One registry model, as the Setup tree needs to render it.

    ``verified`` is tri-state: True/False after a checksum pass, None
    when the file is absent, unpinned, or simply not checked yet.
    ``is_default`` flags the single entry ``Registry.default_entry``
    resolves for this device — the file a default run would load.
    """

    entry_id: str
    label: str
    family: str
    precision: str
    file: str
    path: str
    size_mb: Optional[float]
    bytes_on_disk: int
    installed: bool
    pinned: bool
    verified: Optional[bool]
    is_default: bool


def scan(config_path, models_dir, device=None,
         verify=False) -> List[ModelRow]:
    """One :class:`ModelRow` per registry entry.

    ``device`` (``cpu`` | ``gpu`` | ``coreml``, default: detect) picks
    which entry carries the ``is_default`` flag — deliberately via the
    SAME ``default_entry`` call a run makes, so the highlighted row is
    the file that will actually be loaded, not a second opinion.
    """
    registry = model_registry.load_registry(config_path)
    try:
        default_id = registry.default_entry(device or "auto").id
    except KeyError:
        default_id = None

    rows = []
    for entry in registry.entries.values():
        path = model_registry.local_path(entry, models_dir)
        try:
            on_disk = os.path.getsize(path)
        except OSError:
            on_disk = 0
        installed = on_disk > 0
        verified = None
        if verify and installed and entry.sha256:
            verified = model_registry.verify_file(path, entry.sha256)
        rows.append(ModelRow(
            entry_id=entry.id,
            label=entry.label,
            family=entry.family,
            precision=entry.precision,
            file=entry.file,
            path=path,
            size_mb=entry.size_mb,
            bytes_on_disk=on_disk,
            installed=installed,
            pinned=bool(entry.sha256),
            verified=verified,
            is_default=entry.id == default_id,
        ))
    return rows
