# Known bugs and deployment findings — reimpl lineage

## Synced/roaming profile folders swallow the managed environment (university machines)

Reported 2026-07-29. On university machines the profile area (C:) is a
mounted/synced folder. The plugin installs its managed environment under
`managed_root()` = `<QGIS profile>/winmol` — on Windows that is
`%APPDATA%\Roaming\QGIS\QGIS3\profiles\default\winmol`, which roaming-profile
and sync tooling replicates. The ~1–2 GB venv (thousands of small files) then
syncs continuously.

The location was chosen deliberately (outside the plugin dir so QGIS uninstall
never recursive-deletes the venv), but "outside the plugin dir" does not have
to mean "inside the roaming profile".

**Fix direction (planned, not implemented):**
- Move `managed_root` to a local non-synced base: `%LOCALAPPDATA%` (Windows),
  `~/Library/Application Support` (macOS), `~/.local/share` (Linux).
- Site-admin override (`WINMOL_MANAGED_ROOT` env or a QSettings key).
- Migration: keep honoring an existing env at the old location; new installs
  go to the new one. Optionally detect known sync roots (OneDrive, Nextcloud)
  and warn in the Setup tab.
- Invariants to keep: outside the plugin dir (uninstall safety), stable across
  plugin upgrades, one env shared by all profiles is acceptable.
