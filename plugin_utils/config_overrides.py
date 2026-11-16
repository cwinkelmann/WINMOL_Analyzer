"""Building ``$WINMOL_CONFIG_OVERRIDES_JSON`` for the child process.

The dialog and the child run in different interpreters; this env var is
the only channel that carries a config value across the five positional
run args. A user may already have set it in their own shell, so values
are merged, never clobbered: the dialog's own keys win, everything else
survives. Pure stdlib, no Qt and no QGIS imports.
"""

import json

#: The variable winmol_run.py reads.
ENV_VAR = "WINMOL_CONFIG_OVERRIDES_JSON"


def merge(updates, existing=None):
    """Return the JSON object string for :data:`ENV_VAR`.

    Anything unparsable in ``existing`` is discarded rather than
    propagated -- the child exits with status 2 on invalid JSON.
    """
    base = {}
    if existing:
        try:
            parsed = json.loads(existing)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            base = dict(parsed)
    base.update(updates or {})
    return json.dumps(base, sort_keys=True)


def set_default(existing, key, value) -> str:
    """Return the :data:`ENV_VAR` JSON with ``key`` set only if absent.

    An explicit user-set ``key`` always wins. Deliberately different
    from :func:`merge` on bad input: unparsable ``existing`` is
    returned UNCHANGED (not discarded), so the child reports the real
    JSON error itself.
    """
    raw = (existing or "").strip()
    overrides = {}
    if raw:
        try:
            overrides = json.loads(raw)
        except Exception:
            return existing
        if not isinstance(overrides, dict):
            return existing
    if key not in overrides:
        overrides[key] = value
    return json.dumps(overrides)


#: Config keys behind the dialog's vectorisation spin boxes.
VECTOR_KEYS = (
    "min_length",
    "max_distance",
    "tolerance_angle",
    "max_tree_height",
)


def gui_override_env(values, batch=None, existing=None):
    """``{ENV_VAR: ...}`` carrying the dialog's config values, or ``{}``.

    ``values`` go in as given — what the dialog shows is what the child
    runs with. ``batch`` of 0/None means "Auto": no batch key is
    injected. Keys the user set themselves in ``existing`` survive
    unless the dialog sets them too.
    """
    updates = dict(values or {})
    try:
        if int(batch) >= 1:
            updates["prediction_batch_override"] = int(batch)
    except (TypeError, ValueError):
        pass
    if not updates:
        return {}
    return {ENV_VAR: merge(updates, existing)}


def numeric_overrides(existing, keys):
    """The numeric values among ``keys`` in an existing env JSON.

    Tolerant on purpose: it seeds the dialog's spin boxes at startup,
    so shell input must never raise. Booleans are excluded (JSON true
    is not a spacing or an angle).
    """
    try:
        parsed = json.loads(existing or "")
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        key: parsed[key]
        for key in keys
        if isinstance(parsed.get(key), (int, float))
        and not isinstance(parsed.get(key), bool)
    }
