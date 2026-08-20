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


def batch_override_env(batch, existing=None):
    """``{ENV_VAR: ...}`` pinning the prediction batch size, or ``{}``.

    ``batch`` of 0/None means "Auto": nothing is injected and any value
    the user set themselves is left exactly as it was.
    """
    try:
        value = int(batch)
    except (TypeError, ValueError):
        return {}
    if value < 1:
        return {}
    return {ENV_VAR: merge({"prediction_batch_override": value}, existing)}
