"""Building ``$WINMOL_CONFIG_OVERRIDES_JSON`` for the child process.

The QGIS dialog and the child run in different interpreters, and the run
command is five positional arguments -- the ONLY channel that carries a
config value across is this environment variable (``winmol_run.py`` applies
it to ``Config`` before the execution plan is built).

A user may already have set the variable in their own shell before starting
QGIS, so values are **merged**, never clobbered: what the dialog sets wins for
its own keys and everything else survives.

Pure stdlib, no Qt and no QGIS imports, so it is unit-testable off QGIS.
"""

import json

#: The variable winmol_run.py reads.
ENV_VAR = "WINMOL_CONFIG_OVERRIDES_JSON"


def merge(updates, existing=None):
    """Return the JSON object string for :data:`ENV_VAR`.

    ``existing`` is the current value of the variable (or None). Anything
    unparsable is discarded rather than propagated -- the child exits with
    status 2 on invalid JSON, and inheriting a broken value would turn a
    typo in the user's shell profile into a failed run.
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


def batch_override_env(batch, existing=None):
    """``{ENV_VAR: ...}`` pinning the prediction batch size, or ``{}``.

    ``batch`` of 0/None means "Auto" -- the planner sizes the batch and the
    autotune may refine it, so nothing is injected and any value the user set
    themselves is left exactly as it was.
    """
    try:
        value = int(batch)
    except (TypeError, ValueError):
        return {}
    if value < 1:
        return {}
    return {ENV_VAR: merge({"prediction_batch_override": value}, existing)}
