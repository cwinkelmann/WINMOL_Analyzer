"""Minimal verbosity control for the pipeline's stdout log.

Three levels:

* ``quiet``  - warnings and errors only.
* ``normal`` - the default: phase banners, progress, summaries and results.
* ``debug``  - everything, including the per-tile merge diagnostics
  (MERGE DISCOVERY / MERGE INPUT / MERGE READ OK ...) that were added to
  debug real merge bugs and are far too chatty for a normal run.

Warnings and errors are NEVER suppressed, at any level.

Selection order (first hit wins)::

    WINMOL_LOG_LEVEL=quiet|normal|debug   env
    WINMOL_VERBOSE=1                      env, implies debug
    Config.log_level                      config / overrides JSON
    Config.vector_debug=True              legacy knob, implies debug

The resolved level is written back into ``WINMOL_LOG_LEVEL`` so that the
vector-tile multiprocessing pool - whose workers are *spawned* on macOS and
therefore do not inherit module globals - can re-derive it on entry.

Deliberately stdlib-only and printing to stdout, because the QGIS plugin
reads the child process' merged stdout/stderr line by line.
"""
from __future__ import annotations

import os

QUIET = 0
NORMAL = 1
DEBUG = 2

LEVELS = {"quiet": QUIET, "normal": NORMAL, "debug": DEBUG}
LEVEL_NAMES = {value: name for name, value in LEVELS.items()}

_level = NORMAL

# The value this process exported into WINMOL_LOG_LEVEL. A spawned worker
# starts with None here, so the inherited env var wins there; in the process
# that wrote it, a later configure_from_config() must not be pinned by its
# own export.
_exported = None


def current_level():
    """Return the active level as an int (QUIET / NORMAL / DEBUG)."""
    return _level


def current_level_name():
    return LEVEL_NAMES.get(_level, "normal")


def set_level(level):
    """Set the active level from a name or an int. Unknown -> normal."""
    global _level
    if isinstance(level, str):
        _level = LEVELS.get(level.strip().lower(), NORMAL)
    elif isinstance(level, int) and level in LEVEL_NAMES:
        _level = level
    else:
        _level = NORMAL
    return _level


def resolve_level(config=None):
    """Resolve the level name from the environment, then the config."""
    env_level = os.environ.get("WINMOL_LOG_LEVEL", "").strip().lower()
    if env_level and env_level != _exported:
        return env_level if env_level in LEVELS else "normal"
    if os.environ.get("WINMOL_VERBOSE", "").strip().lower() in (
            "1", "true", "yes", "on"):
        return "debug"
    name = "normal"
    if config is not None:
        raw = getattr(config, "log_level", None)
        if isinstance(raw, str) and raw.strip().lower() in LEVELS:
            name = raw.strip().lower()
        # Legacy knob: asking for vector debug output means debug.
        if getattr(config, "vector_debug", False):
            name = "debug"
    return name


def configure_from_config(config=None):
    """Apply the resolved level and export it for spawned workers."""
    global _exported
    name = resolve_level(config)
    set_level(name)
    os.environ["WINMOL_LOG_LEVEL"] = name
    _exported = name
    return _level


def is_debug():
    return _level >= DEBUG


def info(*args, **kwargs):
    """Print at normal verbosity and above."""
    if _level >= NORMAL:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def debug(*args, **kwargs):
    """Print only at debug verbosity."""
    if _level >= DEBUG:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def warn(*args, **kwargs):
    """Print a warning. Never suppressed."""
    kwargs.setdefault("flush", True)
    print("WARNING:", *args, **kwargs)


def error(*args, **kwargs):
    """Print an error. Never suppressed."""
    kwargs.setdefault("flush", True)
    print("ERROR:", *args, **kwargs)
