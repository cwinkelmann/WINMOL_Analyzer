"""Make the repo root importable no matter where pytest is invoked
from (repo root or tests/). Several test modules also do this insert
themselves; this covers the ones that import winmol_batch / utils /
plugin_utils directly."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture(autouse=True)
def _isolated_autotune_cache(tmp_path, monkeypatch):
    """Give every test its own throwaway autotune cache file.

    Without this, ``_autotune_batch_size`` (utils/Prediction.py) resolves
    ``plugin_utils.autotune_cache.cache_path()`` to the SAME per-user
    location a real WINMOL run uses (e.g. ``~/Library/Caches/winmol`` on
    macOS) whenever a test does not set ``$WINMOL_AUTOTUNE_CACHE`` itself.
    That would (a) leave real files behind on the machine running the
    suite, and (b) let two unrelated tests whose (fake model, Config) hash
    to the same cache key leak a batch size between them. A test that
    wants a specific cache file (or wants to assert the real per-user
    default) still wins by setting ``WINMOL_AUTOTUNE_CACHE`` itself after
    this fixture runs -- monkeypatch layers cleanly.
    """
    monkeypatch.setenv(
        "WINMOL_AUTOTUNE_CACHE", str(tmp_path / "autotune-test-cache.json"))
