"""Make the repo root importable no matter where pytest is invoked
from (repo root or tests/). Several test modules also do this insert
themselves; this covers the ones that import winmol_batch / utils /
plugin_utils directly."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
