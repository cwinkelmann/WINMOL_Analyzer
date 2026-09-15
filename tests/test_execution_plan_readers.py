"""producer_workers is now 'reader threads per prediction worker':
R = clamp((hw_cpu - 1) // n_gpu, 1, 16); CPU-only is always 1.

Not derived from cpu_workers: that is capped at 32 by max_cpu_workers,
which on an 8-GPU box would give 4 readers per GPU -- ~136 tiles/s
against a card that consumes 250."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from classes.ExecutionPlan import _reader_threads  # noqa: E402


@pytest.mark.parametrize("hw_cpu,n_gpu,cpu_only,expected", [
    (224, 8, False, 16),    # carrot: 223 // 8 = 27 -> cap 16
    (12, 1, False, 11),     # T14
    (4, 1, False, 3),
    (2, 1, False, 1),       # floor
    (1, 1, False, 1),       # floor, never 0
    (64, 2, False, 16),     # 63 // 2 = 31 -> cap
    (224, 8, True, 1),      # CPU-only ignores cores
    (12, 1, True, 1),
])
def test_reader_threads_rule(hw_cpu, n_gpu, cpu_only, expected):
    assert _reader_threads(hw_cpu, n_gpu, cpu_only) == expected


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("WINMOL_PREDICTION_READERS", "5")
    assert _reader_threads(224, 8, False) == 5
    assert _reader_threads(224, 8, True) == 5


def test_env_override_ignored_when_not_an_int(monkeypatch):
    monkeypatch.setenv("WINMOL_PREDICTION_READERS", "lots")
    assert _reader_threads(12, 1, False) == 11
