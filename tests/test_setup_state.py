"""The pre-run idle-GPU decision: pure setup_state functions.

The dialog's _gpu_offer_pre_flight puts a modal on screen ONLY when
``pre_run_decision`` says so; these are the rules that keep the
question rare — CPU-only install AND a present GPU AND not previously
dismissed on this machine/configuration.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from plugin_utils import setup_state  # noqa: E402


TOKEN = setup_state.accelerator_token("NVIDIA GeForce RTX 4080 SUPER")


def test_offer_when_cpu_variant_gpu_present_undismissed():
    assert setup_state.pre_run_decision(
        "cpu", True, "", TOKEN) == setup_state.PRERUN_OFFER


def test_run_cpu_when_already_dismissed_for_this_machine():
    assert setup_state.pre_run_decision(
        "cpu", True, TOKEN, TOKEN) == setup_state.PRERUN_RUN_CPU


def test_asks_again_when_the_gpu_changed():
    # A dismissal is an answer about THIS machine/config; a different
    # stored token means the hardware changed.
    other = setup_state.accelerator_token("NVIDIA T4")
    assert setup_state.pre_run_decision(
        "cpu", True, other, TOKEN) == setup_state.PRERUN_OFFER


def test_run_cpu_without_gpu():
    assert setup_state.pre_run_decision(
        "cpu", False, "", TOKEN) == setup_state.PRERUN_RUN_CPU


def test_run_cpu_when_gpu_variant_installed():
    assert setup_state.pre_run_decision(
        "gpu", True, "", TOKEN) == setup_state.PRERUN_RUN_CPU


def test_run_cpu_without_managed_sentinel():
    # BYO interpreter / no venv: installed_variant is None — there is
    # nothing to install the GPU runtime into.
    assert setup_state.pre_run_decision(
        None, True, "", TOKEN) == setup_state.PRERUN_RUN_CPU


def test_token_is_per_machine_and_never_empty():
    assert TOKEN == "gpu_idle|NVIDIA GeForce RTX 4080 SUPER"
    assert setup_state.accelerator_token("") == "gpu_idle|An NVIDIA GPU"


def test_should_nudge_matches_the_offer_precondition():
    assert setup_state.should_nudge("cpu", True)
    assert not setup_state.should_nudge("gpu", True)
    assert not setup_state.should_nudge("cpu", False)
    assert not setup_state.should_nudge(None, True)


def test_nudge_text_names_the_gpu_and_the_cost():
    text = setup_state.accel_nudge_text("NVIDIA T4")
    assert "NVIDIA T4" in text
    assert "5 s" in text and "12 ms" in text
