"""The GPU offer has to find the user, not wait to be found.

The bug this file pins, diagnosed on a Lenovo with an RTX 4080 SUPER and
driver 580.159.03 running v0.6.1-rc2: everything WORKED. detect_gpu()
returned ok, wants_gpu_runtime() returned True, and
accelerator_from_machine() produced exactly the right sentence —

    "NVIDIA GeForce RTX 4080 SUPER found, but the installed runtime is
     CPU-only, so the GPU is sitting idle..."

...on the Setup tab, which the user had no reason to open. Their
environment PRE-EXISTED as a valid CPU install, so "Create environment for
me" (the one path that offers the CUDA runtime) never ran again, and the
env_gpu_button was the only surviving entry point. A full detection ran at
5265.7 ms per tile against a measured 12.4 ms on CUDA — 423x — with the
card idle throughout, and nothing ever said so.

So the decisions tested here are about REACH:

  * pre_run_decision — Run stops and asks, once, at the only moment the
    answer is actionable;
  * should_nudge / accel_nudge_text — the Detection tab carries the same
    warning, so it is visible without opening Setup;
  * and the negative half, which matters just as much: a machine that can
    do nothing about it (no GPU, an Apple GPU, a GPU already in use, a GPU
    we must not sell a download for) is never nagged, and an unfinished
    probe never delays a run.

All headless: no QGIS, no Qt, no GPU, no network. The Qt side's one
irreducible static rule -- that none of this measures anything on the GUI
thread -- lives in tests/test_dialog_lint.py.
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plugin_utils import installer as inst        # noqa: E402
from plugin_utils import setup_state as ss        # noqa: E402


GPU = "NVIDIA GeForce RTX 4080 SUPER"


def _accel(state, can_install=False, label=GPU, text="..."):
    return ss.AcceleratorStatus(state=state, text=text,
                                can_install=can_install, gpu_label=label)


#: Every state the accelerator can be in, with the ONE that may prompt
#: marked. can_install mirrors what accelerator_status() really produces
#: for each: only gpu_idle sets it, which is why gpu_broken and
#: gpu_unusable — GPUs we can see but must not offer 2.4 GB for — are in
#: the silent column despite having hardware.
ALL_STATES = [
    (ss.ACCEL_GPU_IDLE, True, True),
    (ss.ACCEL_GPU_ACTIVE, False, False),
    (ss.ACCEL_COREML, False, False),
    (ss.ACCEL_CPU_ONLY, False, False),
    (ss.ACCEL_UNKNOWN, False, False),
    (ss.ACCEL_GPU_BROKEN, False, False),
    (ss.ACCEL_GPU_UNUSABLE, False, False),
]


# --- the pre-run decision ---------------------------------------------------

@pytest.mark.parametrize("state,can_install,prompts", ALL_STATES)
def test_only_an_idle_gpu_interrupts_the_run(state, can_install, prompts):
    decision = ss.pre_run_decision(_accel(state, can_install))
    assert decision.asks is prompts, (
        f"{state} should {'' if prompts else 'not '}stop a run")


def test_a_machine_with_no_gpu_is_never_nagged():
    """The rule that keeps this feature from being resented: there is
    nothing a CPU-only box can do, so telling it again is noise."""
    decision = ss.pre_run_decision(_accel(ss.ACCEL_CPU_ONLY))
    assert not decision.asks
    assert ss.accel_nudge_text(_accel(ss.ACCEL_CPU_ONLY)) == ""


def test_an_apple_gpu_is_left_alone():
    """CoreML comes out of the stock wheel; there is no download that
    would make it faster, and 'install the GPU runtime' would send a Mac
    user after a CUDA build that does not exist for them."""
    accel = _accel(ss.ACCEL_COREML, label="Apple GPU")
    assert not ss.pre_run_decision(accel).asks
    assert not ss.should_nudge(accel)


def test_a_gpu_already_in_use_is_left_alone():
    assert not ss.pre_run_decision(_accel(ss.ACCEL_GPU_ACTIVE)).asks


def test_a_broken_gpu_runtime_is_not_offered_the_same_download_again():
    """gpu_broken means the 2.4 GB already landed and CUDA still is not
    usable. Re-offering it would be cruel and would not help."""
    assert not ss.pre_run_decision(_accel(ss.ACCEL_GPU_BROKEN)).asks


def test_the_prompt_states_the_numbers_and_the_download_size():
    """"GPU recommended" is advice nobody acts on. The measured figures
    and the cost are the whole reason this interruption is justified."""
    decision = ss.pre_run_decision(_accel(ss.ACCEL_GPU_IDLE, True))
    assert GPU in decision.text
    assert "12 ms" in decision.text and "5 s" in decision.text
    assert "2.4 GB" in decision.text
    assert "2.4 GB" in decision.install_label
    # ...and a real way to decline, not just an X in the corner.
    assert decision.run_label and "CPU" in decision.run_label


# --- the dismissal ----------------------------------------------------------

def test_the_choice_to_run_on_the_cpu_is_remembered():
    """"Run anyway" must proceed immediately AND stop the question coming
    back on every single run."""
    accel = _accel(ss.ACCEL_GPU_IDLE, True)
    first = ss.pre_run_decision(accel, dismissed=None)
    assert first.asks
    again = ss.pre_run_decision(accel, dismissed=first.token)
    assert not again.asks, "the question came back after it was answered"


def test_the_dismissal_survives_being_read_back_as_a_string():
    """QgsSettings hands back whatever it stored, typically as a str; the
    comparison must not depend on the type."""
    accel = _accel(ss.ACCEL_GPU_IDLE, True)
    token = ss.pre_run_decision(accel).token
    assert not ss.pre_run_decision(accel, dismissed=str(token)).asks


def test_a_different_gpu_asks_once_more():
    """A dismissal is an answer about THIS machine. Swap the card and the
    user has never been told about the new one."""
    token = ss.pre_run_decision(_accel(ss.ACCEL_GPU_IDLE, True)).token
    other = _accel(ss.ACCEL_GPU_IDLE, True, label="NVIDIA RTX 6000 Ada")
    assert ss.pre_run_decision(other, dismissed=token).asks


def test_an_empty_or_unrelated_stored_value_still_asks():
    accel = _accel(ss.ACCEL_GPU_IDLE, True)
    for stored in ("", None, "true", "winmol/whatever"):
        assert ss.pre_run_decision(accel, dismissed=stored).asks


def test_the_settings_key_lives_beside_the_interpreter_key():
    """Same group, same discipline — installer names the key, the dialog
    is the only thing that touches QgsSettings."""
    assert inst.QSETTINGS_GPU_PROMPT_KEY.startswith("winmol/")
    assert inst.QSETTINGS_GPU_PROMPT_KEY != inst.QSETTINGS_PYTHON_KEY


# --- never block the run ----------------------------------------------------

def test_an_unfinished_probe_never_stops_a_run():
    """accelerator_from_machine spawns a child interpreter and takes
    SECONDS. If Run arrives before it lands, the run wins — a user who
    pressed Run is not made to wait on a measurement they may not need."""
    assert not ss.pre_run_decision(None).asks
    assert not ss.pre_run_decision(ss.accelerator_status()).asks


def test_the_unmeasured_status_the_dialog_paints_with_asks_nothing():
    """This is literally what _accel_status() returns before the probe
    reports: accelerator_status() with no arguments."""
    unmeasured = ss.accelerator_status()
    assert unmeasured.state == ss.ACCEL_UNKNOWN
    assert not unmeasured.can_install
    assert not ss.pre_run_decision(unmeasured).asks
    assert ss.accel_nudge_text(unmeasured) == ""


def test_a_state_that_claims_gpu_idle_without_can_install_is_not_trusted():
    """The nudge and the Setup button must never disagree about whether
    there is an action behind the sentence."""
    assert not ss.pre_run_decision(_accel(ss.ACCEL_GPU_IDLE, False)).asks
    assert not ss.should_nudge(_accel(ss.ACCEL_GPU_IDLE, False))


# --- the second surface: the Detection tab ---------------------------------

@pytest.mark.parametrize("state,can_install,shown", ALL_STATES)
def test_the_detection_tab_line_appears_for_the_same_single_state(
        state, can_install, shown):
    text = ss.accel_nudge_text(_accel(state, can_install))
    assert bool(text) is shown


def test_the_detection_tab_line_is_short_and_names_the_gpu():
    """It sits above the input fields on a tab full of parameters; a
    paragraph there would be ignored like any other paragraph."""
    text = ss.accel_nudge_text(_accel(ss.ACCEL_GPU_IDLE, True))
    assert GPU in text
    assert "\n" not in text
    assert len(text) < 200
    assert "12 ms" in text


def test_the_nudge_survives_a_dismissal():
    """Dismissing the MODAL must not remove the way back. One gray line
    with a button is the entry point the Setup tab never was."""
    accel = _accel(ss.ACCEL_GPU_IDLE, True)
    token = ss.pre_run_decision(accel).token
    assert not ss.pre_run_decision(accel, dismissed=token).asks
    assert ss.accel_nudge_text(accel), (
        "the dismissal removed the only remaining pointer to the offer")


def test_accelerator_from_machine_still_produces_the_reported_state(
        monkeypatch):
    """End to end through the pure half, with the exact machine from the
    report described rather than owned: an RTX 4080 SUPER, a CPU-only
    onnxruntime, no CUDA provider. It must reach the prompt."""
    class Probe:
        present = True
        has_hardware = True
        label = GPU
        detail = ""

    report = {"ok": True,
              "providers": ["AzureExecutionProvider",
                            "CPUExecutionProvider"],
              "packages": ["onnxruntime"],
              "version": "1.27.0", "error": None,
              "session_providers": None}
    accel = ss.accelerator_status(Probe(), report)
    assert accel.state == ss.ACCEL_GPU_IDLE and accel.can_install
    assert ss.pre_run_decision(accel).asks
    assert ss.should_nudge(accel)
