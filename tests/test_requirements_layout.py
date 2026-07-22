"""The structure of requirements/ is an invariant, not a convention.

Three things used to be enforced only by comments, and comments do not fail
a build:

  1. ``onnxruntime`` and ``onnxruntime-gpu`` provide the SAME Python module,
     so no environment may install both. Before the tidy-up this held only
     because a human remembered that the GPU files had to say ``-r geo.txt``
     and never ``-r base.txt``. Now the shared ancestor (core.txt) names no
     runtime at all, and the first test here proves it for EVERY file rather
     than for the two the plugin happens to install.
  2. The CUDA version window (``>=1.26,<1.27``) was duplicated across three
     files and had already drifted — the container's copy had lost the
     ceiling. It now lives in cuda.txt alone, and that is asserted.
  3. A partial rename of cpu.txt / gpu.txt does not raise: the installer
     falls back to the CPU file, so a GPU machine silently gets a CPU
     environment. Existence is asserted here so CI catches it instead.

Pure file I/O over the repo — no pip, no network, no GPU.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plugin_utils.installer as inst   # noqa: E402

REQ_DIR = inst.repo_requirements_dir()
REQ_FILES = sorted(p.name for p in REQ_DIR.glob("*.txt"))

#: The one string that decides which CUDA userspace an environment gets.
CUDA_REQUIREMENT = "onnxruntime-gpu[cuda,cudnn]>=1.26,<1.27"


def _closure(name):
    """Package names reachable from a requirements file, following ``-r``."""
    return inst._requirement_names(REQ_DIR / name)


def test_the_directory_is_not_empty():
    """Guards every parametrised test below: an empty glob would make them
    all vacuously pass."""
    assert len(REQ_FILES) >= 5, REQ_FILES


# --- rule 1: exactly one inference runtime ---------------------------------

@pytest.mark.parametrize("name", REQ_FILES)
def test_no_requirements_file_installs_both_runtimes(name):
    """onnxruntime and onnxruntime-gpu both provide the `onnxruntime`
    module; with both installed the loser's shared libraries shadow the
    winner's and imports fail in ways that look like a broken CUDA install.

    Asserted over the whole directory, so a future file cannot reintroduce
    the mistake by innocently doing `-r cpu.txt` next to a CUDA line.
    """
    names = set(_closure(name))
    assert not ({"onnxruntime", "onnxruntime-gpu"} <= names), (
        "%s pulls in BOTH runtimes: %s" % (name, sorted(names)))


def test_the_shared_fragment_names_no_runtime():
    """core.txt is what makes rule 1 structural rather than remembered: the
    only file both cpu.txt and gpu.txt include must carry no runtime, so
    there is no include that COULD leak one into the other."""
    names = _closure("core.txt")
    assert "onnxruntime" not in names
    assert "onnxruntime-gpu" not in names


def test_the_installable_files_each_declare_exactly_one_runtime():
    assert "onnxruntime" in _closure("cpu.txt")
    assert "onnxruntime-gpu" in _closure("gpu.txt")
    assert "onnxruntime" in _closure("ci.txt")
    assert "onnxruntime-gpu" in _closure("cuda.txt")


def test_the_notebook_extras_stay_out_of_the_compute_environments():
    """They are 32 packages (ipython, jupyter_client, pyzmq, debugpy,
    tornado, fonttools...) that nothing on the compute path imports. The
    only matplotlib site in the codebase is a lazy import inside
    utils.IO.save_image; an eager one broke the GPU image once already
    (docker/gpu/Dockerfile), which is why this is a test."""
    for name in ("cpu.txt", "gpu.txt", "ci.txt", "core.txt"):
        names = _closure(name)
        assert "matplotlib" not in names, name
        assert "ipykernel" not in names, name
    assert "matplotlib" in _closure("notebook.txt")


# --- rule 2: one place that names the CUDA runtime -------------------------

def _payload(name):
    """The installable lines of a requirements file — comments stripped."""
    return [ln.strip() for ln in (REQ_DIR / name).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def test_only_cuda_txt_REQUIRES_the_gpu_runtime():
    """Anti-drift guard. gpu.txt and plugin-gpu.txt used to duplicate this
    line and had already diverged: one carried the <1.27 ceiling and the
    other did not, so the container was one NVIDIA release away from
    silently switching to CUDA 13.

    Comments elsewhere are free to explain the rule — it is the REQUIREMENT
    that must exist in exactly one place.
    """
    owners = [n for n in REQ_FILES
              if any("onnxruntime-gpu" in ln for ln in _payload(n))]
    assert owners == ["cuda.txt"], owners


def test_the_cuda_version_window_is_pinned_at_both_ends():
    """CEILING: onnxruntime-gpu 1.27 moved its extras to CUDA 13 and fails
    at import with `ImportError: libcudart.so.13` on a cu12 driver.

    FLOOR: an unsatisfiable extra is a pip WARNING, not an error — without a
    lower bound pip can backtrack to a release that has no [cuda,cudnn]
    extras, install zero nvidia wheels and exit 0, leaving a build that
    claims success and then runs on the CPU.
    """
    assert _payload("cuda.txt") == [CUDA_REQUIREMENT], _payload("cuda.txt")


def test_the_cuda_pin_reaches_the_gpu_environment():
    """gpu.txt must get the window by INCLUDING cuda.txt, not by copying
    it — that copy is what drifted last time."""
    body = (REQ_DIR / "gpu.txt").read_text()
    assert re.search(r"^-r\s+cuda\.txt\s*$", body, re.M), body


# --- rule 3: a rename must fail loudly -------------------------------------

@pytest.mark.parametrize("name", ["core.txt", "cpu.txt", "gpu.txt",
                                  "cuda.txt", "ci.txt", "notebook.txt",
                                  "convert.txt", "dev.txt"])
def test_every_expected_file_exists(name):
    """plugin_requirements_path() falls back to the CPU file when the GPU
    one is missing, and _variant_hashes() hard-codes the GPU name a second
    time, so a partial rename produces NO error — GPU machines just quietly
    get CPU environments. This is the backstop for that."""
    assert (REQ_DIR / name).exists()


def test_the_installer_and_the_directory_agree():
    assert (REQ_DIR / inst.CPU_REQUIREMENTS).exists()
    assert (REQ_DIR / inst.GPU_REQUIREMENTS).exists()
    assert inst.plugin_requirements_path().name == inst.CPU_REQUIREMENTS
    assert (inst.plugin_requirements_path(gpu=True).name
            == inst.GPU_REQUIREMENTS)


# --- the install table cannot go stale -------------------------------------

@pytest.mark.parametrize("name", REQ_FILES)
def test_every_requirements_file_is_documented(name):
    """requirements/README.md is the canonical "which file do I install"
    table and ships inside the plugin ZIP with the directory. A file nobody
    can find is the problem this whole cleanup exists to fix."""
    readme = (REQ_DIR / "README.md").read_text()
    assert name in readme, "%s is missing from requirements/README.md" % name


def test_every_requirements_file_says_who_installs_it_and_when():
    """Plain-language header, first line, every file. Twelve files with no
    audience stated is how the directory became unreadable."""
    for name in REQ_FILES:
        head = (REQ_DIR / name).read_text().splitlines()
        assert head and head[0].startswith("#"), name
        block = "\n".join(ln for ln in head[:12] if ln.startswith("#"))
        assert "WHO:" in block or "NOBODY INSTALLS THIS" in block, name


# --- the rename must not force a 2.4 GB reinstall --------------------------

@pytest.mark.parametrize("variant,digest", [
    ("cpu", "ace1011007ef1edc"),     # plugin.txt, v0.6.0.1 .. v0.6.1-rc4
    ("gpu", "b84b6df907e74baa"),     # plugin-gpu.txt, v0.6.1-rc2 .. rc4
])
def test_an_environment_installed_before_the_rename_is_still_ready(
        tmp_path, variant, digest):
    """The venv sentinel stores the hash of the file it was installed from,
    so renaming plugin.txt -> cpu.txt would declare every existing
    environment stale and offer a few-hundred-MB (CPU) or ~2.4 GB (GPU)
    re-download for an environment that already works.

    Those closures are a SUPERSET of today's — same pins plus the notebook
    extras that moved to notebook.txt, and not one version of the packages
    that remain changed — so accepting them is correct, not a fudge.
    """
    import json
    venv = tmp_path / "winmol_venv"
    venv.mkdir()
    with open(inst._marker_path(str(venv)), "w") as f:
        json.dump({"req_hash": digest, "variant": variant}, f)
    assert inst.marker_variant(str(venv)) == variant
    assert inst.marker_matches(str(venv)) is True


def test_an_unrelated_digest_is_still_rejected(tmp_path):
    """The allowlist must not become "any hash is fine"."""
    import json
    venv = tmp_path / "winmol_venv"
    venv.mkdir()
    with open(inst._marker_path(str(venv)), "w") as f:
        json.dump({"req_hash": "0123456789abcdef"}, f)
    assert inst.marker_variant(str(venv)) is None
    assert inst.marker_matches(str(venv)) is False
