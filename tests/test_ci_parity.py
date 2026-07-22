"""CI must test the environment the CLI, the plugin and Docker actually run.

This is a drift alarm, not a style check. The divergence it exists to catch
was measured, not hypothetical: ci.txt hung off core.txt with its own pins, so
CI installed ``onnxruntime==1.19.2`` while the QGIS plugin's venv resolved
``1.27.0`` on the same day, and CI installed no ``psutil`` at all -- the
package ``classes/HardwareInfo.py`` uses to report RAM and
``utils/Prediction.py`` uses for the autotune memory ceiling. Both code paths
were reachable in CI only through test stubs, which is the most plausible
reason a green suite kept missing accelerator bugs that turned up by hand.

The check is deliberately OFFLINE: it compares the declared ``-r`` closures of
the requirements files, so it runs in any environment with no pip, no network
and no resolver. A real resolve is a different job --
``.github/workflows/plugin-env.yml`` pip-installs cpu.txt on Linux, Windows
and macOS -- and the two are complementary: this test catches a declaration
drifting apart, that workflow catches a declaration that cannot be satisfied.

Every failure message names the package that drifted.
"""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plugin_utils.installer as inst   # noqa: E402

REQ_DIR = inst.repo_requirements_dir()
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fixtures")

#: The file that defines the SHIPPED environment -- CLI, plugin venv,
#: standalone scripts. Everything else is measured against it.
SHIPPED = "cpu.txt"

#: The file the CI image is built from.
CI = "ci.txt"

#: The only packages ci.txt may add on top of the shipped environment. A test
#: runner is not something the analyzer imports, so it cannot change what the
#: pipeline computes. Anything else added here means CI is exercising an
#: environment no user has.
TEST_ONLY = frozenset({"pytest"})


def _canonical(name):
    """PEP 503 normalisation: Shapely, shapely and scikit_image all collide
    with their distribution name, so compare on this and not on raw text."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _split_requirement(line):
    """('numpy', '==1.26.4') from 'numpy==1.26.4', extras and markers kept
    out of the name so onnxruntime-gpu[cuda,cudnn] compares as one package."""
    text = line.split("#", 1)[0].strip()
    match = re.match(r"^([A-Za-z0-9._-]+)", text)
    name = _canonical(match.group(1))
    return name, text[match.end():].strip()


def _declared(name, seen=None):
    """{package: full requirement text} reachable from a file through ``-r``.

    Mirrors what pip does with an include graph, minus the resolving: the
    point is to compare what the files SAY, offline.
    """
    seen = seen if seen is not None else set()
    if name in seen:
        return {}
    seen.add(name)
    out = {}
    for raw in (REQ_DIR / name).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        include = re.match(r"^-r\s+(\S+)", line)
        if include:
            out.update(_declared(include.group(1), seen))
            continue
        package, spec = _split_requirement(line)
        out[package] = spec
    return out


# --- the headline invariant ------------------------------------------------

def test_ci_installs_the_shipped_environment_plus_test_tools_only():
    """ci.txt = cpu.txt + TEST_ONLY. Nothing more, nothing less.

    Named per package so the failure says WHAT drifted rather than dumping
    two lists and leaving the reader to diff them.
    """
    shipped = _declared(SHIPPED)
    ci = _declared(CI)

    extra = sorted(set(ci) - set(shipped) - TEST_ONLY)
    assert not extra, (
        "%s installs packages the shipped environment (%s) does not: %s. "
        "CI would be validating an environment no user has. Either add them "
        "to %s so users get them too, or to TEST_ONLY if they are genuinely "
        "test-only." % (CI, SHIPPED, extra, SHIPPED))

    missing = sorted(set(shipped) - set(ci))
    assert not missing, (
        "%s is MISSING packages the shipped environment installs: %s. "
        "Whatever those packages guard is exercised in CI only through "
        "stubs -- this is exactly how psutil went untested." % (CI, missing))


def test_no_package_is_declared_differently_in_ci_and_in_the_shipped_env():
    """The 1.19.2-vs-1.27.0 regression, expressed as an assertion.

    A shared package pinned two ways is worse than a missing one: the suite
    stays green while testing a different build of the very library the
    accelerator bugs lived in.
    """
    shipped = _declared(SHIPPED)
    ci = _declared(CI)
    drifted = [
        "%s: %s declares %r, %s declares %r"
        % (package, SHIPPED, shipped[package] or "(unconstrained)",
           CI, ci[package] or "(unconstrained)")
        for package in sorted(set(shipped) & set(ci))
        if shipped[package] != ci[package]
    ]
    assert not drifted, (
        "CI and the shipped environment disagree about %d package(s):\n  %s\n"
        "A version constraint must be written in ONE place -- put it in "
        "core.txt or %s, where users receive it too."
        % (len(drifted), "\n  ".join(drifted), SHIPPED))


@pytest.mark.parametrize("package", ["onnxruntime", "psutil"])
def test_the_two_packages_that_actually_drifted_are_covered(package):
    """Regression guard with names on it, so a future refactor that quietly
    drops one of these from the comparison still fails.

    onnxruntime: the inference runtime CI pinned three years behind users.
    psutil: absent from CI entirely, so HardwareInfo's RAM detection and the
    autotune memory ceiling ran only against stubs.
    """
    shipped = _declared(SHIPPED)
    ci = _declared(CI)
    assert package in shipped, "%s lost %s" % (SHIPPED, package)
    assert package in ci, (
        "%s no longer reaches %s -- that is the original bug" % (package, CI))
    assert shipped[package] == ci[package], (
        "%s: %s says %r but %s says %r"
        % (package, SHIPPED, shipped[package], CI, ci[package]))


def test_ci_reaches_the_shipped_environment_by_including_it():
    """`-r cpu.txt`, not a copy of its contents. A copy is what drifts."""
    body = (REQ_DIR / CI).read_text()
    assert re.search(r"^-r\s+%s\s*$" % re.escape(SHIPPED), body, re.M), (
        "%s must include %s with `-r`; duplicating its lines is how the two "
        "came apart last time." % (CI, SHIPPED))


def test_every_include_target_is_a_file_that_ships():
    """scripts/build_plugin_zip.sh copies requirements/ wholesale, so every
    `-r` target must be a real file IN that directory.

    An include pointing outside it, or at a file deleted in a tidy-up, does
    not fail here in the repo -- it fails on a user's machine, halfway
    through pip, after the plugin has already created the venv. This is the
    reason cuda.txt was inlined rather than kept as a hop.
    """
    for name in sorted(p.name for p in REQ_DIR.glob("*.txt")):
        for raw in (REQ_DIR / name).read_text().splitlines():
            include = re.match(r"^-r\s+(\S+)",
                               raw.split("#", 1)[0].strip())
            if not include:
                continue
            target = include.group(1)
            assert "/" not in target and "\\" not in target, (
                "%s includes %s from outside requirements/; the plugin ZIP "
                "ships only this directory" % (name, target))
            assert (REQ_DIR / target).exists(), (
                "%s includes %s, which does not exist -- pip would fail "
                "mid-install on a user's machine" % (name, target))


# --- the pins that genuinely determine the fixtures ------------------------

def _manifest_versions():
    path = os.path.join(FIXTURES_DIR, "manifest.json")
    if not os.path.exists(path):
        pytest.skip("fixtures manifest absent")
    with open(path) as handle:
        return json.load(handle).get("versions", {})


def test_the_fixture_relevant_pins_live_in_the_shipped_environment():
    """tests/fixtures/manifest.json records the versions that produced the
    golden fixtures. That list IS the definition of fixture-relevant, and
    every entry must be constrained in the environment users install --
    otherwise "the fixtures are reproducible under these versions" is a claim
    about the CI image only, which is what it used to be for numpy.

    python is excluded: it is not a pip requirement.
    """
    shipped = _declared(SHIPPED)
    versions = _manifest_versions()
    unconstrained = sorted(
        package for package in versions
        if package != "python" and not shipped.get(_canonical(package)))
    assert not unconstrained, (
        "manifest.json says the golden fixtures were generated under a "
        "specific %s, but the shipped environment (%s) does not constrain "
        "it. Users -- and therefore CI -- can float to any version."
        % (unconstrained, SHIPPED))


def test_an_exact_pin_matches_the_version_that_made_the_fixtures():
    """A pin that disagrees with the manifest is worse than no pin: it
    asserts reproducibility against a version that never produced them."""
    shipped = _declared(SHIPPED)
    for package, version in _manifest_versions().items():
        if package == "python":
            continue
        spec = shipped.get(_canonical(package), "")
        if spec.startswith("=="):
            assert spec == "==" + version, (
                "%s is pinned %r but the golden fixtures were generated "
                "under %s (tests/fixtures/manifest.json). Regenerate the "
                "fixtures or correct the pin."
                % (package, spec, version))


# --- the GPU path ----------------------------------------------------------

def test_the_gpu_environment_has_no_second_declaration_to_drift():
    """docker/gpu/Dockerfile and the plugin's GPU venv install the SAME file,
    so GPU parity holds by construction rather than by comparison.

    Worth asserting anyway, because the previous split (gpu.txt vs
    plugin-gpu.txt) had already drifted: the container's copy of the CUDA
    window had lost its `<1.27` ceiling while the plugin's kept it, leaving
    the image one NVIDIA release from silently switching to CUDA 13.
    """
    assert inst.GPU_REQUIREMENTS == "gpu.txt"
    dockerfile = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docker", "gpu", "Dockerfile")
    body = open(dockerfile).read()
    assert "-r /tmp/req/gpu.txt" in body, (
        "the CUDA image must install requirements/gpu.txt itself, not a "
        "second list that can drift from what the plugin installs")


def test_the_gpu_and_shipped_environments_differ_only_in_the_runtime():
    """gpu.txt is cpu.txt with the CUDA wheel swapped in. Anything else that
    differs is a package the GPU user silently does or does not get."""
    shipped = _declared(SHIPPED)
    gpu = _declared("gpu.txt")
    runtimes = {"onnxruntime", "onnxruntime-gpu"}
    difference = sorted(
        (set(shipped) ^ set(gpu))
        | {p for p in set(shipped) & set(gpu) if shipped[p] != gpu[p]})
    assert set(difference) <= runtimes, (
        "gpu.txt and %s differ by more than the inference runtime: %s"
        % (SHIPPED, [p for p in difference if p not in runtimes]))
