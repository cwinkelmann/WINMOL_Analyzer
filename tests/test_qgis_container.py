"""Guards for the QGIS GUI-testing container (root Dockerfile + startDocker.sh).

The image exists purely to run the plugin under a specific QGIS build over X11,
so nothing about it can be exercised by an automated test: there is no daemon in
CI and no X server. What *can* be pinned down is the contract between the three
files that have to agree, and the two mistakes that already cost a bump:

1. A version-pinned interpreter package. The old file ran
   ``apt install python3.10-venv``, which exists only on Ubuntu 22.04 (jammy,
   the QGIS 3.28 base). Ubuntu 24.04 ships 3.12 and Debian trixie / Ubuntu
   25.10 ship 3.13, so that single line turns any base-image bump into an
   "Unable to locate package" build failure.

2. A floating or OS-ambiguous tag. ``ltr``, ``stable`` and ``latest`` move under
   you, and on this registry the unsuffixed release tags are *not* the LTS
   build: ``ltr``, ``3.44`` and ``3.44.12`` all resolve to the same digest as
   ``3.44.12-questing`` (Ubuntu 25.10, a 9-month interim release). Only the
   ``-noble`` variant is Ubuntu 24.04 LTS. Verified against
   ``hub.docker.com/v2/repositories/qgis/qgis/tags`` on 2026-07-21.

The last test encodes a QGIS rule rather than a Docker one, because it is the
thing that will actually break the QGIS 4 trial the container is meant to
enable. See its docstring.
"""
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKERFILE = os.path.join(REPO_ROOT, "Dockerfile")
START_SCRIPT = os.path.join(REPO_ROOT, "startDocker.sh")
METADATA = os.path.join(REPO_ROOT, "metadata.txt")

# Tags that resolve to something different tomorrow than they do today.
FLOATING_TAGS = {"ltr", "stable", "latest"}


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _instructions(text):
    """The Dockerfile with comment lines removed.

    Comments are prose and legitimately *name* the packages they warn against,
    so scanning the raw text for a banned package matches the warning too.
    """
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith("#"))


def _dockerfile_arg_default(text, name):
    """The default value of a top-level ``ARG <name>=<value>``, or None."""
    m = re.search(
        r"^\s*ARG\s+%s=(\S+)\s*$" % re.escape(name), text, re.MULTILINE)
    return m.group(1) if m else None


@pytest.fixture(scope="module")
def dockerfile():
    return _read(DOCKERFILE)


@pytest.fixture(scope="module")
def start_script():
    return _read(START_SCRIPT)


def test_base_image_is_parameterised(dockerfile):
    """Bumping QGIS must be a --build-arg, not an edit.

    ``ARG`` is only honoured for the base image when it appears *before* the
    ``FROM``, so position is part of the contract, not style.
    """
    lines = [ln.strip() for ln in dockerfile.splitlines()]
    arg_idx = [i for i, ln in enumerate(lines)
               if re.match(r"^ARG\s+QGIS_TAG=", ln)]
    from_idx = [i for i, ln in enumerate(lines) if ln.startswith("FROM ")]

    assert arg_idx, "Dockerfile must declare ARG QGIS_TAG=<default>"
    assert from_idx, "Dockerfile must have a FROM"
    assert arg_idx[0] < from_idx[0], (
        "ARG QGIS_TAG must precede FROM, otherwise Docker does not expand it "
        "in the base image reference")
    from_line = lines[from_idx[0]]
    assert re.search(r"^FROM\s+qgis/qgis:\$\{?QGIS_TAG\}?", from_line), (
        "FROM must reference the ARG, found: %r" % from_line)


def test_default_tag_is_pinned_and_lts(dockerfile):
    """The default must be reproducible and an explicit OS variant.

    An unsuffixed qgis/qgis release tag is the *questing* (Ubuntu 25.10) build,
    not the LTS one -- identical digest, verified on the registry. A long-lived
    image should sit on an LTS base, so require an explicit OS suffix.
    """
    tag = _dockerfile_arg_default(dockerfile, "QGIS_TAG")
    assert tag not in FLOATING_TAGS, (
        "%r moves under you; pin an exact release tag" % tag)
    assert re.match(r"^\d+\.\d+(\.\d+)?-[a-z]+$", tag), (
        "default QGIS_TAG %r must be <version>-<os>, e.g. 3.44.12-noble; a "
        "bare version tag is the interim-release build, not the LTS one" % tag)


def test_no_version_pinned_python_package(dockerfile):
    """``python3.10-venv`` does not exist on any base newer than jammy."""
    hits = re.findall(r"python3\.\d+-\S+", _instructions(dockerfile))
    assert not hits, (
        "version-pinned interpreter packages break every base-image bump; "
        "use the unversioned package instead. Found: %r" % hits)


def test_venv_support_is_installed(dockerfile):
    """The plugin builds a virtualenv on first run, so venv must be present."""
    assert re.search(r"\bpython3-venv\b", _instructions(dockerfile)), (
        "plugin_utils/installer.py creates a venv; the base image needs "
        "python3-venv")


def test_python_symlink_is_idempotent(dockerfile):
    """A bare ``ln -s`` fails if the base already ships /usr/bin/python."""
    body = _instructions(dockerfile)
    links = re.findall(r"ln\s+(-\S+\s+)*/usr/bin/python3\s+/usr/bin/python",
                       body)
    assert links, "Dockerfile should symlink python -> python3"
    assert re.search(r"ln\s+-[a-z]*f[a-z]*\s+/usr/bin/python3", body), (
        "use `ln -sf`: a plain `ln -s` aborts the build if the base image "
        "already provides /usr/bin/python")


def test_start_script_default_matches_dockerfile(dockerfile, start_script):
    """One default, two files -- they must not drift apart."""
    docker_tag = _dockerfile_arg_default(dockerfile, "QGIS_TAG")
    m = re.search(r'QGIS_TAG="\$\{QGIS_TAG:-([^}"]+)\}"', start_script)
    assert m, "startDocker.sh must define an overridable QGIS_TAG default"
    assert m.group(1) == docker_tag, (
        "startDocker.sh default %r != Dockerfile default %r"
        % (m.group(1), docker_tag))


def test_start_script_tags_the_image_with_the_qgis_version(start_script):
    """A 3.44 and a 4.2 image have to be able to coexist on one machine."""
    pattern = r'winmol_analyser_docker:\$\{?QGIS_TAG\}?'
    assert re.search(pattern, start_script), (
        "the image name must embed $QGIS_TAG, otherwise trialling a second "
        "QGIS version silently overwrites the first")


def _effective_qgis_max_version():
    """Apply QGIS's own default for an omitted ``qgisMaximumVersion``.

    From ``python/pyplugin_installer/installer_data.py``::

        if not qgisMaximumVersion:
            qgisMaximumVersion = qgisMinimumVersion[0] + ".99"

    Note ``[0]`` indexes the *string*, so ``qgisMinimumVersion=3.22`` yields
    ``"3.99"`` -- the plugin is declared incompatible with QGIS 4 and the
    plugin manager refuses to enable it, however Qt6-ready the code is.
    """
    text = _read(METADATA)
    minimum = re.search(r"^qgisMinimumVersion=(\S+)", text, re.MULTILINE)
    assert minimum, "metadata.txt must declare qgisMinimumVersion"
    maximum = re.search(r"^qgisMaximumVersion=(\S+)", text, re.MULTILINE)
    if maximum:
        return maximum.group(1)
    return minimum.group(1)[0] + ".99"


def test_metadata_covers_the_qgis_major_the_container_ships(dockerfile):
    """Bumping the image to 4.x without touching metadata.txt is a silent trap.

    The plugin would build and start fine; QGIS would simply list it as
    incompatible with no hint that a one-line metadata field is the cause.
    Fail here instead, where the message says so.
    """
    tag = _dockerfile_arg_default(dockerfile, "QGIS_TAG")
    tag_major = int(tag.split(".")[0])
    max_major = int(_effective_qgis_max_version().split(".")[0])
    assert max_major >= tag_major, (
        "the container defaults to QGIS %d.x but metadata.txt declares "
        "compatibility only up to %s -- QGIS would refuse to enable the "
        "plugin. Raise qgisMaximumVersion." % (tag_major,
                                               _effective_qgis_max_version()))
