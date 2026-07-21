"""Release packaging: the zip filename must carry the plugin version.

Regression cover for two production defects observed on the tags
``v0.0.0-demo1`` and ``models-onnx-v1``:

1. every release shipped an asset literally named
   ``WINMOL_Analyzer_QGIS_Plugin.zip``, so two downloads were
   indistinguishable on disk;
2. the workflow triggered on ``*``, so the model-asset tag
   ``models-onnx-v1`` produced a plugin whose ``metadata.txt`` read
   ``version=models-onnx-v1`` — not a valid QGIS version string.

The version-derivation rules live in ``scripts/plugin_version.py`` so they
are testable without QGIS, TensorFlow or a network.
"""

import os
import shutil
import subprocess
import sys
import zipfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import plugin_version  # noqa: E402


# -- version derivation -------------------------------------------------

@pytest.mark.parametrize("tag,expected", [
    # The project's own convention: four segments, not strict semver.
    ("v0.6.0.2", "0.6.0.2"),
    ("v0.5.0", "0.5.0"),
    # Pre-release suffix is kept verbatim; QGIS compares it lexically and
    # sorts it below 0.6.0, which is what a throwaway demo build should do.
    ("v0.0.0-demo1", "0.0.0-demo1"),
    ("v1.0.0-rc.1", "1.0.0-rc.1"),
    # A missing leading "v" is tolerated.
    ("0.6.0", "0.6.0"),
    ("1", "1"),
    # Only ONE leading "v" is stripped.
    ("vv1.0", None),
])
def test_derive_version(tag, expected):
    if expected is None:
        with pytest.raises(ValueError):
            plugin_version.derive_version(tag)
    else:
        assert plugin_version.derive_version(tag) == expected


@pytest.mark.parametrize("tag", [
    "models-onnx-v1",   # the tag that actually shipped a broken metadata.txt
    "",
    "v",
    "latest",
    "v1.0/../etc",
])
def test_derive_version_rejects_non_versions(tag):
    with pytest.raises(ValueError) as exc:
        plugin_version.derive_version(tag)
    assert repr(tag) in str(exc.value)


def test_zip_name_carries_the_version():
    assert (plugin_version.zip_name("0.0.0-demo1")
            == "WINMOL_Analyzer-0.0.0-demo1.zip")
    assert plugin_version.zip_name("0.6.0.2") == "WINMOL_Analyzer-0.6.0.2.zip"


def test_zip_name_rejects_path_separators():
    for bad in ("../evil", "a/b", ""):
        with pytest.raises(ValueError):
            plugin_version.zip_name(bad)


def test_stable_alias_is_unversioned():
    """documentation/index.html links releases/latest/download/<name>.

    That permalink form cannot contain a version, so the stable alias must
    stay byte-for-byte the historical name.
    """
    assert (plugin_version.STABLE_ZIP_NAME
            == "WINMOL_Analyzer_QGIS_Plugin.zip")


def test_cli_prints_the_derived_version():
    out = subprocess.run(
        [sys.executable,
         os.path.join(SCRIPTS_DIR, "plugin_version.py"), "v0.0.0-demo1"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0.0.0-demo1"


def test_cli_fails_on_a_non_version_tag():
    out = subprocess.run(
        [sys.executable,
         os.path.join(SCRIPTS_DIR, "plugin_version.py"), "models-onnx-v1"],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "models-onnx-v1" in out.stderr


# -- the workflow wiring ------------------------------------------------

WORKFLOW = os.path.join(
    REPO_ROOT, ".github", "workflows", "on-push-tags.yml")


def _workflow():
    yaml = pytest.importorskip("yaml")
    with open(WORKFLOW, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_workflow_is_valid_yaml_and_triggers_only_on_v_tags():
    wf = _workflow()
    # PyYAML parses the bare key `on` as the boolean True.
    trigger = wf.get("on", wf.get(True))
    assert trigger["push"]["tags"] == ["v*"], (
        "a '*' trigger builds a plugin zip for non-plugin tags such as "
        "models-onnx-v1")


def test_workflow_uploads_a_versioned_artifact():
    with open(WORKFLOW, encoding="utf-8") as fh:
        text = fh.read()
    assert "plugin_version.py" in text, (
        "the workflow must derive the version through the tested module")
    assert 'artifacts: "WINMOL_Analyzer_QGIS_Plugin.zip"' not in text


def test_workflow_trigger_line_is_v_prefixed():
    """Text-level twin of the parsed check, so it runs without PyYAML."""
    with open(WORKFLOW, encoding="utf-8") as fh:
        lines = [ln.split("#")[0].strip() for ln in fh]
    assert "- 'v*'" in lines
    assert "- '*'" not in lines


# -- end-to-end packaging (no QGIS needed) ------------------------------

BUILD_SCRIPT = os.path.join(REPO_ROOT, "scripts", "build_plugin_zip.sh")

needs_tools = pytest.mark.skipif(
    not (shutil.which("git") and shutil.which("zip")
         and shutil.which("bash")),
    reason="git, zip and bash are required to build the plugin package")


@needs_tools
def test_build_produces_versioned_file_with_unversioned_package_dir(tmp_path):
    """The FILE gains the version; the package DIRECTORY must not.

    QGIS identifies an installed plugin by the top-level directory inside
    the archive. Versioning that directory would install every release as a
    separate plugin and break upgrades.
    """
    version = "0.0.0-demo1"
    out = tmp_path / plugin_version.zip_name(version)
    subprocess.run(
        ["bash", BUILD_SCRIPT, "HEAD", str(out), version],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True)

    assert out.name == "WINMOL_Analyzer-0.0.0-demo1.zip"
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        tops = {n.split("/")[0] for n in names}
        assert tops == {"WINMOL_Analyzer"}
        meta = zf.read("WINMOL_Analyzer/metadata.txt").decode("utf-8")
    lines = [ln for ln in meta.splitlines() if ln.startswith("version=")]
    assert lines == ["version=%s" % version]
