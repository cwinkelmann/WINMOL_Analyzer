"""Guard the contents of the shipped QGIS plugin zip.

`scripts/build_plugin_zip.sh` is the single source of truth for what end
users install, and it only runs in CI on a *tag* push — so a packaging
mistake surfaces at release time, not in PR CI. Nothing else in the suite
exercises it.

These tests pin both halves of the manifest:

* every file the plugin needs at runtime is present, and
* the development-only files listed in the script's EXCLUDE array are gone.

The second half is what makes root-directory cleanups safe: deleting a file
that is dead in Python but still named by the packaging script would abort
every release build, and deleting a file QGIS loads by path would break the
plugin silently. Both now fail here instead.
"""

import os
import shutil
import subprocess
import zipfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_SCRIPT = os.path.join(REPO_ROOT, "scripts", "build_plugin_zip.sh")
PREFIX = "WINMOL_Analyzer/"

# Must be in the zip. Split by the reason it has to ship, because the two
# groups fail differently: a missing QGIS file breaks plugin load, a missing
# core file breaks the subprocess that does the actual work.
REQUIRED_QGIS_FILES = [
    "__init__.py",              # classFactory() — the QGIS entry contract
    "metadata.txt",             # read from the plugin dir root by QGIS
    "icon.png",                 # metadata.txt icon= is plugin-dir relative
    "winmol_analyzer.py",
    "winmol_analyzer_dialog.py",
    "winmol_analyzer_dialog_base.ui",   # loaded via uic at runtime
    "tasks_threads.py",         # Worker / EnvSetupWorker
]

REQUIRED_CORE_FILES = [
    "winmol_run.py",
    "winmol_batch.py",
    "config.json",
]

REQUIRED_CORE_DIRS = [
    "classes/",
    "utils/",
    "plugin_utils/",
    "requirements/",
]

# Development-only paths that must never reach an end user.
FORBIDDEN_PREFIXES = [
    ".github/",
    "docker/",
    "docs/",
    "documentation/",
    "scripts/",
    "standalone/",
    "tests/",
    "benchmark/",
]

FORBIDDEN_FILES = [
    "CLAUDE.md",
    "Dockerfile",
    "Makefile",
    "setup.cfg",
    "startDocker.sh",
    ".gitignore",
]


def _build(out_path):
    if shutil.which("bash") is None or shutil.which("zip") is None:
        pytest.skip("bash and zip are required to build the plugin package")
    if not os.path.exists(os.path.join(REPO_ROOT, ".git")):
        pytest.skip("not a git checkout — build_plugin_zip.sh needs git")
    proc = subprocess.run(
        ["bash", BUILD_SCRIPT, "HEAD", out_path],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"build_plugin_zip.sh failed:\n{proc.stdout}\n{proc.stderr}")
    return out_path


@pytest.fixture(scope="module")
def package_names(tmp_path_factory):
    """The zip's entry names, with the plugin-dir prefix stripped."""
    out = str(tmp_path_factory.mktemp("pkg") / "WINMOL_Analyzer.zip")
    _build(out)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert names, "package is empty"
    for name in names:
        assert name.startswith(PREFIX), (
            f"{name} is outside the {PREFIX} plugin directory")
    return sorted(n[len(PREFIX):] for n in names)


@pytest.mark.parametrize("name", REQUIRED_QGIS_FILES + REQUIRED_CORE_FILES)
def test_required_file_ships(package_names, name):
    assert name in package_names, f"{name} missing from the plugin package"


@pytest.mark.parametrize("directory", REQUIRED_CORE_DIRS)
def test_required_directory_ships(package_names, directory):
    assert any(n.startswith(directory) for n in package_names), (
        f"{directory} missing from the plugin package")


@pytest.mark.parametrize("name", FORBIDDEN_FILES)
def test_development_file_is_stripped(package_names, name):
    assert name not in package_names, f"{name} must not ship to end users"


@pytest.mark.parametrize("prefix", FORBIDDEN_PREFIXES)
def test_development_directory_is_stripped(package_names, prefix):
    leaked = [n for n in package_names if n.startswith(prefix)]
    assert not leaked, f"{prefix} must not ship to end users: {leaked}"


def test_no_pycache_or_ds_store(package_names):
    junk = [n for n in package_names
            if "__pycache__" in n or n.endswith(".DS_Store")]
    assert not junk, f"build artefacts leaked into the package: {junk}"


def test_script_requires_its_own_manifest():
    """The script's own FATAL list must not name a file we deleted.

    build_plugin_zip.sh hard-aborts if any entry of its `required` loop is
    absent. Keeping that list and the repository in sync is the failure mode
    that only shows up on a tag push, so check it directly.
    """
    with open(BUILD_SCRIPT, encoding="utf-8") as fh:
        body = fh.read()
    marker = "for required in "
    start = body.index(marker) + len(marker)
    listing = body[start:body.index("; do", start)]
    entries = listing.replace("\\\n", " ").split()
    assert entries, "could not parse the required-file list"
    for entry in entries:
        assert os.path.exists(os.path.join(REPO_ROOT, entry)), (
            f"build_plugin_zip.sh requires {entry}, which no longer exists "
            f"in the repository — every tagged release build would abort")


#: The Setup tab's Qt-free decision layer. It lands inside the already
#: required plugin_utils/ directory, so this should pass unchanged — but
#: a plugin that ships the dialog without them is a dialog that raises on
#: import in QGIS, which no other test here would catch.
SETUP_TAB_MODULES = [
    "plugin_utils/setup_state.py",
    "plugin_utils/model_status.py",
    "plugin_utils/model_registry.py",
    "plugin_utils/installer.py",
    "tasks_threads.py",
]


@pytest.mark.parametrize("name", SETUP_TAB_MODULES)
def test_setup_tab_modules_ship(package_names, name):
    assert name in package_names, f"{name} missing from the plugin package"
