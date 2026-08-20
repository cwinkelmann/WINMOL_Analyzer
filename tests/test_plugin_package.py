"""Smoke test for scripts/build_plugin_zip.sh: the self-contained plugin
ZIP that "Install from ZIP" needs. Runs the real script against a real git
archive of HEAD from a clean tmp cwd, then inspects the resulting zip.
"""
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_plugin_zip.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("zip") is None,
    reason="build_plugin_zip.sh needs git and zip on PATH",
)

REQUIRED = [
    "WINMOL_Analyzer/__init__.py",
    "WINMOL_Analyzer/metadata.txt",
    "WINMOL_Analyzer/config.json",
    "WINMOL_Analyzer/winmol_run.py",
    "WINMOL_Analyzer/winmol_batch.py",
    "WINMOL_Analyzer/winmol_analyzer.py",
    "WINMOL_Analyzer/winmol_analyzer_dialog.py",
    "WINMOL_Analyzer/winmol_analyzer_dialog_base.ui",
    "WINMOL_Analyzer/tasks_threads.py",
    "WINMOL_Analyzer/resources.py",
    "WINMOL_Analyzer/icon.png",
    "WINMOL_Analyzer/requirements/cpu.txt",
    "WINMOL_Analyzer/requirements/gpu.txt",
    "WINMOL_Analyzer/plugin_utils/model_registry.py",
    "WINMOL_Analyzer/plugin_utils/childenv.py",
    "WINMOL_Analyzer/plugin_utils/gpu_probe.py",
    # the plugin-gui feature's helper modules — the dialog imports them
    "WINMOL_Analyzer/plugin_utils/run_progress.py",
    "WINMOL_Analyzer/plugin_utils/setup_state.py",
    "WINMOL_Analyzer/plugin_utils/model_status.py",
    "WINMOL_Analyzer/plugin_utils/output_selection.py",
    "WINMOL_Analyzer/plugin_utils/config_overrides.py",
    # rc11-parity modules
    "WINMOL_Analyzer/plugin_utils/py311.py",
    "WINMOL_Analyzer/plugin_utils/autotune_cache.py",
]

EXCLUDED_DIR_PREFIXES = [
    "WINMOL_Analyzer/tests/",
    "WINMOL_Analyzer/docs/",
    "WINMOL_Analyzer/documentation/",
    "WINMOL_Analyzer/.github/",
    "WINMOL_Analyzer/scripts/",
    "WINMOL_Analyzer/standalone/",
]


def _build(tmp_path, out_name="out.zip", version=None):
    """Run the script from a clean tmp cwd and return the produced zip path."""
    out = tmp_path / out_name
    args = [str(SCRIPT), "HEAD", str(out)]
    if version is not None:
        args.append(version)
    subprocess.run(
        args, cwd=tmp_path, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert out.is_file()
    return out


def test_zip_has_expected_layout(tmp_path):
    out = _build(tmp_path)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()

    assert any(n == "WINMOL_Analyzer/" for n in names), (
        "inner dir must be exactly WINMOL_Analyzer/ so QGIS keys the "
        "installed plugin on it"
    )
    for req in REQUIRED:
        assert req in names, f"missing required file: {req}"
    for prefix in EXCLUDED_DIR_PREFIXES:
        assert not any(n.startswith(prefix) for n in names), (
            f"dev-only path shipped: {prefix}"
        )


def test_version_override_writes_metadata(tmp_path):
    out = _build(tmp_path, version="0.7.0-reimpl1")
    with zipfile.ZipFile(out) as zf:
        metadata = zf.read("WINMOL_Analyzer/metadata.txt").decode()

    versions = [line for line in metadata.splitlines()
                if line.startswith("version=")]
    assert versions == ["version=0.7.0-reimpl1"]


def test_missing_required_file_is_fatal(tmp_path, monkeypatch):
    # Point the script at a ref that can't contain winmol_run.py: an
    # orphan-ish empty tree object is not available, so instead exercise
    # the FATAL branch directly by archiving a subtree that lacks it.
    result = subprocess.run(
        [str(SCRIPT), "HEAD:tests", str(tmp_path / "out.zip")],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0
    assert "FATAL" in result.stderr
