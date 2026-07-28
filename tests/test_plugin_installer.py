"""Off-QGIS tests for plugin_utils.installer (concise ONNX installer).

The module must import and resolve without QGIS/PyQt/pkg_resources, the
sentinel must key on the current requirements hash, and the requirements
algebra is: core.txt names no runtime, cpu.txt = core + one runtime.
"""
import ast
import importlib
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

installer = importlib.import_module("plugin_utils.installer")


def _requirement_names(text):
    names = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.append(re.split(r"[\s<>=!~;\[]", line, 1)[0].lower())
    return names


def test_imports_without_qgis_pyqt_or_pkg_resources(monkeypatch):
    for mod in ("qgis", "qgis.core", "qgis.PyQt", "PyQt5",
                "pkg_resources"):
        monkeypatch.setitem(sys.modules, mod, None)
    monkeypatch.delitem(sys.modules, "plugin_utils.installer",
                        raising=False)
    fresh = importlib.import_module("plugin_utils.installer")
    assert hasattr(fresh, "resolve_environment")


def test_requirements_file_is_cpu_txt():
    path = installer.plugin_requirements_path()
    assert path.name == "cpu.txt"
    assert path.parent.name == "requirements"


def test_venv_python_path_per_platform(monkeypatch):
    monkeypatch.setattr(installer.sys, "platform", "win32")
    win = installer.get_venv_python_path("venvdir")
    assert win == os.path.join("venvdir", "Scripts", "python.exe")
    monkeypatch.setattr(installer.sys, "platform", "linux")
    nix = installer.get_venv_python_path("venvdir")
    assert nix.startswith(os.path.join("venvdir", "bin", "python"))


def test_marker_roundtrip_and_hash_invalidation(tmp_path, monkeypatch):
    req = tmp_path / "cpu.txt"
    req.write_text("-r core.txt\nonnxruntime>=1.17\n")
    monkeypatch.setattr(installer, "plugin_requirements_path",
                        lambda: req)
    venv = tmp_path / "venv"
    venv.mkdir()
    assert not installer.marker_matches(str(venv))
    assert not installer.is_ready(str(venv))
    installer._write_marker(str(venv))
    assert installer.marker_matches(str(venv))
    # marker alone is not readiness: the venv has no python
    assert not installer.is_ready(str(venv))
    req.write_text("-r core.txt\nonnxruntime>=1.18\n")
    assert not installer.marker_matches(str(venv))


def test_core_txt_names_no_runtime():
    names = _requirement_names(
        (REPO / "requirements" / "core.txt").read_text())
    assert names, "core.txt must list the geo/science stack"
    assert not [n for n in names
                if "onnxruntime" in n or "tensorflow" in n]


def test_cpu_txt_is_core_plus_one_runtime_and_psutil():
    text = (REPO / "requirements" / "cpu.txt").read_text()
    includes = [line.strip() for line in text.splitlines()
                if line.strip().startswith("-r")]
    assert includes == ["-r core.txt"]
    names = _requirement_names(text)
    assert names.count("onnxruntime") == 1
    assert "onnxruntime-gpu" not in names
    assert "psutil" in names


def test_stale_managed_pointer_falls_through_to_needs_setup(
        tmp_path, monkeypatch):
    plugin_dir = str(tmp_path)
    venv = installer.venv_location(plugin_dir)
    ghost = os.path.join(venv, "bin", "python")
    assert not os.path.exists(ghost)
    monkeypatch.setattr(installer, "configured_python_executable",
                        lambda: ghost)
    result = installer.resolve_environment(plugin_dir, build=False)
    assert result["status"] == "needs_setup"
    assert result["python"] is None
    assert result["venv_path"] == venv
    assert set(result) >= {"status", "python", "venv_path", "message"}


# Files that run before/without the "does compute deps import" probe
# (winmol_run.py -> utils.Prediction/PredictWorkers -> utils.IO), so a
# module-level import missing from requirements crashes the first real
# run even though the venv built successfully.
_CLOSURE_FILES = (
    "winmol_run.py",
    "utils/IO.py",
    "utils/Prediction.py",
    "utils/PredictWorkers.py",
    "utils/onnx_runtime.py",
)
_LOCAL_PACKAGES = {"utils", "classes", "plugin_utils"}
# Import roots not literally named in requirements/*.txt because they
# ride in as transitive deps of a package that IS: geopandas pulls in
# fiona/pandas/pyproj, and scikit-image is imported as "skimage".
_IMPORT_TO_REQUIREMENT_NAME = {
    "skimage": "scikit-image",
    "fiona": "geopandas",
    "pandas": "geopandas",
    "pyproj": "geopandas",
}


def _module_level_import_roots(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_module_level_imports_are_covered_by_requirements():
    """A module-level import (e.g. matplotlib) with no path into
    requirements/core.txt + cpu.txt builds a fine venv and then crashes
    on the first real run. Anything not needed at import time belongs
    inside the one function that uses it, not at module scope."""
    declared = set(_requirement_names(
        (REPO / "requirements" / "core.txt").read_text())
        + _requirement_names(
            (REPO / "requirements" / "cpu.txt").read_text()))
    stdlib = set(sys.stdlib_module_names)
    missing = []
    for rel in _CLOSURE_FILES:
        for root in _module_level_import_roots(REPO / rel):
            if root in stdlib or root in _LOCAL_PACKAGES:
                continue
            req_name = _IMPORT_TO_REQUIREMENT_NAME.get(root, root)
            if req_name.lower() not in declared:
                missing.append("%s: %r" % (rel, root))
    assert not missing, (
        "module-level imports uncovered by requirements/core.txt + "
        "cpu.txt (lazy-import inside the one function that needs it "
        "instead): " + ", ".join(missing))
