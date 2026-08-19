"""The shipped requirements must cover what the runtime actually imports.

Written after 0.7.0-reimpl14 shipped a plugin that died at model load with
``ModuleNotFoundError: No module named 'onnx'``. Two independent defects
had to line up for that to reach a user, and both are pinned here:

1. Making ``prediction_read_strategy=graph`` the default (6ff1da2) put
   ``utils/onnx_preprocess.py`` -- and its top-level ``import onnx`` -- on
   the default code path, but no requirements file named ``onnx``.
2. The install sentinel hashed only ``cpu.txt``, which is little more than
   ``-r core.txt``. Adding the dependency to core.txt would not have
   changed the hash, so no existing install would have rebuilt its venv.

CI cannot catch (1) on its own: tests.yml installs an ad-hoc package list
that happens to include onnx, so a drifted requirements file looks fine.
"""
import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNTIME_DIRS = ("utils", "classes", "plugin_utils")

#: import name -> the distributions that can provide it. More than one
#: where a module has competing providers: onnxruntime and
#: onnxruntime-gpu both ship the `onnxruntime` module, which is why
#: cpu.txt and gpu.txt must never be installed together (see gpu.txt).
DIST_OF_MODULE = {
    "skimage": {"scikit-image"},
    "PIL": {"Pillow"},
    "cv2": {"opencv-python"},
    "onnxruntime": {"onnxruntime", "onnxruntime-gpu"},
}

#: Provided by the QGIS host process, never pip-installed by us.
HOST_PROVIDED = {"qgis", "PyQt5", "osgeo", "processing"}

#: Not declared directly, but guaranteed by a declared package. Keep this
#: list short and justified -- every entry is an assumption that can break
#: when the provider changes (geopandas 1.0 made fiona optional, which is
#: why the geopandas pin in core.txt is load-bearing).
TRANSITIVE = {
    "fiona": "geopandas",
    "pandas": "geopandas",
    "pyproj": "geopandas",
}


def _requirements_closure(path):
    """``path`` plus everything it pulls in with ``-r``."""
    seen, out, queue = set(), [], [pathlib.Path(path)]
    while queue:
        current = queue.pop(0)
        if current in seen or not current.is_file():
            continue
        seen.add(current)
        out.append(current)
        for line in current.read_text().splitlines():
            line = line.strip()
            if line.startswith("-r "):
                queue.append(current.parent / line[3:].strip())
    return out


def _declared_distributions(entry):
    names = set()
    for part in _requirements_closure(entry):
        for line in part.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            name = line.split("[")[0]
            for sep in ("==", ">=", "<=", "~=", ">", "<", "!="):
                name = name.split(sep)[0]
            names.add(name.strip().lower())
    return names


def _top_level_third_party_imports():
    """Modules imported at module scope by the runtime packages.

    Module scope only: a lazy import inside a function (matplotlib in
    IO.py) fails at the point of use, not at plugin load, and is not part
    of the base install contract.
    """
    stdlib = set(__import__("sys").stdlib_module_names)
    local = {d for d in RUNTIME_DIRS} | {
        "conftest", "winmol_run", "winmol_batch", "resources",
        "tasks_threads", "qgisutil",
    }
    found = {}
    for directory in RUNTIME_DIRS:
        for path in sorted((REPO / directory).glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [(node.module or "").split(".")[0]]
                for module in modules:
                    if module and module not in stdlib and module not in local:
                        found.setdefault(module, path.relative_to(REPO))
    return found


@pytest.mark.parametrize("entry", ["cpu.txt", "gpu.txt"])
def test_runtime_imports_are_installable(entry):
    """Every module the runtime imports at load time must be installed by
    the requirements file the plugin builds its venv from."""
    declared = _declared_distributions(REPO / "requirements" / entry)
    missing = []
    for module, source in sorted(_top_level_third_party_imports().items()):
        if module in HOST_PROVIDED or module in TRANSITIVE:
            continue
        providers = DIST_OF_MODULE.get(module, {module})
        if not any(p.lower() in declared for p in providers):
            missing.append(
                f"{module} (imported by {source}) -> "
                f"{' or '.join(sorted(providers))}")
    assert not missing, (
        f"{entry} does not install: " + "; ".join(missing) +
        "\nDeclare it in requirements/core.txt, or add it to TRANSITIVE "
        "with the package that guarantees it.")


def test_onnx_is_declared():
    """The regression itself: the graph read strategy is the default, so
    the onnx authoring library is a base dependency, not an extra."""
    for entry in ("cpu.txt", "gpu.txt"):
        declared = _declared_distributions(REPO / "requirements" / entry)
        assert "onnx" in declared, f"onnx missing from {entry}"


def test_sentinel_hash_covers_included_files(tmp_path):
    """A change to core.txt must invalidate the install sentinel.

    Without this the fix above could never reach an existing install: the
    venv is rebuilt only when the recorded hash stops matching.
    """
    from plugin_utils.installer import _file_hash

    core = tmp_path / "core.txt"
    entry = tmp_path / "cpu.txt"
    core.write_text("numpy==1.26.4\n")
    entry.write_text("-r core.txt\nonnxruntime>=1.17\n")

    before = _file_hash(entry)
    core.write_text("numpy==1.26.4\nonnx>=1.15\n")
    after = _file_hash(entry)

    assert before != after, (
        "editing an -r included file left the sentinel hash unchanged; "
        "installs would keep a stale venv")
