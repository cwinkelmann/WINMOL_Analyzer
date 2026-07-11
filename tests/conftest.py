import json
import os
import sys

# ---------------------------------------------------------------------------
# Determinism guard (MUST run before anything hashes strings into sets).
#
# The pipeline's output depends on set-iteration order because Part.__hash__
# hashes a tuple containing strings (docs/CODE_REVIEW_2.md A-4). String hashes
# are salted per process unless PYTHONHASHSEED is fixed, so golden-master
# comparisons are only meaningful under one specific seed. The fixtures are
# generated with PYTHONHASHSEED=0; enforce the same here by re-executing
# pytest when the seed differs. Children spawned by multiprocessing inherit
# the environment, so pool workers are covered too.
# ---------------------------------------------------------------------------
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    # pytest's fd-capture is already active while conftest loads, so the
    # re-exec'd child would write into the (now orphaned) capture pipe and
    # appear silent. Re-attach stdio to the terminal first; without a tty
    # (CI, pipes) the run stays silent but the exit code is still correct.
    try:
        tty = os.open("/dev/tty", os.O_WRONLY)
        os.dup2(tty, 1)
        os.dup2(tty, 2)
    except OSError:
        pass
    os.execv(sys.executable, [sys.executable, "-m", "pytest"] + sys.argv[1:])

# The pretrained WINMOL models are Keras 2 HDF5 artifacts and fail to load under
# the Keras 3 bundled with TensorFlow >= 2.16. Route tf.keras to the legacy
# Keras 2 shim (tf-keras). This runs at collection time, before any test module
# imports TensorFlow.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import pytest  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO_ROOT, TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fixtures")


def _fixture(name):
    path = os.path.join(FIXTURES_DIR, name)
    if not os.path.exists(path):
        pytest.skip(
            f"fixture {name} missing — run "
            f"PYTHONHASHSEED=0 python tests/generate_fixtures.py")
    return path


@pytest.fixture(scope="session")
def fixtures_dir():
    if not os.path.isdir(FIXTURES_DIR):
        pytest.skip("tests/fixtures/ missing — run generate_fixtures.py")
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def pipeline_config():
    """Config built from the snapshot the fixtures were generated with.

    Never use bare Config() in golden tests: a future default change would
    silently invalidate the comparison instead of failing loudly here.
    """
    from classes.Config import Config
    path = _fixture("config_snapshot.json")
    with open(path) as f:
        snapshot = json.load(f)
    config = Config()
    for key, value in snapshot.items():
        setattr(config, key, value)
    return config


@pytest.fixture(scope="session")
def stem_map():
    """(pred array uint8, profile dict) from the golden stem-map raster.

    This is the resampled prediction grid — its transform is the one
    predict_with_resampling_per_tile produced, which all downstream stages
    consume.
    """
    import rasterio
    path = _fixture("stem_map.tif")
    with rasterio.open(path) as src:
        pred = src.read(1)
        profile = dict(src.profile)
    return pred, profile


@pytest.fixture(scope="session")
def crop_input_path():
    return _fixture("crop_input.tif")


@pytest.fixture(scope="session")
def golden(fixtures_dir):
    """Loader for stage fixtures: golden('stage_find_segments')."""
    from helpers import read_json_gz

    def _load(stage_name):
        return read_json_gz(_fixture(stage_name + ".json.gz"))
    return _load
