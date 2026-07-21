"""Guards against the class of bug that shipped in v0.0.0-demo3: every
model URL in config.json 404'd because the release assets had been
renamed after the registry was written, and nothing checked.

Two layers:

* OFFLINE (always runs, no socket) — internal consistency. Every entry's
  ``file`` must equal the basename of its ``url``, and every sha256 in
  config.json must appear in the vendored ground truth
  ``tests/fixtures/models_v1_SHA256SUMS`` under exactly that filename.
  A rename in the release therefore cannot be half-applied here.

* NETWORK (opt-in) — actual anonymous reachability of every registry
  URL. Skipped unless ``WINMOL_NET_TESTS=1`` so CI and normal runs stay
  entirely offline. Run it before cutting a release::

      WINMOL_NET_TESTS=1 pytest tests/test_model_urls.py -q
"""

import os
import sys
import urllib.error
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from plugin_utils import model_registry as mr   # noqa: E402
from tests.test_model_registry import (         # noqa: E402
    SHIPPED_CONFIG, ZOO_SHA256, ZOO_SUMS_FIXTURE)

#: Opt-in switch for the tests that actually open a socket.
NET_ENV = "WINMOL_NET_TESTS"

net = pytest.mark.skipif(
    os.environ.get(NET_ENV, "") not in ("1", "true", "yes"),
    reason=f"network test; set {NET_ENV}=1 to enable")


def _entries():
    return list(mr.load_registry(SHIPPED_CONFIG).entries.values())


def _basename(url):
    """URL basename with any query string stripped — the filename the
    downloader will land on disk."""
    return os.path.basename(url.split("?", 1)[0].split("#", 1)[0])


# --- offline: internal consistency -------------------------------------------

def test_file_equals_url_basename():
    """The registry invariant: 'file' is the URL basename, so a model
    fetched by hand (gh release download / browser) is recognised."""
    bad = [(e.id, e.file, _basename(e.url))
           for e in _entries() if e.file != _basename(e.url)]
    assert bad == [], (
        "file != basename(url) for: "
        + "; ".join(f"{i}: {f!r} vs {b!r}" for i, f, b in bad))


def test_zoo_digests_match_vendored_sha256sums():
    """Every models-v1 asset referenced by config.json must exist in the
    vendored SHA256SUMS under that exact filename, with that exact
    digest. This is what fails when the release renames its assets."""
    assert len(ZOO_SHA256) == 22, ZOO_SUMS_FIXTURE
    zoo = [e for e in _entries() if "WINMOL_segmentor_pt" in e.url]
    assert len(zoo) == 22
    unknown = [(e.id, e.file) for e in zoo if e.file not in ZOO_SHA256]
    assert unknown == [], (
        "filename(s) absent from " + ZOO_SUMS_FIXTURE + ": "
        + "; ".join(f"{i} -> {f}" for i, f in unknown))
    for e in zoo:
        assert e.sha256 == ZOO_SHA256[e.file], e.id
    # ...and every vendored asset is actually offered by the registry.
    assert sorted(e.file for e in zoo) == sorted(ZOO_SHA256)


def test_every_zoo_url_targets_the_same_release():
    prefix = ("https://github.com/cwinkelmann/WINMOL_segmentor_pt"
              "/releases/download/models-v1/")
    for e in _entries():
        if "WINMOL_segmentor_pt" not in e.url:
            continue
        assert e.url == prefix + e.file, e.id


# --- opt-in: real reachability -----------------------------------------------

def _head_status(url, timeout=30):
    """Status code for `url`, following redirects. Falls back to a
    1-byte ranged GET for hosts that reject HEAD (Zenodo does)."""
    def _try(method, headers=None):
        req = urllib.request.Request(
            url, method=method, headers=headers or {})
        req.add_header("User-Agent", "winmol-registry-check/1")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    try:
        return _try("HEAD")
    except urllib.error.HTTPError as exc:
        if exc.code not in (403, 405, 501):
            return exc.code
    except urllib.error.URLError:
        pass
    try:
        return _try("GET", {"Range": "bytes=0-0"})
    except urllib.error.HTTPError as exc:
        return exc.code


@net
@pytest.mark.parametrize(
    "entry", _entries(), ids=lambda e: e.id)
def test_registry_url_is_anonymously_reachable(entry):
    """Every advertised model must download for a user with no
    credentials. A private repo or a renamed asset fails here instead
    of in the QGIS dialog."""
    status = _head_status(entry.url)
    assert status in (200, 206), (
        f"{entry.id}: HTTP {status} for {entry.url}")
