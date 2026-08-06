"""Off-QGIS tests for plugin_utils.py311 — the Python 3.11 auto-download —
and its wiring into plugin_utils.installer (managed_base_python /
choose_base_python).

Pure functions (asset selection, digest table, path layout) run for real.
Anything that would touch the network (``_download``, and therefore
``urllib.request.urlopen``) is replaced with a fake. NO test in this file
makes a network call.
"""
import hashlib
import importlib
import io
import os
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

py311 = importlib.import_module("plugin_utils.py311")
installer = importlib.import_module("plugin_utils.installer")


# --- pinned release -----------------------------------------------------

def test_pinned_release_matches_rc11():
    """Regression guard: rc11 field-tested this exact release. A bump
    here must be a deliberate, evidenced change, not an accident."""
    assert py311.PBS_TAG == "20260623"
    assert py311.PBS_PYVER == "3.11.15"


def test_release_base_url_is_the_official_pbs_repo():
    assert py311._RELEASE_BASE == (
        "https://github.com/astral-sh/python-build-standalone/releases/"
        "download/20260623/")


# --- asset name / url ----------------------------------------------------

def test_asset_name_embeds_version_tag_and_triple():
    name = py311.asset_name("x86_64-unknown-linux-gnu")
    assert name == (
        "cpython-3.11.15+20260623-x86_64-unknown-linux-gnu-"
        "install_only.tar.gz")


def test_asset_url_is_release_base_plus_asset_name():
    triple = "aarch64-apple-darwin"
    assert py311.asset_url(triple) == (
        py311._RELEASE_BASE + py311.asset_name(triple))


# --- per-(platform, machine) target-triple selection ---------------------

_TRIPLE_MATRIX = [
    ("Darwin", "arm64", "aarch64-apple-darwin"),
    ("Darwin", "aarch64", "aarch64-apple-darwin"),
    ("Darwin", "x86_64", "x86_64-apple-darwin"),
    ("Windows", "AMD64", "x86_64-pc-windows-msvc"),
    ("Windows", "x86_64", "x86_64-pc-windows-msvc"),
    ("Windows", "ARM64", "aarch64-pc-windows-msvc"),
    ("Linux", "x86_64", "x86_64-unknown-linux-gnu"),
    ("Linux", "aarch64", "aarch64-unknown-linux-gnu"),
    ("Linux", "arm64", "aarch64-unknown-linux-gnu"),
]


@pytest.mark.parametrize("system,machine,expected", _TRIPLE_MATRIX)
def test_target_triple_matrix(monkeypatch, system, machine, expected):
    monkeypatch.setattr(py311.platform, "system", lambda: system)
    monkeypatch.setattr(py311.platform, "machine", lambda: machine)
    assert py311.target_triple() == expected


@pytest.mark.parametrize("system,machine", [
    ("FreeBSD", "x86_64"),
    ("Darwin", "i386"),
    ("Windows", "i686"),
    ("Linux", "i386"),
    ("SunOS", "x86_64"),
])
def test_target_triple_unsupported_combos_return_none(
        monkeypatch, system, machine):
    monkeypatch.setattr(py311.platform, "system", lambda: system)
    monkeypatch.setattr(py311.platform, "machine", lambda: machine)
    assert py311.target_triple() is None


def test_binding_requirement_four_platforms_are_covered(monkeypatch):
    """The plan's BINDING REQUIREMENT names these four explicitly:
    win-x86_64, mac-arm64, mac-x86_64, linux-x86_64. (RR6 ships two more
    — win/linux arm64 — which the matrix above already exercises.)"""
    combos = {
        ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
        ("Darwin", "arm64"): "aarch64-apple-darwin",
        ("Darwin", "x86_64"): "x86_64-apple-darwin",
        ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
    }
    for (system, machine), expected in combos.items():
        monkeypatch.setattr(py311.platform, "system", lambda s=system: s)
        monkeypatch.setattr(
            py311.platform, "machine", lambda m=machine: m)
        assert py311.target_triple() == expected
        assert expected in py311._DIGESTS
        assert py311.asset_url(expected).startswith(py311._RELEASE_BASE)


# --- digest table ---------------------------------------------------------

def test_digests_cover_all_six_pbs_triples():
    assert set(py311._DIGESTS) == {
        "aarch64-apple-darwin", "x86_64-apple-darwin",
        "x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu",
    }


@pytest.mark.parametrize("triple", sorted(py311._DIGESTS))
def test_digest_is_a_valid_lowercase_sha256_hex_string(triple):
    digest = py311._DIGESTS[triple]
    assert len(digest) == 64
    assert digest == digest.lower()
    int(digest, 16)  # raises ValueError if it is not hex


# --- extraction layout / interpreter path ---------------------------------

def test_interp_path_windows(monkeypatch):
    monkeypatch.setattr(py311.sys, "platform", "win32")
    assert py311._interp_path("root") == os.path.join(
        "root", "python.exe")


def test_interp_path_posix_prefers_python3_11(monkeypatch, tmp_path):
    monkeypatch.setattr(py311.sys, "platform", "darwin")
    root = tmp_path / "root"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python3.11").write_text("x")
    assert py311._interp_path(str(root)) == str(
        root / "bin" / "python3.11")


def test_interp_path_posix_falls_back_to_python3(monkeypatch, tmp_path):
    monkeypatch.setattr(py311.sys, "platform", "linux")
    root = tmp_path / "root"
    (root / "bin").mkdir(parents=True)  # no python3.11 present
    assert py311._interp_path(str(root)) == str(root / "bin" / "python3")


def test_existing_interpreter_none_when_absent(tmp_path):
    assert py311.existing_interpreter(str(tmp_path)) is None


def test_existing_interpreter_found(tmp_path, monkeypatch):
    monkeypatch.setattr(py311.sys, "platform", "linux")
    exe_dir = tmp_path / "python" / "bin"
    exe_dir.mkdir(parents=True)
    exe = exe_dir / "python3.11"
    exe.write_text("x")
    assert py311.existing_interpreter(str(tmp_path)) == str(exe)


# --- checksum --------------------------------------------------------------

def test_sha256_matches_hashlib_on_the_same_bytes(tmp_path):
    data = b"the quick brown fox jumps over the lazy dog" * 100
    f = tmp_path / "blob.bin"
    f.write_bytes(data)
    assert py311._sha256(str(f)) == hashlib.sha256(data).hexdigest()


# --- safe extraction --------------------------------------------------------

def _make_tar_gz(path, members):
    """Write a real gzip tarball at ``path`` with regular files
    ``{arcname: bytes}`` — a stand-in for a genuine PBS release asset."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))


def test_safe_extract_writes_real_members(tmp_path):
    archive = tmp_path / "a.tar.gz"
    _make_tar_gz(str(archive), {"python/bin/python3.11": b"stub"})
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(str(archive), "r:gz") as tar:
        py311._safe_extract(tar, str(dest))
    assert (dest / "python" / "bin" / "python3.11").read_bytes() == b"stub"


def _make_tar_gz_with_symlink(path, regular, symlinks):
    """Tarball with regular files ``{name: bytes}`` plus symlinks
    ``{name: linkname}`` — mirrors the terminfo symlinks in a real PBS asset."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data in regular.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        for name, linkname in symlinks.items():
            info = tarfile.TarInfo(name=name)
            info.type = tarfile.SYMTYPE
            info.linkname = linkname
            tar.addfile(info)


def test_safe_extract_allows_escaping_symlinks_data_would_reject(tmp_path):
    """PBS ships terminfo symlinks whose targets the strict 'data' filter
    rejects as "outside the destination", aborting the whole extract (issue
    #25). _safe_extract uses the 'tar' filter, which still blocks name
    traversal but permits these links — reverting to 'data' makes this raise."""
    archive = tmp_path / "pbs.tar.gz"
    _make_tar_gz_with_symlink(
        str(archive),
        {"python/bin/python3.11": b"stub"},
        {"python/share/terminfo/1/1178": "../../../../../a/adm1178"},
    )
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(str(archive), "r:gz") as tar:
        py311._safe_extract(tar, str(dest))
    assert (dest / "python" / "bin" / "python3.11").read_bytes() == b"stub"
    assert (
        dest / "python" / "share" / "terminfo" / "1" / "1178"
    ).is_symlink()


class _FakeMember:
    def __init__(self, name):
        self.name = name


class _FakeTarNoFilterSupport:
    """A tarfile.TarFile stand-in for a Python old enough that
    extractall() has no ``filter`` kwarg (raises TypeError), forcing
    _safe_extract's manual traversal-check fallback — the branch that
    is otherwise unreachable on a modern interpreter (which blocks
    traversal via the 'data' filter itself, with a different exception
    type)."""

    def __init__(self, members, calls):
        self._members = members
        self._calls = calls

    def extractall(self, path, filter=None):
        if filter is not None:
            raise TypeError(
                "extractall() got an unexpected keyword argument "
                "'filter'")
        self._calls.append(path)

    def getmembers(self):
        return self._members


def test_safe_extract_manual_fallback_rejects_path_traversal():
    calls = []
    fake = _FakeTarNoFilterSupport([_FakeMember("../evil.txt")], calls)
    with pytest.raises(RuntimeError, match="unsafe path"):
        py311._safe_extract(fake, "/some/dest")
    assert calls == []  # never fell through to the real extractall


def test_safe_extract_manual_fallback_accepts_safe_members(tmp_path):
    calls = []
    fake = _FakeTarNoFilterSupport(
        [_FakeMember("python/bin/python3.11")], calls)
    py311._safe_extract(fake, str(tmp_path))
    assert calls == [str(tmp_path)]


# --- _verify_runs -----------------------------------------------------------

def test_verify_runs_true_for_the_current_interpreter():
    # The repo's test convention runs the whole suite under Python 3.11
    # (installer.MIN_PY == MAX_PY == (3, 11)), so the interpreter running
    # THIS test is a legitimate, network-free stand-in for a freshly
    # downloaded one.
    assert py311._verify_runs(sys.executable) is True


def test_verify_runs_false_for_a_bogus_path():
    assert py311._verify_runs("/no/such/interpreter-xyz") is False


def test_verify_runs_false_when_reported_version_does_not_match(
        monkeypatch):
    class _Result:
        returncode = 0
        stdout = "3.9"
    # _verify_runs probes through childenv.run_isolated.
    childenv = importlib.import_module("plugin_utils.childenv")
    monkeypatch.setattr(
        childenv.subprocess, "run", lambda *a, **k: _Result())
    assert py311._verify_runs("whatever") is False


# --- _download (network calls faked at urlopen) -----------------------------

class _FakeResponse:
    def __init__(self, data, content_length=None):
        self._buf = io.BytesIO(data)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def read(self, n):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_writes_the_full_response_body(tmp_path, monkeypatch):
    data = b"x" * 4096
    monkeypatch.setattr(
        py311.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse(
            data, content_length=len(data)))
    dest = tmp_path / "out.bin"
    py311._download("https://example.invalid/f", str(dest))
    assert dest.read_bytes() == data


def test_download_reports_progress_with_known_total(tmp_path, monkeypatch):
    data = b"y" * 4096
    monkeypatch.setattr(
        py311.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse(
            data, content_length=len(data)))
    messages = []
    py311._download("https://example.invalid/f", str(tmp_path / "o"),
                    progress=messages.append)
    assert any("Downloading Python 3.11" in m for m in messages)
    assert any("/" in m for m in messages)  # "N/Total MB" form


def test_download_reports_progress_without_known_total(
        tmp_path, monkeypatch):
    data = b"z" * 4096
    monkeypatch.setattr(
        py311.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse(data))
    messages = []
    py311._download("https://example.invalid/f", str(tmp_path / "o"),
                    progress=messages.append)
    assert messages
    assert all("/" not in m for m in messages)  # total unknown


# --- ensure_python311 orchestration (fake fetcher, NO network) -------------

def _install_fake_download(monkeypatch, members):
    """Replace py311._download with one that writes a real tarball built
    from ``members`` instead of touching the network. Returns the list
    of dest paths it was called with (len(...) == call count)."""
    calls = []

    def fake_download(url, dest, progress=None):
        calls.append(dest)
        _make_tar_gz(dest, members)
        if progress:
            progress("Downloading Python 3.11… 1/1 MB")
    monkeypatch.setattr(py311, "_download", fake_download)
    return calls


def test_existing_verified_interpreter_skips_download(
        tmp_path, monkeypatch):
    exe_dir = tmp_path / "python" / "bin"
    exe_dir.mkdir(parents=True)
    exe = exe_dir / "python3.11"
    exe.write_text("x")
    monkeypatch.setattr(py311.sys, "platform", "linux")
    monkeypatch.setattr(py311, "_verify_runs", lambda e: True)

    def boom(url, dest, progress=None):
        raise AssertionError(
            "_download must not run when a verified interpreter "
            "already exists")
    monkeypatch.setattr(py311, "_download", boom)
    assert py311.ensure_python311(str(tmp_path)) == str(exe)


def test_stale_interpreter_failing_verify_triggers_one_redownload(
        tmp_path, monkeypatch):
    exe_dir = tmp_path / "python" / "bin"
    exe_dir.mkdir(parents=True)
    (exe_dir / "python3.11").write_text("stale")
    triple = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr(py311, "target_triple", lambda: triple)
    monkeypatch.setattr(py311.sys, "platform", "linux")
    responses = iter([False, True])  # stale exe fails, fresh one passes
    monkeypatch.setattr(py311, "_verify_runs", lambda e: next(responses))
    calls = _install_fake_download(
        monkeypatch, {"python/bin/python3.11": b"fresh"})
    monkeypatch.setattr(
        py311, "_sha256", lambda path: py311._DIGESTS[triple])
    py311.ensure_python311(str(tmp_path))
    assert len(calls) == 1


def test_unsupported_platform_raises_actionable_runtime_error(
        tmp_path, monkeypatch):
    monkeypatch.setattr(py311, "target_triple", lambda: None)
    with pytest.raises(RuntimeError, match="Setup tab"):
        py311.ensure_python311(str(tmp_path))


def test_download_failure_raises_and_cleans_up_partial_archive(
        tmp_path, monkeypatch):
    triple = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr(py311, "target_triple", lambda: triple)
    monkeypatch.setattr(py311.sys, "platform", "linux")

    def flaky(url, dest, progress=None):
        Path(dest).write_bytes(b"partial")  # simulate a dropped conn
        raise OSError("connection reset")
    monkeypatch.setattr(py311, "_download", flaky)
    dest = tmp_path / "py311"
    with pytest.raises(RuntimeError, match="Could not download") as ei:
        py311.ensure_python311(str(dest))
    assert "Setup tab" in str(ei.value)
    assert not (dest / py311.asset_name(triple)).exists()


def test_checksum_mismatch_raises_and_cleans_up_archive(
        tmp_path, monkeypatch):
    triple = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr(py311, "target_triple", lambda: triple)
    monkeypatch.setattr(py311.sys, "platform", "linux")
    _install_fake_download(
        monkeypatch, {"python/bin/python3.11": b"stub"})
    monkeypatch.setattr(py311, "_sha256", lambda path: "0" * 64)
    dest = tmp_path / "py311"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        py311.ensure_python311(str(dest))
    assert not (dest / py311.asset_name(triple)).exists()


def test_ensure_python311_extracts_to_expected_interpreter_path_posix(
        tmp_path, monkeypatch):
    triple = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr(py311, "target_triple", lambda: triple)
    monkeypatch.setattr(py311.sys, "platform", "linux")
    _install_fake_download(
        monkeypatch, {"python/bin/python3.11": b"stub"})
    monkeypatch.setattr(
        py311, "_sha256", lambda path: py311._DIGESTS[triple])
    monkeypatch.setattr(py311, "_verify_runs", lambda e: True)
    dest = tmp_path / "py311"
    exe = py311.ensure_python311(str(dest))
    assert exe == os.path.join(str(dest), "python", "bin", "python3.11")
    assert os.path.exists(exe)
    assert not (dest / py311.asset_name(triple)).exists()  # cleaned up


def test_ensure_python311_extracts_to_expected_interpreter_path_windows(
        tmp_path, monkeypatch):
    triple = "x86_64-pc-windows-msvc"
    monkeypatch.setattr(py311, "target_triple", lambda: triple)
    monkeypatch.setattr(py311.sys, "platform", "win32")
    _install_fake_download(
        monkeypatch, {"python/python.exe": b"stub"})
    monkeypatch.setattr(
        py311, "_sha256", lambda path: py311._DIGESTS[triple])
    monkeypatch.setattr(py311, "_verify_runs", lambda e: True)
    dest = tmp_path / "py311"
    exe = py311.ensure_python311(str(dest))
    assert exe == os.path.join(str(dest), "python", "python.exe")


def test_extraction_without_expected_member_raises(tmp_path, monkeypatch):
    triple = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr(py311, "target_triple", lambda: triple)
    monkeypatch.setattr(py311.sys, "platform", "linux")
    _install_fake_download(
        monkeypatch, {"python/README.txt": b"no interpreter in here"})
    monkeypatch.setattr(
        py311, "_sha256", lambda path: py311._DIGESTS[triple])
    with pytest.raises(RuntimeError, match="no interpreter was found"):
        py311.ensure_python311(str(tmp_path / "py311"))


def test_downloaded_interpreter_failing_verify_raises(
        tmp_path, monkeypatch):
    triple = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr(py311, "target_triple", lambda: triple)
    monkeypatch.setattr(py311.sys, "platform", "linux")
    _install_fake_download(
        monkeypatch, {"python/bin/python3.11": b"stub"})
    monkeypatch.setattr(
        py311, "_sha256", lambda path: py311._DIGESTS[triple])
    monkeypatch.setattr(py311, "_verify_runs", lambda e: False)
    with pytest.raises(RuntimeError, match="did not run correctly"):
        py311.ensure_python311(str(tmp_path / "py311"))


def test_unpinned_triple_skips_checksum_verification(
        tmp_path, monkeypatch):
    """Defensive branch: _DIGESTS.get(triple) is None for a triple with
    no pinned digest — extraction must still succeed, just unverified."""
    monkeypatch.setattr(py311, "target_triple", lambda: "made-up-triple")
    monkeypatch.setattr(py311.sys, "platform", "linux")
    assert "made-up-triple" not in py311._DIGESTS
    _install_fake_download(
        monkeypatch, {"python/bin/python3.11": b"stub"})
    monkeypatch.setattr(py311, "_verify_runs", lambda e: True)
    exe = py311.ensure_python311(str(tmp_path / "py311"))
    assert os.path.exists(exe)


def test_progress_callback_receives_fetch_and_extract_messages(
        tmp_path, monkeypatch):
    triple = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr(py311, "target_triple", lambda: triple)
    monkeypatch.setattr(py311.sys, "platform", "linux")
    _install_fake_download(
        monkeypatch, {"python/bin/python3.11": b"stub"})
    monkeypatch.setattr(
        py311, "_sha256", lambda path: py311._DIGESTS[triple])
    monkeypatch.setattr(py311, "_verify_runs", lambda e: True)
    messages = []
    py311.ensure_python311(str(tmp_path / "py311"),
                           progress=messages.append)
    joined = " ".join(messages)
    assert "Fetching" in joined
    assert "Extracting" in joined
    assert "Downloading Python 3.11" in joined  # from the fake fetcher


def test_ensure_python311_is_idempotent(tmp_path, monkeypatch):
    triple = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr(py311, "target_triple", lambda: triple)
    monkeypatch.setattr(py311.sys, "platform", "linux")
    calls = _install_fake_download(
        monkeypatch, {"python/bin/python3.11": b"stub"})
    monkeypatch.setattr(
        py311, "_sha256", lambda path: py311._DIGESTS[triple])
    monkeypatch.setattr(py311, "_verify_runs", lambda e: True)
    dest = tmp_path / "py311"
    first = py311.ensure_python311(str(dest))
    second = py311.ensure_python311(str(dest))
    assert first == second
    assert len(calls) == 1


# --- plugin_utils.installer wiring ------------------------------------------

def test_managed_base_python_prefers_a_path_3_11(monkeypatch):
    monkeypatch.setattr(
        installer.shutil, "which",
        lambda name: "/usr/bin/python3.11" if name == "python3.11"
        else None)
    monkeypatch.setattr(installer, "_python_version", lambda exe: (3, 11))

    def boom(dest, progress=None):
        raise AssertionError(
            "ensure_python311 must not run when PATH already has 3.11")
    monkeypatch.setattr(py311, "ensure_python311", boom)
    assert installer.managed_base_python("/plugin/dir") == (
        "/usr/bin/python3.11")


def test_managed_base_python_downloads_when_path_has_no_3_11(
        monkeypatch, tmp_path):
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        installer, "managed_root", lambda plugin_dir: str(tmp_path))
    calls = []

    def fake_ensure(dest, progress=None):
        calls.append(dest)
        return "/fake/interpreter"
    monkeypatch.setattr(py311, "ensure_python311", fake_ensure)
    result = installer.managed_base_python("plugin-dir-is-irrelevant")
    assert result == "/fake/interpreter"
    assert calls == [os.path.join(str(tmp_path), "py311")]


def test_managed_base_python_skips_a_wrong_version_on_path(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        installer.shutil, "which",
        lambda name: "/usr/bin/python3" if name == "python3" else None)
    monkeypatch.setattr(installer, "_python_version", lambda exe: (3, 9))
    monkeypatch.setattr(
        installer, "managed_root", lambda plugin_dir: str(tmp_path))
    monkeypatch.setattr(
        py311, "ensure_python311",
        lambda dest, progress=None: "/downloaded")
    assert installer.managed_base_python("pd") == "/downloaded"


def test_choose_base_python_prefers_a_path_3_11(monkeypatch):
    monkeypatch.setattr(
        installer.shutil, "which",
        lambda name: "/usr/bin/python3.11" if name == "python3.11"
        else None)
    monkeypatch.setattr(installer, "_python_version", lambda exe: (3, 11))

    def boom(plugin_dir, progress=None):
        raise AssertionError("must not download when PATH has 3.11")
    monkeypatch.setattr(installer, "managed_base_python", boom)
    assert installer.choose_base_python() == "/usr/bin/python3.11"


def test_choose_base_python_falls_back_when_path_has_no_3_11(
        monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    seen = {}

    def fake_managed(plugin_dir, progress=None):
        seen["plugin_dir"] = plugin_dir
        seen["progress"] = progress
        return "/downloaded/python3.11"
    monkeypatch.setattr(installer, "managed_base_python", fake_managed)
    marker = object()
    result = installer.choose_base_python(progress=marker)
    assert result == "/downloaded/python3.11"
    assert seen["plugin_dir"] == installer._PLUGIN_DIR
    assert seen["progress"] is marker


def test_choose_base_python_no_longer_raises_with_an_empty_path(
        monkeypatch):
    """Regression: this used to RuntimeError here. The whole point of
    Task 1 is that a bare machine now gets a silent, successful
    fallback instead of a dead end."""
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        installer, "managed_base_python",
        lambda plugin_dir, progress=None: "/downloaded/python3.11")
    installer.choose_base_python()  # must not raise


def test_choose_base_python_propagates_download_failure(monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)

    def fake_managed(plugin_dir, progress=None):
        raise RuntimeError(
            "Could not download Python 3.11 (x): boom. Install Python "
            "3.11 yourself and point WINMOL at it from the plugin's "
            "Setup tab.")
    monkeypatch.setattr(installer, "managed_base_python", fake_managed)
    with pytest.raises(RuntimeError, match="Setup tab"):
        installer.choose_base_python()


def test_create_venv_threads_progress_into_choose_base_python(
        monkeypatch, tmp_path):
    seen = {}

    def fake_choose(progress=None):
        seen["progress"] = progress
        return "/fake/python3.11"
    monkeypatch.setattr(installer, "choose_base_python", fake_choose)
    monkeypatch.setattr(installer, "_run_streamed", lambda *a, **k: None)
    logs = []
    installer.create_venv(str(tmp_path / "venv"), progress=logs.append)
    assert seen["progress"] is not None
    assert callable(seen["progress"])
