"""Obtain a self-contained CPython 3.11 on any platform — no admin rights,
no system Python required.

WINMOL's compute environment (onnxruntime + the geo stack) needs Python
3.11 (validated only on 3.11). A bare machine may have none: fresh Windows
ships none, macOS ships 3.9, and QGIS's own bundled Python has a baked-in
prefix and dies standalone ("No module named encodings"). So this module
downloads a *relocatable* python-build-standalone (PBS) 3.11 build and
hands it back as the base interpreter for the venv.

Pinned to one known-good PBS release with per-platform SHA-256 digests, so
the download is reproducible and integrity-checked. Pure stdlib -> import-
safe off QGIS and unit-testable without a network.

Reference: https://github.com/astral-sh/python-build-standalone
"""
import hashlib
import os
import platform
import sys
import tarfile
import time
import urllib.request

from .childenv import PY_VERSION_PROBE, run_isolated

# Pinned PBS release (see the release assets' `digest` field on GitHub).
PBS_TAG = "20260623"
PBS_PYVER = "3.11.15"
_RELEASE_BASE = (
    "https://github.com/astral-sh/python-build-standalone/releases/"
    f"download/{PBS_TAG}/")

# PBS platform triple -> sha256 of
#   cpython-<PBS_PYVER>+<PBS_TAG>-<triple>-install_only.tar.gz
_DIGESTS = {
    "aarch64-apple-darwin":
        "d2324bfd1a7b9fc44ccd884c3a2505bcab6691dbfd4f8270e10c50aaa4e19506",
    "x86_64-apple-darwin":
        "38f3c18a4ccbd6faa09243c45c85d8e09b5a7b345e02f174346cf72ebf901f87",
    "x86_64-pc-windows-msvc":
        "7e0a8abfee952efc63dff290022a73f0185b586f522678ae7a757a56f23c289b",
    "aarch64-pc-windows-msvc":
        "047ba6cc431f9ace9a8b2cb4d74546dd439118a7e9340fdec0b466ecb82c4cb5",
    "x86_64-unknown-linux-gnu":
        "60295e3e703b48c270e8d8c685195b8d5c2f0b8a596c1a910d7e24a2cc55afdd",
    "aarch64-unknown-linux-gnu":
        "1de978b7039f345dacdddc3efb0726ce5b957bbbd34161037a4b426aabb18bf5",
}


def target_triple():
    """The PBS platform triple for this machine, or None if unsupported."""
    sysname = platform.system()
    mach = platform.machine().lower()
    arm = mach in ("arm64", "aarch64")
    x64 = mach in ("x86_64", "amd64", "x64")
    if sysname == "Darwin":
        return ("aarch64-apple-darwin" if arm else
                "x86_64-apple-darwin" if x64 else None)
    if sysname == "Windows":
        return ("aarch64-pc-windows-msvc" if arm else
                "x86_64-pc-windows-msvc" if x64 else None)
    if sysname == "Linux":
        return ("aarch64-unknown-linux-gnu" if arm else
                "x86_64-unknown-linux-gnu" if x64 else None)
    return None


def asset_name(triple):
    return f"cpython-{PBS_PYVER}+{PBS_TAG}-{triple}-install_only.tar.gz"


def asset_url(triple):
    return _RELEASE_BASE + asset_name(triple)


def _interp_path(python_root):
    """Interpreter path inside an extracted PBS 'python/' directory."""
    if sys.platform.startswith("win"):
        return os.path.join(python_root, "python.exe")
    p = os.path.join(python_root, "bin", "python3.11")
    return p if os.path.exists(p) else os.path.join(
        python_root, "bin", "python3")


def existing_interpreter(dest_dir):
    """A working 3.11 interpreter already extracted under dest_dir, else
    None. (dest_dir/python is where the archive extracts.)"""
    exe = _interp_path(os.path.join(dest_dir, "python"))
    return exe if os.path.exists(exe) else None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest, progress=None):
    req = urllib.request.Request(
        url, headers={"User-Agent": "winmol-analyzer"})
    # timeout bounds each socket read: a stalled/half-open connection raises
    # instead of blocking forever (which would hang the env-build thread).
    with urllib.request.urlopen(req, timeout=120) as resp, \
            open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        # Throttle to ~1 line/s or every 5 MB, else a fast link would push
        # dozens of lines/second into the log widget.
        last_at, last_mb = 0.0, -5
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if not progress:
                continue
            read_mb = read // (1 << 20)
            now = time.monotonic()
            if now - last_at < 1.0 and read_mb - last_mb < 5:
                continue
            last_at, last_mb = now, read_mb
            if total:
                progress(f"Downloading Python 3.11… "
                         f"{read_mb}/{total // (1 << 20)} MB")
            else:
                progress(f"Downloading Python 3.11… {read_mb} MB")


def _safe_extract(tar, path):
    """extractall guarding against path traversal. Uses the stdlib 'tar'
    filter where available (Py>=3.8.17/3.9.17/3.10.12/3.11.4/3.12), else a
    manual member check.

    The 'tar' filter -- not 'data' -- is deliberate. 'data' additionally
    rejects any symlink whose target resolves outside the destination, which
    the relocatable Python build's terminfo tree trips (a member such as
    share/terminfo/1/1178 -> ../a/adm1178), aborting the whole extract with
    "'...' would link to '...', which is outside the destination" (issue #25).
    The archive is verified by pinned SHA-256 above, so it is trusted; the
    'tar' filter still blocks the actual traversal attack -- absolute paths and
    '..' components in member names -- while honouring those symlinks."""
    try:
        tar.extractall(path, filter="tar")
        return
    except TypeError:
        pass  # old Python without the filter kwarg
    base = os.path.abspath(path)
    for member in tar.getmembers():
        dest = os.path.abspath(os.path.join(path, member.name))
        if not (dest == base or dest.startswith(base + os.sep)):
            raise RuntimeError(f"unsafe path in archive: {member.name}")
    tar.extractall(path)


def _verify_runs(exe):
    """True if exe runs and reports Python 3.11."""
    try:
        out = run_isolated(exe, PY_VERSION_PROBE, timeout=60)
        return out.returncode == 0 and out.stdout.strip() == "3.11"
    except Exception:
        return False


def ensure_python311(dest_dir, progress=None):
    """The path to a CPython 3.11 interpreter under dest_dir, downloading
    and extracting a relocatable PBS build if not already present.

    Idempotent: a prior successful extract is reused. Integrity is checked
    against the pinned SHA-256. Raises RuntimeError with an actionable
    message on any failure (unsupported platform, download/verify/extract
    error) — naming the manual fallback (install Python 3.11 yourself and
    point WINMOL at it from the plugin's Setup tab).
    """
    exe = existing_interpreter(dest_dir)
    if exe and _verify_runs(exe):
        return exe

    triple = target_triple()
    if triple is None:
        raise RuntimeError(
            "No prebuilt Python 3.11 is available for this platform "
            f"({platform.system()}/{platform.machine()}). Install Python "
            "3.11 yourself and point WINMOL at it from the plugin's "
            "Setup tab.")

    os.makedirs(dest_dir, exist_ok=True)
    name = asset_name(triple)
    archive = os.path.join(dest_dir, name)
    if progress:
        progress(f"Fetching {name}")
    try:
        _download(asset_url(triple), archive, progress=progress)
    except Exception as exc:
        if os.path.exists(archive):
            os.remove(archive)
        raise RuntimeError(
            f"Could not download Python 3.11 ({name}): {exc}. Install "
            "Python 3.11 yourself and point WINMOL at it from the "
            "plugin's Setup tab.")

    expect = _DIGESTS.get(triple)
    if expect:
        got = _sha256(archive)
        if got != expect:
            os.remove(archive)
            raise RuntimeError(
                f"Checksum mismatch for {name}\n  expected {expect}\n  "
                f"got      {got}")

    if progress:
        progress("Extracting Python 3.11…")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, dest_dir)
    finally:
        if os.path.exists(archive):
            os.remove(archive)

    exe = existing_interpreter(dest_dir)
    if not exe:
        raise RuntimeError(
            "Python 3.11 archive extracted but no interpreter was found "
            f"under {dest_dir}/python.")
    if not _verify_runs(exe):
        raise RuntimeError(
            f"Downloaded Python 3.11 at {exe} did not run correctly on "
            "this machine.")
    return exe
