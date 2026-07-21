#!/usr/bin/env python3
"""Derive the plugin version and the release zip name from a git tag.

Single source of truth for release naming. Imported by
``tests/test_plugin_packaging.py`` and invoked as a CLI by
``.github/workflows/on-push-tags.yml`` and the ``Makefile``, so the rule
exists once instead of being duplicated in YAML, shell and Make.

Pure stdlib on purpose: it runs on a bare CI runner before any dependency
is installed, and it must never need QGIS or TensorFlow.

Version grammar
---------------
A tag may carry one leading ``v``, which is stripped. What remains must
start with a numeric segment and may have any number of dot-separated
numeric segments plus an optional alphanumeric suffix::

    v0.6.0.2      -> 0.6.0.2      (this project's four-segment convention)
    v0.5.0        -> 0.5.0
    v0.0.0-demo1  -> 0.0.0-demo1  (suffix kept verbatim)
    models-onnx-v1 -> ValueError

Deliberately looser than strict semver, because the project's own release
history is not semver (``v0.6.0.2`` has four segments and would be
rejected by a semver validator).

Suffixes are kept rather than mangled. QGIS's plugin installer
(``pyplugin_installer.version_compare``) compares versions segment-wise and
orders an unknown trailing text segment below the plain numeric version, so
``0.0.0-demo1`` sorts *below* the released ``0.6.0``. A demo build will
therefore never offer itself as an upgrade over a real release — which is
the desired behaviour for a throwaway tag, but it does mean tagging a
``-demo`` build on top of a release does not produce an installable update.

What is *not* tolerated is a tag with no numeric lead at all. The tag
``models-onnx-v1`` (a model-asset tag) once produced a published plugin
whose ``metadata.txt`` read ``version=models-onnx-v1``; that is not a valid
QGIS version string. Such tags now raise instead of shipping.
"""

import argparse
import re
import sys

#: The importable package name inside the archive. QGIS identifies an
#: installed plugin by this directory, so it must never carry a version:
#: a versioned directory would install every release as a *different*
#: plugin and break in-place upgrades.
PLUGIN_NAME = "WINMOL_Analyzer"

#: Stable, unversioned alias uploaded alongside the versioned artifact.
#: ``documentation/index.html`` links
#: ``releases/latest/download/WINMOL_Analyzer_QGIS_Plugin.zip``; that
#: permalink form cannot contain a version, so the historical name has to
#: keep resolving. Same bytes, second name.
STABLE_ZIP_NAME = "WINMOL_Analyzer_QGIS_Plugin.zip"

_VERSION_RE = re.compile(r"^\d+(\.\d+)*([-.][0-9A-Za-z][0-9A-Za-z.]*)?$")

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def derive_version(tag):
    """Return the plugin version encoded by a git tag.

    :param tag: the tag name, e.g. ``v0.6.0.2`` or ``0.6.0``.
    :returns: the version string for ``metadata.txt``.
    :raises ValueError: if the tag does not encode a usable version.
    """
    if not isinstance(tag, str):
        raise ValueError("tag must be a string, got %r" % (tag,))
    version = tag[1:] if tag.startswith("v") else tag
    if not _VERSION_RE.match(version):
        raise ValueError(
            "tag %r does not encode a plugin version: expected an optional "
            "leading 'v' followed by dot-separated numbers and an optional "
            "alphanumeric suffix (e.g. v0.6.0.2, v0.0.0-demo1)" % (tag,))
    return version


def zip_name(version):
    """Return the release artifact filename for *version*.

    The version goes in the *file* name only — see :data:`PLUGIN_NAME`.
    """
    if not isinstance(version, str) or not _SAFE_NAME_RE.match(version):
        raise ValueError(
            "version %r is not safe for a filename: expected only letters, "
            "digits, dot, underscore and hyphen" % (version,))
    return "%s-%s.zip" % (PLUGIN_NAME, version)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Derive the plugin version / zip name from a git tag.")
    parser.add_argument("tag", help="git tag, e.g. v0.6.0.2")
    parser.add_argument(
        "--zip-name", action="store_true",
        help="print the release zip filename instead of the version")
    args = parser.parse_args(argv)
    try:
        version = derive_version(args.tag)
    except ValueError as exc:
        parser.exit(2, "error: %s\n" % exc)
    print(zip_name(version) if args.zip_name else version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
