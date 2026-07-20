#!/usr/bin/env bash
# Run QGIS inside a container with this repo mounted as the plugin, and pass the
# GUI out to the host's X server. It exists to test the plugin against a
# specific QGIS version (the image pins one) without installing that QGIS
# locally -- useful on Linux, and the only practical way to test a QGIS version
# you do not want on your workstation.
#
# Build the image first:
#     docker build -t winmol_analyser_docker .
#
# NOTE: `xhost +` below disables X access control for ALL clients, not just this
# container. Prefer `xhost +local:docker` and run `xhost -` when you are done.

set -euo pipefail

xhost +

# Historical note: this script used to `rm -rf winmol_venv` before starting,
# because the plugin created its virtualenv INSIDE the plugin directory -- which
# is this repo, bind-mounted below -- so a host-built venv leaked into the
# container and QGIS tried to run a host interpreter under Linux.
#
# That is fixed: installer.managed_root() now puts the venv under the QGIS
# PROFILE directory instead (see plugin_utils/installer.py), which was the fix
# for QGIS reporting "plugin uninstall failed". Inside this container that
# resolves to /root/.local/share/QGIS/QGIS3/profiles/default/winmol/, which is
# container-local and not part of the mount. So there is nothing to delete.
#
# For the record, .dockerignore would NOT have solved this either: it filters
# the build context for COPY/ADD during `docker build`, and has no effect on
# the bind mount below.

docker run --rm -it --name qgis --net host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)":/root/.local/share/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyser \
    -e DISPLAY="unix$DISPLAY" \
    winmol_analyser_docker \
    qgis
