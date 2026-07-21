###
## QGIS Docker container to run QGIS in, for X11 GUI testing of the plugin.
## Contains no TensorFlow and nothing machine-specific -- see docs/CONTAINERS.md.
###
## The QGIS version is a build arg, so bumping it needs no edit here:
##     docker build --build-arg QGIS_TAG=4.2.0-trixie -t winmol_analyser_docker:4.2.0-trixie .
## ARG before FROM is the only placement Docker expands in a base image ref.
##
## Tags verified on hub.docker.com/v2/repositories/qgis/qgis/tags, 2026-07-21:
##   3.44.12-noble     current LTR on Ubuntu 24.04 LTS   <- default
##   3.44.12-trixie    same LTR on Debian 13
##   4.2.0-trixie      current stable (Qt6); see docs/CONTAINERS.md before use
## Do NOT use `ltr`/`3.44`/`3.44.12`: all three share one digest with
## 3.44.12-questing, i.e. Ubuntu 25.10, a 9-month interim release.
ARG QGIS_TAG=3.44.12-noble
FROM qgis/qgis:${QGIS_TAG}

# Unversioned on purpose. `python3.10-venv` only exists on jammy (the old 3.28
# base); noble ships 3.12 and trixie/questing ship 3.13, so a pinned name turns
# every version bump into "Unable to locate package".
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

# -f: newer bases may already provide /usr/bin/python, and a plain `ln -s`
# would abort the build.
RUN ln -sf /usr/bin/python3 /usr/bin/python
