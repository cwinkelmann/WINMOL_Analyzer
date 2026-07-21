#/***************************************************************************
# WINMOLAnalyzer

#
# Plugin to detect stems from UAV
#							 -------------------
#		begin				: 2023-11-29
#		git sha				: $Format:%H$
#		copyright			: (C) 2023 by Hochschule für nachhaltige Entwicklung Eberswalde | mundialis GmbH & Co. KG | terrestris GmbH & Co. KG
#		email				: Stefan.Reder@hnee.de
# ***************************************************************************/
#
#/***************************************************************************
# *																		 *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU General Public License as published by  *
# *   the Free Software Foundation; either version 2 of the License, or	 *
# *   (at your option) any later version.								   *
# *																		 *
# ***************************************************************************/

#################################################
# Edit the following to match your sources lists
#################################################


PLUGINNAME = WINMOL_Analyzer

# Interpreter used by `make test`. Override on the command line to select the
# environment that has the runtime dependencies installed.
PYTHON = python3

# Two distinct things that both used to be called VERSION:
#   REF            — the git ref `make package` exports (a commit or tag).
#                    Kept aliased to VERSION so the historically documented
#                    `make package VERSION=Version_0.3.2` still means the ref.
#   PLUGIN_VERSION — the version stamped into metadata.txt and into the zip
#                    FILE name. Defaults to whatever metadata.txt already
#                    says; CI derives it from the tag instead
#                    (scripts/plugin_version.py).
VERSION ?= HEAD
REF ?= $(VERSION)
PLUGIN_VERSION ?= $(shell sed -n 's/^version=//p' metadata.txt)

# The plugin GUI (QGIS side) AND the compute core it shells out to: winmol_run.py
# needs classes/, utils/, plugin_utils/, config.json and requirements/ to run.
PY_FILES = \
	__init__.py \
	winmol_analyzer.py winmol_analyzer_dialog.py \
	tasks_threads.py winmol_run.py winmol_batch.py

UI_FILES = winmol_analyzer_dialog_base.ui

EXTRAS = metadata.txt icon.png config.json

EXTRA_DIRS = classes utils plugin_utils requirements

# Vestigial: resources.qrc compiles exactly one file, icon.png, and nothing
# resolves a ':/plugins/...' path any more — winmol_analyzer.py loads the icon
# from disk so it works under both Qt5 and Qt6, and the .ui has an empty
# <resources/>. The import at winmol_analyzer.py is wrapped in try/except.
# Kept because scripts/build_plugin_zip.sh hard-requires resources.py in the
# package and aborts without it; removing it means editing that script in the
# same commit. See docs/root-layout.md.
COMPILED_RESOURCE_FILES = resources.py

PEP8EXCLUDE=pydev,resources.py,conf.py,third_party,ui

# QGISDIR points to the location where your plugin should be installed.
# This varies by platform, relative to your HOME directory:
#	* Linux:
#	  .local/share/QGIS/QGIS3/profiles/default/python/plugins/
#	* Mac OS X:
#	  Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins
#	* Windows:
#	  AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins'

# Auto-detect the profile location by OS (override on the command line if you
# use a non-default profile). Windows: pass QGISDIR explicitly.
ifeq ($(shell uname),Darwin)
QGISDIR=Library/Application Support/QGIS/QGIS3/profiles/default
else
QGISDIR=.local/share/QGIS/QGIS3/profiles/default
endif

#################################################
# Normally you would not need to edit below here
#################################################

RESOURCE_SRC=$(shell grep '^ *<file' resources.qrc | sed 's@</file>@@g;s/.*>//g' | tr '\n' ' ')

.PHONY: default compile test deploy dclean derase zip package clean pep8

default:
	@echo "Targets:"
	@echo "  compile  - rebuild resources.py from resources.qrc (pyrcc5)"
	@echo "  deploy   - install the plugin into your local QGIS profile"
	@echo "  package  - build the installable zip (scripts/build_plugin_zip.sh)"
	@echo "  test     - run the pytest suite"
	@echo "  pep8     - style check"

compile: $(COMPILED_RESOURCE_FILES)

%.py : %.qrc $(RESOURCES_SRC)
	pyrcc5 -o $*.py  $<

# The suite is QGIS-free and runs on the standalone interpreter. PYTHONHASHSEED
# is pinned because the golden-master fixtures depend on set-iteration order
# (tests/conftest.py re-executes pytest otherwise). Override PYTHON to point
# at the interpreter that has the runtime deps, e.g.
#   make test PYTHON=~/opt/anaconda3/envs/WINMOL_Analyzer/bin/python
test:
	@echo
	@echo "----------------------"
	@echo "Regression Test Suite"
	@echo "----------------------"
	PYTHONHASHSEED=0 TF_USE_LEGACY_KERAS=1 $(PYTHON) -m pytest tests/ -q

deploy: compile
	@echo
	@echo "------------------------------------------"
	@echo "Deploying plugin to your QGIS profile."
	@echo "------------------------------------------"
	# Installs into $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME).
	$(eval DEST := $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME))
	mkdir -p "$(DEST)"
	cp -vf $(PY_FILES) "$(DEST)"
	cp -vf $(UI_FILES) "$(DEST)"
	cp -vf $(COMPILED_RESOURCE_FILES) "$(DEST)"
	cp -vf $(EXTRAS) "$(DEST)"
	# Copy the compute-core directories the subprocess needs.
	for d in $(EXTRA_DIRS); do \
		rm -rf "$(DEST)/$$d"; cp -vfr "$$d" "$(DEST)/"; done


# The dclean target removes compiled python files from plugin directory
# also deletes any .git entry
dclean:
	@echo
	@echo "-----------------------------------"
	@echo "Removing any compiled python files."
	@echo "-----------------------------------"
	find $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME) -iname "*.pyc" -delete
	find $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME) -iname ".git" -prune -exec rm -Rf {} \;


derase:
	@echo
	@echo "-------------------------"
	@echo "Removing deployed plugin."
	@echo "-------------------------"
	rm -Rf $(HOME)/$(QGISDIR)/python/plugins/$(PLUGINNAME)

zip: deploy dclean
	@echo
	@echo "---------------------------"
	@echo "Creating plugin zip bundle."
	@echo "---------------------------"
	# The zip target deploys the plugin and creates a zip file with the deployed
	# content. You can then upload the zip file on http://plugins.qgis.org
	rm -f $(PLUGINNAME).zip
	cd $(HOME)/$(QGISDIR)/python/plugins; zip -9r $(CURDIR)/$(PLUGINNAME).zip $(PLUGINNAME)

package: compile
	# Create a zip package named $(PLUGINNAME)-<version>.zip. The version is
	# in the FILE name only: the directory inside the archive stays
	# $(PLUGINNAME), because QGIS identifies an installed plugin by it.
	# This requires use of git (your plugin development directory must be a
	# git repository).
	# Pass a valid commit or tag to export:
	#   make package REF=v0.6.0.2          (VERSION= is an accepted alias)
	# and optionally override the stamped version:
	#   make package REF=v0.6.0.2 PLUGIN_VERSION=0.6.0.2
	@echo
	@echo "------------------------------------"
	@echo "Exporting plugin to zip package.	"
	@echo "------------------------------------"
	bash scripts/build_plugin_zip.sh $(REF) \
		$(PLUGINNAME)-$(PLUGIN_VERSION).zip $(PLUGIN_VERSION)

clean:
	@echo
	@echo "------------------------------------"
	@echo "Removing rcc generated files"
	@echo "------------------------------------"
	rm -f $(COMPILED_RESOURCE_FILES)


# Run pep8 style checking
#http://pypi.python.org/pypi/pep8
pep8:
	@echo
	@echo "-----------"
	@echo "PEP8 issues"
	@echo "-----------"
	@pep8 --repeat --ignore=E203,E121,E122,E123,E124,E125,E126,E127,E128 --exclude $(PEP8EXCLUDE) . || true
	@echo "-----------"
	@echo "Ignored in PEP8 check:"
	@echo $(PEP8EXCLUDE)
