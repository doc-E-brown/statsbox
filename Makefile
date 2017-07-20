#!/bin/bash
#
# Statsbox build scripts
#
# Ben Johnston
# License: 3-Clause BSD
# S.D.G

## VARIABLES#################################################################
PACKAGES=statsbox
COVER_HTML=cover
OMIT=*/test*.py,*/migrations/*.py
#############################################################################

all: test lint 

test: 
	nosetests -s --with-coverage --cover-html --cover-html-dir=${COVER_HTML} --cover-package=${PACKAGES}

lint:
	pylint --msg-template='{msg_id}:{line:3d},{column}: {obj}: {msg}' statsbox 

.PHONY:
	clean

