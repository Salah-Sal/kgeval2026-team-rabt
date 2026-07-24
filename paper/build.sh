#!/bin/sh
# Build the paper with tectonic (brew install tectonic).
# First run needs network to fetch TeX packages; cached afterwards.
cd "$(dirname "$0")" && exec tectonic main.tex "$@"
