#!/bin/bash
# Remove the existing build directory
rm -rf _build/

# Build the HTML documentation using Sphinx
sphinx-build -b html . _build/html/
