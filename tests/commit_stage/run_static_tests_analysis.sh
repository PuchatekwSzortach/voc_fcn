#!/usr/bin/env bash

# Exit on error
set -e

source activate voc_fcn_environment

DIRECTORIES_TO_SCAN="net scripts tests"

echo "Running pylint..."
pylint $DIRECTORIES_TO_SCAN

echo "Running pycodestyle..."
pycodestyle $DIRECTORIES_TO_SCAN

echo "Running xenon..."
xenon . --max-absolute B
