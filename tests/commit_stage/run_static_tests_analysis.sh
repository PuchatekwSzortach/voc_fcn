#!/usr/bin/env bash

# Exit on error
set -e

source activate voc_fcn_environment

echo "Running pylint..."
pylint ./*

echo "Running pycodestyle..."
pycodestyle ./*

echo "Running xenon..."
xenon . --max-absolute A