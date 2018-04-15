#!/usr/bin/env bash

source activate voc_fcn_environment

echo "Running pylint..."
pylint ./*

echo "Running pycodestyle..."
pycodestyle ./*
