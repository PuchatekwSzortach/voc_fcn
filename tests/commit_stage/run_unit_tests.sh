#!/usr/bin/env bash

source activate voc_fcn_environment

conda env list
which python

py.test
