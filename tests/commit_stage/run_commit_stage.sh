#!/usr/bin/env bash

set -e

./tests/commit_stage/run_unit_tests.sh
./tests/commit_stage/run_static_tests_analysis.sh
