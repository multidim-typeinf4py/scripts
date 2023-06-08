#!/usr/bin/env bash

set -o nounset

source ./hyper/_common.sh
variable_inference "typet5topn10" "$1"
