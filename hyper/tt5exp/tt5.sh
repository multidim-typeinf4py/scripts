#!/usr/bin/env bash

set -o nounset

source ./hyper/_common.sh
all_inference "typet5topn1" "$1"
