#!/usr/bin/env bash

set -o nounset

source ./hyper/_common.sh
variable_inference "pyrequery" "$1"
