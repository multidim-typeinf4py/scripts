#!/usr/bin/env bash

set -o nounset

source ./hyper/_common.sh
parameter_inference "pyreinfer" "$1"
