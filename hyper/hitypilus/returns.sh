#!/usr/bin/env bash

set -o nounset

source ./hyper/_common.sh
return_inference "hitypilustop10" "$1"
