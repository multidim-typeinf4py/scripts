#!/usr/bin/env bash

set -o nounset

source ./slurm/_common.sh
parameter_inference "typet5top10" "$1"
