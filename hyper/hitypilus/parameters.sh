#!/usr/bin/env bash

set -o nounset

source ./slurm/_common.sh
parameter_inference "hitypilustop10" "$1"
