#!/usr/bin/env bash

set -o nounset

source ./slurm/_common.sh
parameter_inference "hitype4pytop10" "$1"
