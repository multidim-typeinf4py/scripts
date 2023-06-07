#!/usr/bin/env bash

set -o nounset

source ./slurm/_common.sh
parameter_inference "typewritertop10" "$1"
