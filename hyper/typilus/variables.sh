#!/usr/bin/env bash

set -o nounset

source ./slurm/_common.sh
parameter_inference "typilustop10" "$1"
