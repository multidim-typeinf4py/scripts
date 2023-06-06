#!/usr/bin/env bash

#SBATCH --output=slurm-runs/%x.%j.out
#SBATCH --error=slurm-runs/%x.%j.err

#SBATCH --mail-user=ab270@stud.uni-heidelberg.de
#SBATCH --mail-type=BEGIN,END,FAIL

set -o nounset

source ./slurm/_env.sh
source ./slurm/_common.sh

variable_inference "mypy" "$1"