#!/usr/bin/env bash

#SBATCH --output=slurm-runs/%x.%j.out
#SBATCH --error=slurm-runs/%x.%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --mail-user=ab270@stud.uni-heidelberg.de
#SBATCH --mail-type=BEGIN,END,FAIL

set -o nounset

source ./slurm/scripts/_env.sh
source ./slurm/scripts/_common.sh

return_inference "mypy" "$1"