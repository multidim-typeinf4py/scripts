#!/bin/bash

#SBATCH --output=REP-%x.%j.out
#SBATCH --error=REP-%x.%j.err
#SBATCH --mail-user=ab270@stud.uni-heidelberg.de
#SBATCH --mail-type=BEGIN,END,FAIL

source ./slurm/scripts/_common.sh

manytypes4py_repos "$1" | while IFS= read -r -d $'\0' repository; do
    parameter_inference "pyreinfer" "$repository"
done
