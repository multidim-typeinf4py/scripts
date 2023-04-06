#!/bin/bash

#SBATCH --output=REP-%x.%j.out
#SBATCH --error=REP-%x.%j.err
#SBATCH --mail-user=ab270@stud.uni-heidelberg.de
#SBATCH --mail-type=BEGIN,END,FAIL

source ./slurm/scripts/_common.sh

manytypes4py_repos "$1" | xargs -0 -L1 -P$SLURM_NTASKS \
    srun -n1 -N1 --exclusive installenv.sh