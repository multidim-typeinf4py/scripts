#!/usr/bin/env bash

module load devel/miniconda
conda activate scripts-venv

which python
echo CPUs: "$(nproc)"