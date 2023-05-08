#!/usr/bin/env bash

set -o xtrace
set -e

BASE_ENV=monkeytype-base
EXECUTING_ENV=monkeytype-executor

# Create environment for running project
conda create --clone ${BASE_ENV} \
    --name ${EXECUTING_ENV}

if [ -f repo/pyproject.toml ]; then
    conda env update --name ${EXECUTING_ENV} \
        --cwd repo \
        poetry install pyproject.toml
fi

if [ -f repo/requirements.txt ]; then
    conda env update --name ${EXECUTING_ENV} \
        --cwd repo \
        pip install -f requirements.txt
fi

if [ -f repo/environment.yml ]; then
    conda env update --name ${EXECUTING_ENV} \
        --cwd repo \
        environment.yml
fi

if [ -f repo/setup.py ]; then
    conda run --name ${EXECUTING_ENV} \
        --cwd repo \
        pip install -e .
fi

# Try to find entrypoints

entrypoint_hints=( 'if __name__ == "__main__":' "if __name__ == '__main__':" )
for ep in "${entrypoint_hints[@]}"
do
    grep -l --null --fixed-strings "$ep" -r repo/ | \
        xargs -0 -I{} basename {} | \
        xargs -I{} conda run --name ${EXECUTING_ENV} --cwd repo monkeytype run "{}"
done

PYTEST=$(conda run --name ${EXECUTING_ENV} which pytest)
conda run --name ${EXECUTING_ENV} \
    --no-capture-output --cwd repo \
    monkeytype run -m pytest --timeout=30

conda run --name ${EXECUTING_ENV} \
    --no-capture-output --cwd repo \
    monkeytype apply --disable-type-rewriting --ignore-existing-annotations --pep_563 