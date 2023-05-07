#!/usr/bin/env sh

BASE_ENV=monkeytype-base
EXECUTING_ENV=monkeytype-executor

# Clear results of previous run
conda remove --name ${EXECUTING_ENV}
rm repo/monkeytype.sqlite3

# Create environment for running project
conda create --clone ${BASE_ENV} \
    --name ${EXECUTING_ENV}

if [ -f repo/pyproject.toml ]; then
    conda run --name ${EXECUTING_ENV} \
        poetry install repo/pyproject.toml
fi

if [ -f repo/requirements.txt ]; then
    conda run --name ${EXECUTING_ENV} \
        pip install repo/requirements.txt
fi

if [ -f repo/environment.yml ]; then
    conda env update --name ${EXECUTING_ENV} \
        repo/environment.yml
fi

# Try to find entrypoints
grep -l --null --fixed-strings 'if __name__ == "__main__":' -r . | xargs -i{} --null \
    conda run --name ${EXECUTING_ENV} monkeytype run {}

PYTEST=$(conda run --name ${EXECUTING_ENV} which pytest)
conda run --name ${EXECUTING_ENV} \ 
    monkeytype run ${PYTEST} --timeout=30
