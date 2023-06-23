#!/usr/bin/env bash

set -o xtrace
set -o nounset
set -e

export BASE_ENV=monkeytype-base
export EXECUTING_ENV=monkeytype-executor

READABLE_REPO=$1
export MT_DB_PATH=$2

# Create environment for running project
conda create --clone ${BASE_ENV} \
    --name ${EXECUTING_ENV}


### Poetry
if [ -f repo/pyproject.toml ] && [ ! -f repo/poetry.lock ]; then
    conda run --name ${EXECUTING_ENV} \
        --no-capture-output \
        --cwd "$READABLE_REPO" \
        poetry lock --no-update

fi

if [ -f repo/poetry.lock ]; then
    conda run --name ${EXECUTING_ENV} \
        --no-capture-output \
        --cwd "$READABLE_REPO" \
        poetry export --without-hashes -f requirements.txt --output poetry-requirements.txt

    conda run --name ${EXECUTING_ENV} \
        --no-capture-output \
        --cwd "$READABLE_REPO" \
        pip install -r poetry-requirements.txt
fi



### requirements.txt
if [ -f repo/requirements.txt ]; then
    conda run --name ${EXECUTING_ENV} \
      --no-capture-output \
      --cwd "$READABLE_REPO" \
      pip install -r requirements.txt
fi


if [ -f repo/requirements-test.txt ]; then
    conda run --name ${EXECUTING_ENV} \
      --no-capture-output \
      --cwd "$READABLE_REPO" \
      pip install -r requirements-test.txt
fi

if [ -f repo/setup.py ]; then
    conda run --name ${EXECUTING_ENV} \
        --no-capture-output \
        --cwd "$READABLE_REPO" \
        pip install -e . || true
fi

# Try to find entrypoints
#entrypoint_hints=( 'if __name__ == "__main__":' "if __name__ == '__main__':" )
#for ep in "${entrypoint_hints[@]}"
#do
#    grep -l --null --fixed-strings "$ep" -r "$READABLE_REPO"/ | \
#        xargs -I{} conda run --name ${EXECUTING_ENV} --cwd "$READABLE_REPO" \
#            monkeytype --disable-type-rewriting run "{}"
#done

conda run --name ${EXECUTING_ENV} \
    --no-capture-output --cwd "$READABLE_REPO" \
    monkeytype --disable-type-rewriting run -m pytest --timeout=60 || true

conda run \
  --name ${EXECUTING_ENV} \
  --no-capture-output \
  --cwd "$READABLE_REPO" monkeytype list-modules | xargs -n1 -I{} \
    conda run --name ${EXECUTING_ENV} \
    --no-capture-output \
    --cwd "$READABLE_REPO" \
    monkeytype apply {} --ignore-existing-annotations --pep_563 || true