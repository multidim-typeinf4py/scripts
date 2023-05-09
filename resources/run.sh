#!/usr/bin/env bash

set -o xtrace
set -o nounset
set -e

export BASE_ENV=monkeytype-base
export EXECUTING_ENV=monkeytype-executor

READABLE_REPO=$1
MTOUTPUT=$2
MONKEYTYPE_TRACE_MODULES=$3
export MT_DB_PATH=$MTOUTPUT/monkeytype.sqlite3

# Create environment for running project
conda create --clone ${BASE_ENV} \
    --name ${EXECUTING_ENV}

if [ -f repo/pyproject.toml ]; then
    conda run --name ${EXECUTING_ENV} \
        --cwd "$READABLE_REPO" \
        poetry install pyproject.toml
fi

if [ -f repo/requirements.txt ]; then
    conda run --name ${EXECUTING_ENV} \
        --cwd "$READABLE_REPO" \
        pip install -f requirements.txt
fi

if [ -f repo/environment.yml ]; then
    conda run --name ${EXECUTING_ENV} \
        --cwd "$READABLE_REPO" \
        environment.yml
fi

if [ -f repo/setup.py ]; then
    conda run --name ${EXECUTING_ENV} \
        --cwd "$READABLE_REPO" \
        pip install -e .
fi

# Try to find entrypoints
entrypoint_hints=( 'if __name__ == "__main__":' "if __name__ == '__main__':" )
for ep in "${entrypoint_hints[@]}"
do
    grep -l --null --fixed-strings "$ep" -r "$READABLE_REPO"/ | \
        xargs -I{} conda run --name ${EXECUTING_ENV} --cwd "$READABLE_REPO" \
            monkeytype --disable-type-rewriting run "{}"
done

conda run --name ${EXECUTING_ENV} \
    --no-capture-output --cwd "$READABLE_REPO" \
    monkeytype --disable-type-rewriting run -m pytest --timeout=30 || true

echo "$MONKEYTYPE_TRACE_MODULES" | tr ',' '\n' | xargs -n1 -I{} conda run --name ${EXECUTING_ENV} \
      --no-capture-output --cwd "$MTOUTPUT" \
      monkeytype --disable-type-rewriting apply {} --ignore-existing-annotations --pep_563