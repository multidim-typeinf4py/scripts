#!/usr/bin/env bash

export MDTI4PY_CPUS=${MDTI4PY_CPUS:=12}

parameter_inference() {
    echo "Tool: $1 - Inferring: Parameters"
    poetry run python main.py infer --dataset "$2" \
        --tool "$1" \
        --task CALLABLE_PARAMETER \
        --outpath "$(dirname "$2")/$1"
}

variable_inference() {
    echo "Tool: $1 - Inferring: Variables"
    poetry run python main.py infer --dataset "$2" \
        --tool "$1" \
        --task VARIABLE \
        --outpath "$(dirname "$2")/$1"
}

return_inference() {
    echo "Tool: $1 - Inferring: Returns"
    poetry run python main.py infer --dataset "$2" \
        --tool "$1" \
        --task CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}

all_inference() {
    echo "Tool: $1 - TT5-Experiment"
    poetry run python main.py tt5exp --dataset "$2" \
        --tool "$1" \
        --outpath "$(dirname "$2")/$1"
}
