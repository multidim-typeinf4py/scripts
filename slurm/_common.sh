#!/usr/bin/env bash

export PYTHONUNBUFFERED=1


## Single removal

parameter_inference() {
    echo "Tool: $1 - Inferring: Parameters"
    conda run  --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --outpath "$(dirname "$2")/$1"
}

variable_inference() {
    set -o xtrace

    echo "Tool: $1 - Inferring: Variables"
    conda run  --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --outpath "$(dirname "$2")/$1"
}

return_inference() {
    echo "Tool: $1 - Inferring: Returns"
    conda run  --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}


## Double removal

parameter_return_inference() {
    echo "Tool: $1 - Inferring: Parameters and Returns"
    conda run  --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}


variable_return_inference() {
    echo "Tool: $1 - Inferring: Variables and Returns"
    conda run  --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}


variable_parameter_inference() {
    echo "Tool: $1 - Inferring: Variables and Params"
    conda run  --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --outpath "$(dirname "$2")/$1"
}

## Triple

variable_parameter_return_inference() {
    echo "Tool: $1 - Inferring: Variables, Parameters and Returns"
    conda run  --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}
