#!/usr/bin/env bash

export PYTHONUNBUFFERED=1

manytypes4py_repos() {
    find "$1" -maxdepth 2 -mindepth 2 -type d -print0
}

## Single removal

parameter_inference() {
    echo "Tool: $1 - Inferring: Parameters"
    conda run --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --outpath "$(dirname "$2")/$1"
}

variable_inference() {
    echo "Tool: $1 - Inferring: Variables"
    conda run --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --outpath "$(dirname "$2")/$1"
}

return_inference() {
    echo "Tool: $1 - Inferring: Returns"
    conda run --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}


## Double removal

parameter_return_inference() {
    echo "Tool: $1 - Inferring: Parameters and Returns"
    conda run --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}


variable_return_inference() {
    echo "Tool: $1 - Inferring: Variables and Returns"
    conda run --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}


variable_parameter_inference() {
    echo "Tool: $1 - Inferring: Variables and Params"
    conda run --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --outpath "$(dirname "$2")/$1"
}

## Triple

variable_parameter_return_inference() {
    echo "Tool: $1 - Inferring: Variables, Parameters and Returns"
    conda run --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}
