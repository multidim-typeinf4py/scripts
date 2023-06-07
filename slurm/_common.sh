#!/usr/bin/env bash

parameter_inference() {
    echo "Tool: $1 - Inferring: Parameters"
    conda run --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --task CALLABLE_PARAMETER \
        --outpath "$(dirname "$2")/$1"
}

variable_inference() {
    echo "Tool: $1 - Inferring: Variables"
    conda run --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --task VARIABLE \
        --outpath "$(dirname "$2")/$1"
}

return_inference() {
    echo "Tool: $1 - Inferring: Returns"
    conda run --no-capture-output --name scripts-venv python -u main.py infer --dataset "$2" \
        --tool "$1" \
        --task CALLABLE_RETURN \
        --outpath "$(dirname "$2")/$1"
}
