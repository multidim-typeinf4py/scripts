#!/usr/bin/env bash

manytypes4py_repos() {
    find "$1" -maxdepth 2 -mindepth 2 -type d -print0
}

## Single removal

parameter_inference() {
    echo "Tool: $1 - Inferring: Parameters"
    python main.py infer --dataset "$2" \
        --tool "$1" \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER
}

variable_inference() {
    echo "Tool: $1 - Inferring: Variables"
    python main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE
}

return_inference() {
    echo "Tool: $1 - Inferring: Returns"
    python main.py infer --dataset "$2" \
        --tool "$1" \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN
}


## Double removal

parameter_return_inference() {
    echo "Tool: $1 - Inferring: Parameters and Returns"
    python main.py infer --dataset "$2" \
        --tool "$1" \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN
}


variable_return_inference() {
    echo "Tool: $1 - Inferring: Variables and Returns"
    python main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN
}


variable_parameter_inference() {
    echo "Tool: $1 - Inferring: Variables and Params"
    python main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER
}

## Triple

variable_parameter_return_inference() {
    echo "Tool: $1 - Inferring: Variables, Parameters and Returns"
    python main.py infer --dataset "$2" \
        --tool "$1" \
        --remove VARIABLE --infer VARIABLE \
        --remove CALLABLE_PARAMETER --infer CALLABLE_PARAMETER \
        --remove CALLABLE_RETURN --infer CALLABLE_RETURN
}
