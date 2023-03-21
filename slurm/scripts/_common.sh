#!/bin/bash

manytypes4py_repos() {
    find "$1" -maxdepth 2 -mindepth 2 -type d -print0
}

## Single removal

parameter_inference() {
    echo "Tool: $1 - Inferring: Parameters"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-param-annos
}

variable_inference() {
    echo "Tool: $1 - Inferring: Variables"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-var-annos
}

return_inference() {
    echo "Tool: $1 - Inferring: Returns"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-ret-annos
}


## Double removal

parameter_return_inference() {
    echo "Tool: $1 - Inferring: Parameters and Returns"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-param-annos --remove-ret-annos
}


variable_return_inference() {
    echo "Tool: $1 - Inferring: Variables and Returns"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-var-annos --remove-ret-annos
}


variable_parameter_inference() {
    echo "Tool: $1 - Inferring: Variables and Params"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-var-annos --remove-param-annos
}

## Triple

variable_parameter_return_inference() {
    echo "Tool: $1 - Inferring: Variables, Parameters and Returns"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-var-annos --remove-param-annos --remove-ret-annos
}
