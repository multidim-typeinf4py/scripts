#!/bin/bash

manytypes4py_repos() {
    find "$1" -maxdepth 2 -mindepth 2 -type d -print0
}

variable_inference() {
    echo "Tool: $1 - Inferring: Variables & Instance Attributes"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-var-annos
}

parameter_inference() {
    echo "Tool: $1 - Inferring: Parameters"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-param-annos
}

return_inference() {
    echo "Tool: $1 - Inferring: Returns"
    ./.venv/bin/python3 main.py infer --inpath "$2" --overwrite \
        --tool "$1" \
        --remove-ret-annos
}
