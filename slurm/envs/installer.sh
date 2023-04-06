#!/bin/bash

if [[ -f "$1/setup.py" ]]; then
    (cd $1 && python -m venv .venv && source .venv/bin/activate && pip install .)
else
    echo "$1 no setup.py found!"
fi


