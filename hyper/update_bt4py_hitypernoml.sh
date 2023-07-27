#!/usr/bin/env bash

rsync -az -P --info=progress2 --no-i-r  container@pvs-hyper.ifi.uni-heidelberg.de:/home/container/mdti4py/hitypernoml/BetterTypes4Py/ experiments/BetterTypes4Py
