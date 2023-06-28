#!/usr/bin/env bash

rsync -az -P --info=progress2 --no-i-r \
		container@pvs-hyper.ifi.uni-heidelberg.de:/home/container/mdti4py/BetterTypes4Py \
		experiments/
