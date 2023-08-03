#!/usr/bin/env bash

for folder in test train valid; do
	rsync -az -P --info=progress2 --no-i-r \
		container@pvs-hyper.ifi.uni-heidelberg.de:/home/container/mdti4py/better-types-4-py-dataset/repos/${folder} \
		experiments/better-types-4-py-dataset/repos/
done	
