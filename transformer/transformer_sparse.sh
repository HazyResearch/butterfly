#!/bin/bash

# Sparse
python no_ray.py ntrials=1 density=0.0703125 structlr=1.0
python no_ray.py ntrials=1 density=0.140625 structlr=1.0
python no_ray.py ntrials=1 density=0.28125 structlr=1.0
python no_ray.py ntrials=1 density=0.5625 structlr=1.0
python no_ray.py ntrials=1 density=0.84375 structlr=1.0

sudo shutdown -h now
