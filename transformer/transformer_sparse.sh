#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "You must enter exactly 1 command line argument."
  exit 1
fi

structlr=$1

# Sparse
qsub run_transformer.py model=DynamicConvBasic density=0.0703125 structlr=$structlr
sleep 1
qsub run_transformer.py model=DynamicConvBasic density=0.140625 structlr=$structlr
sleep 1
qsub run_transformer.py model=DynamicConvBasic density=0.28125 structlr=$structlr
sleep 1
qsub run_transformer.py model=DynamicConvBasic density=0.5625 structlr=$structlr

#python run_transformer.py density=0.84375 structlr=1.0
