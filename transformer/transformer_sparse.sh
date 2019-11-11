#!/bin/bash

qsub run_transformer.py model=DynamicConvBasic density=0.1 structlr=0.25
sleep 5
qsub run_transformer.py model=DynamicConvBasic density=0.1 structlr=0.5
sleep 5
qsub run_transformer.py model=DynamicConvBasic density=0.1 structlr=2.0
sleep 5
qsub run_transformer.py model=DynamicConvBasic density=0.1 structlr=4.0
#sleep 5; qsub run_transformer.py model=DynamicConvBasic density=0.1 structlr=1.0
