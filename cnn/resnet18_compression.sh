#!/bin/bash
# for numlayer in $(seq 1 3); do
#     python cifar_experiment.py with sgd model=ResNet18 model_args.num_structured_layers=$numlayer &
#     sleep 40m
# done
# wait
# for numlayer in $(seq 0 3); do
#     python cifar_experiment.py with model=ResNet18 model_args.num_structured_layers=$numlayer optimizer=Adam lr_decay=True weight_decay=True &
#     sleep 40m
# done
# wait

for numlayer in $(seq 2 2); do
    for structure in BBT BBTBBT; do
        python cifar_experiment.py with model=ResNet18 model_args.num_structured_layers=$numlayer model_args.structure_type=$structure optimizer=Adam lr_decay=True weight_decay=True &
        sleep 3h
    done
done
wait
