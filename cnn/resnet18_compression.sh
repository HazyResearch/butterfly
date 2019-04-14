#!/bin/bash
for numlayer in $(seq 0 4); do
    python cifar_experiment.py with sgd model=ResNet18 model_args.num_structured_layers=$numlayer &
    sleep 10m
done
wait
for numlayer in $(seq 0 4); do
    python cifar_experiment.py with model=ResNet18 model_args.num_structured_layers=$numlayer optimizer=Adam lr_decay=True weight_decay=True &
    sleep 40m
done
wait
