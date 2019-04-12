#!/bin/bash
for numlayer in $(seq 0 12); do
    python cifar_experiment.py with sgd model=MobileNet model_args.num_structured_layers=$numlayer model_args.structure_type=Butterfly &
    sleep 10m
done
wait
for numlayer in $(seq 0 13); do
    python cifar_experiment.py with sgd model=MobileNet model_args.num_structured_layers=$numlayer model_args.structure_type=Circulant &
    sleep 10m
done
wait
for numlayer in $(seq 0 13); do
    python cifar_experiment.py with sgd model=MobileNet model_args.num_structured_layers=$numlayer model_args.structure_type=Toeplitzlike &
    sleep 20m
done
wait
