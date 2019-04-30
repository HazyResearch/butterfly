#!/bin/bash
# for numlayer in $(seq 2 2 6); do
#     python dynamic_conv_experiment.py with ntrials=12 n_encoder_structure_layer=$((numlayer + 1)) n_decoder_structure_layer=$numlayer &
#     sleep 3h
# done
# wait

for numlayer in $(seq 2 2 6); do
    python dynamic_conv_experiment.py with ntrials=12 structure_type=BBT n_encoder_structure_layer=$((numlayer + 1)) n_decoder_structure_layer=$numlayer &
    sleep 4h
done
wait
