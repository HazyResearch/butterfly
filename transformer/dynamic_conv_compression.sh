#!/bin/bash
# for numlayer in $(seq 2 2 6); do
#     python dynamic_conv_experiment.py with ntrials=12 n_encoder_structure_layer=$((numlayer + 1)) n_decoder_structure_layer=$numlayer &
#     sleep 3h
# done
# wait

# for numlayer in $(seq 2 2 6); do
#     python dynamic_conv_experiment.py with ntrials=12 structure_type=BBT n_encoder_structure_layer=$((numlayer + 1)) n_decoder_structure_layer=$numlayer &
#     sleep 4h
# done
# wait

# for numlayer in $(seq 2 2 6); do
#     python dynamic_conv_experiment.py with ntrials=12 structure_type=B n_encoder_structure_layer=0 n_decoder_structure_layer=$numlayer &
#     sleep 4h
# done
# wait

# for numlayer in $(seq 2 2 6); do
#     python dynamic_conv_experiment.py with ntrials=12 structure_type=B n_encoder_structure_layer=0 n_decoder_structure_layer=$numlayer structured_attention=True &
#     sleep 4h
# done
# wait

# for numlayer in $(seq 2 2 6); do
#     python dynamic_conv_experiment.py with ntrials=12 structure_type=B n_encoder_structure_layer=$((numlayer + 1)) n_decoder_structure_layer=6 structured_attention=True &
#     sleep 2h
# done
# wait

# for numlayer in $(seq 2 2 6); do
#     python dynamic_conv_experiment.py with ntrials=12 model=Transformer structure_type=B n_encoder_structure_layer=$numlayer n_decoder_structure_layer=$numlayer structured_attention=True &
#     sleep 3h
# done
# wait

# # Experiment on structured attention on Transformer
# for structure in B BBT BBTBBT; do
#     python dynamic_conv_experiment.py with ntrials=8 model=Transformer structure_type=$structure n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True &
#     sleep 2h
# done
# wait

python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=1 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=2 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
