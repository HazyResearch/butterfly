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

# For Google Cloud:
# cd learning-circuits && git pull && cd butterfly/factor_multiply && python setup.py install && cd ../../fairseq && git pull && cd ../transformer
# # p100-template-1
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# # p100-template-2
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=1 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# # p100-template-3
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=2 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# # p100-template-4
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# # p100-template-5
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q

# Structured attention with ODO parameterization
# k80-template-1
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# k80-template-2
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=1 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-1
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=2 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-2
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-6
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q

# Different constraints on the diagonal of the ODO parameterization
# raiders1,2,4,5
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOPos nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOBnd nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOSqr nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOPos nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# p100-template-3
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOBnd nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOSqr nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
