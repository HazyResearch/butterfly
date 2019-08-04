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
# # p100-template-5
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=B nblocks=7 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q

# Structured attention with ODO parameterization
# raiders1,2,4,5
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=1 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=2 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# p100-template-2
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-4
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q

# Different constraints on the diagonal of the ODO parameterization
# p100-template-6
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOPos nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# k80-template 1
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOBnd nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# p100-template-1
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOSqr nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# p100-template-4
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOPos nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# p100-template-3
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOBnd nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# p100-template-3
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODOSqr nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True

# p100-template-1
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=8 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-2
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=10 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-5
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=12 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-6
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=14 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q


# # raiders 2,4,5
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=1 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=2 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# # p100-template-3
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# # p100-template-4
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# # k80-template-1
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=7 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q

# # p100-template-2
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODO nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=ODONorm nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True

# p100-template-2
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OBDOBTDbl nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-3
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OBDOBTDbl nblocks=1 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-4
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OBDOBTDbl nblocks=2 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-5
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OBDOBTDbl nblocks=4 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-6
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OBDOBTDbl nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-7
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OBDOBTDbl nblocks=7 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q

# Relaunch LR that compresses the dot product only
# k80-template-1
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True && python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=3 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# k80-template-2
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=1 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True && python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=2 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-5
LR nblocks 0, lr 2.5e-4, 10e-4
New LR nblocks 0, lr 2.5e-4, 5e-4
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-6
LR nblocks 1, lr 2.5e-4, 10e-4
New LR nblocks 0, lr 1e-4, 0.5e-4
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=0 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-10
LR nblocks 3, lr 2.5e-4, 10e-4
# dawn
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=LR nblocks=1 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True

# p100-template-11
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OPDO nblocks=1 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# dawn
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OPDO nblocks=2 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True
# p100-template-7
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OPDO nblocks=3 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-8
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OPDO nblocks=6 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-9
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OPDO nblocks=12 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-10
python dynamic_conv_experiment.py with ntrials=1 model=Transformer structure_type=OPDO nblocks=18 n_encoder_structure_layer=6 n_decoder_structure_layer=6 structured_attention=True; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# OOM

# ICLR experiments

# Low rank
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["LR_18", "LR_18", "LR_18", "LR_18", "LR_18", "LR_18"]' 'decoder=["LR_18", "LR_18", "LR_18", "LR_18", "LR_18", "LR_18"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["LR_36", "LR_36", "LR_36", "LR_36", "LR_36", "LR_36"]' 'decoder=["LR_36", "LR_36", "LR_36", "LR_36", "LR_36", "LR_36"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["LR_72", "LR_72", "LR_72", "LR_72", "LR_72", "LR_72"]' 'decoder=["LR_72", "LR_72", "LR_72", "LR_72", "LR_72", "LR_72"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["LR_108", "LR_108", "LR_108", "LR_108", "LR_108", "LR_108"]' 'decoder=["LR_108", "LR_108", "LR_108", "LR_108", "LR_108", "LR_108"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["D", "D", "D", "D", "D", "D"]' 'decoder=["D", "D", "D", "D", "D", "D"]'

python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["B", "B", "B", "B", "B", "B"]' 'decoder=["B", "B", "B", "B", "B", "B"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["B_1", "B_1", "B_1", "B_1", "B_1", "B_1"]' 'decoder=["B_1", "B_1", "B_1", "B_1", "B_1", "B_1"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["B_2", "B_2", "B_2", "B_2", "B_2", "B_2"]' 'decoder=["B_2", "B_2", "B_2", "B_2", "B_2", "B_2"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["B_3", "B_3", "B_3", "B_3", "B_3", "B_3"]' 'decoder=["B_3", "B_3", "B_3", "B_3", "B_3", "B_3"]'
