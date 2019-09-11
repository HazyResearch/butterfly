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
# p100-template-7
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["B_2", "B_2", "B_2", "B_2", "B_2", "B_2"]' 'decoder=["B_2", "B_2", "B_2", "B_2", "B_2", "B_2"]'; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-6
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["B_3", "B_3", "B_3", "B_3", "B_3", "B_3"]' 'decoder=["B_3", "B_3", "B_3", "B_3", "B_3", "B_3"]'; gcloud compute instances stop $(hostname) --zone us-west1-b -q

python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_1", "ODO_1", "ODO_1", "ODO_1", "ODO_1", "ODO_1"]' 'decoder=["ODO_1", "ODO_1", "ODO_1", "ODO_1", "ODO_1", "ODO_1"]'
# p100-template-5
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_2", "ODO_2", "ODO_2", "ODO_2", "ODO_2", "ODO_2"]' 'decoder=["ODO_2", "ODO_2", "ODO_2", "ODO_2", "ODO_2", "ODO_2"]'; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-4
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]' 'decoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]'; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-3
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_8", "ODO_8", "ODO_8", "ODO_8", "ODO_8", "ODO_8"]' 'decoder=["ODO_8", "ODO_8", "ODO_8", "ODO_8", "ODO_8", "ODO_8"]'; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-2
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_12", "ODO_12", "ODO_12", "ODO_12", "ODO_12", "ODO_12"]' 'decoder=["ODO_12", "ODO_12", "ODO_12", "ODO_12", "ODO_12", "ODO_12"]'; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-1 TODO shutdown
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_14", "ODO_14", "ODO_14", "ODO_14", "ODO_14", "ODO_14"]' 'decoder=["ODO_14", "ODO_14", "ODO_14", "ODO_14", "ODO_14", "ODO_14"]'; gcloud compute instances stop $(hostname) --zone us-west1-b -q

# pip install -U ray && cd learning-circuits && git reset origin/master --hard && git pull && git checkout cnn && cd fairseq && git reset origin/master --hard && git pull && git checkout butterfly_transformer && cd ../butterfly/factor_multiply && python setup.py install && cd ../factor_multiply_fast && python setup.py install && cd ../../transformer

python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_4_2.0", "ODO_4_2.0", "ODO_4_2.0", "ODO_4_2.0", "ODO_4_2.0", "ODO_4_2.0"]' 'decoder=["ODO_4_2.0", "ODO_4_2.0", "ODO_4_2.0", "ODO_4_2.0", "ODO_4_2.0", "ODO_4_2.0"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_4_4.0", "ODO_4_4.0", "ODO_4_4.0", "ODO_4_4.0", "ODO_4_4.0", "ODO_4_4.0"]' 'decoder=["ODO_4_4.0", "ODO_4_4.0", "ODO_4_4.0", "ODO_4_4.0", "ODO_4_4.0", "ODO_4_4.0"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_4_0.5", "ODO_4_0.5", "ODO_4_0.5", "ODO_4_0.5", "ODO_4_0.5", "ODO_4_0.5"]' 'decoder=["ODO_4_0.5", "ODO_4_0.5", "ODO_4_0.5", "ODO_4_0.5", "ODO_4_0.5", "ODO_4_0.5"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_4_0.25", "ODO_4_0.25", "ODO_4_0.25", "ODO_4_0.25", "ODO_4_0.25", "ODO_4_0.25"]' 'decoder=["ODO_4_0.25", "ODO_4_0.25", "ODO_4_0.25", "ODO_4_0.25", "ODO_4_0.25", "ODO_4_0.25"]'

# cd learning-circuits && git pull && cd fairseq && git pull && cd ../transformer

# p100-template-1
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]' 'decoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]' structure_lr_multiplier=2.0; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-2
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]' 'decoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]' structure_lr_multiplier=4.0; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-3
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]' 'decoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]' structure_lr_multiplier=0.5; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-4
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]' 'decoder=["ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4", "ODO_4"]' structure_lr_multiplier=0.25; gcloud compute instances stop $(hostname) --zone us-west1-b -q

python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_1", "ODO_1", "ODO_1", "ODO_1", "ODO_1", "ODO_1"]' 'decoder=["ODO_1", "ODO_1", "ODO_1", "ODO_1", "ODO_1", "ODO_1"]' structure_lr_multiplier=4.0; gcloud compute instances stop $(hostname) --zone us-west1-b -q
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_2", "ODO_2", "ODO_2", "ODO_2", "ODO_2", "ODO_2"]' 'decoder=["ODO_2", "ODO_2", "ODO_2", "ODO_2", "ODO_2", "ODO_2"]' structure_lr_multiplier=4.0; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-3
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_8", "ODO_8", "ODO_8", "ODO_8", "ODO_8", "ODO_8"]' 'decoder=["ODO_8", "ODO_8", "ODO_8", "ODO_8", "ODO_8", "ODO_8"]' structure_lr_multiplier=4.0; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-2
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_12", "ODO_12", "ODO_12", "ODO_12", "ODO_12", "ODO_12"]' 'decoder=["ODO_12", "ODO_12", "ODO_12", "ODO_12", "ODO_12", "ODO_12"]' structure_lr_multiplier=4.0; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-1
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["ODO_14", "ODO_14", "ODO_14", "ODO_14", "ODO_14", "ODO_14"]' 'decoder=["ODO_14", "ODO_14", "ODO_14", "ODO_14", "ODO_14", "ODO_14"]' structure_lr_multiplier=4.0; gcloud compute instances stop $(hostname) --zone us-west1-b -q

# Low rank
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["LR_9", "LR_9", "LR_9", "LR_9", "LR_9", "LR_9"]' 'decoder=["LR_9", "LR_9", "LR_9", "LR_9", "LR_9", "LR_9"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["LR_18", "LR_18", "LR_18", "LR_18", "LR_18", "LR_18"]' 'decoder=["LR_18", "LR_18", "LR_18", "LR_18", "LR_18", "LR_18"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["LR_36", "LR_36", "LR_36", "LR_36", "LR_36", "LR_36"]' 'decoder=["LR_36", "LR_36", "LR_36", "LR_36", "LR_36", "LR_36"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["LR_72", "LR_72", "LR_72", "LR_72", "LR_72", "LR_72"]' 'decoder=["LR_72", "LR_72", "LR_72", "LR_72", "LR_72", "LR_72"]'
python dynamic_conv_experiment.py with ntrials=1 model=Transformer 'encoder=["LR_108", "LR_108", "LR_108", "LR_108", "LR_108", "LR_108"]' 'decoder=["LR_108", "LR_108", "LR_108", "LR_108", "LR_108", "LR_108"]'

# Sparse
python dynamic_conv_experiment.py with ntrials=1 model=TransformerBasic density=0.0703125
python dynamic_conv_experiment.py with ntrials=1 model=TransformerBasic density=0.140625
python dynamic_conv_experiment.py with ntrials=1 model=TransformerBasic density=0.28125
python dynamic_conv_experiment.py with ntrials=1 model=TransformerBasic density=0.5625
python dynamic_conv_experiment.py with ntrials=1 model=TransformerBasic density=0.84375
