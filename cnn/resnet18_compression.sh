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

# for numlayer in $(seq 2 2); do
#     for structure in BBT BBTBBT; do
#         python cifar_experiment.py with model=ResNet18 model_args.num_structured_layers=$numlayer model_args.structure_type=$structure optimizer=Adam lr_decay=True weight_decay=True &
#         sleep 3h
#     done
# done
# wait

# for numlayer in $(seq 2 2); do
#     for structure in B BBT BBTBBT; do
#         python cifar_experiment.py with model=ResNet18 model_args.num_structured_layers=$numlayer model_args.structure_type=$structure model_args.param='ortho' optimizer=Adam lr_decay=True weight_decay=True &
#         sleep 3h
#     done
# done
# wait

# for numlayer in $(seq 2 2); do
#     for structure in B BBT BBTBBT; do
#         python cifar_experiment.py with model=ResNet18 model_args.num_structured_layers=$numlayer model_args.structure_type=$structure model_args.param='svd' optimizer=Adam lr_decay=True weight_decay=True &
#         sleep 2h
#     done
# done
# wait

# for numlayer in $(seq 2 2); do
#     for nblocks in 3 4 5; do
#         python cifar_experiment.py with model=ResNet18 model_args.num_structured_layers=$numlayer model_args.structure_type=BBT model_args.nblocks=$nblocks model_args.param='regular' optimizer=Adam lr_decay=True weight_decay=True &
#         sleep 1.25h
#     done
# done
# sleep 1h

# for numlayer in $(seq 2 2); do
#     for nblocks in 3 4 5; do
#         python cifar_experiment.py with model=ResNet18 model_args.num_structured_layers=$numlayer model_args.structure_type=BBT model_args.nblocks=$nblocks model_args.param='svd' optimizer=Adam lr_decay=True weight_decay=True &
#         sleep 2.25h
#     done
# done
# wait

# raiders2,4,5
python cifar_experiment.py with ntrials=3 model=ResNet18 model_args.num_structured_layers=0 model_args.structure_type=B optimizer=SGD nmaxepochs=200
python cifar_experiment.py with ntrials=3 model=ResNet18 model_args.num_structured_layers=1 model_args.structure_type=B optimizer=SGD nmaxepochs=200
# p100-template-4
python cifar_experiment.py with ntrials=3 model=ResNet18 model_args.num_structured_layers=2 model_args.structure_type=B optimizer=SGD nmaxepochs=200; gcloud compute instances stop $(hostname) --zone us-west1-b -q
# p100-template-3
python cifar_experiment.py with ntrials=3 model=ResNet18 model_args.num_structured_layers=2 model_args.structure_type=BBT optimizer=SGD nmaxepochs=200; gcloud compute instances stop $(hostname) --zone us-west1-b -q
