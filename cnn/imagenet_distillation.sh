result_dir=hyperband_results_imagenet
ntrials=20
nmaxepochs=5
max_lr=2e-1
min_lr=5e-4

#for nblocks in $(seq 0 2); do
#    python distill_experiment.py with model_args.layer=features.6.0.conv1 max_lr=$max_lr min_lr=$min_lr ntrials=$ntrials nmaxepochs=$nmaxepochs optimizer=Adam model_args.structure_type=B model_args.nblocks=$nblocks dataset=imagenet teacher_model=resnet18 result_dir=$result_dir train_dir=/distillation/imagenet/activations

#for nblocks in $(seq 0 2); do
 #   python distill_experiment.py with model_args.layer=features.6.0.conv1 max_lr=$max_lr min_lr=$min_lr ntrials=$ntrials nmaxepochs=$nmaxepochs optimizer=Adam model_args.structure_type=LR model_args.nblocks=$nblocks dataset=imagenet teacher_model=resnet18 result_dir=$result_dir train_dir=/distillation/imagenet/activations

for rank in $(seq 2 2 10); do
   python distill_experiment.py with model_args.layer=features.6.0.conv1 max_lr=$max_lr min_lr=$min_lr ntrials=$ntrials nmaxepochs=$nmaxepochs optimizer=Adam model_args.structure_type=LR model_args.rank=$rank dataset=imagenet teacher_model=resnet18 result_dir=$result_dir train_dir=/distillation/imagenet/activations

#for nblocks in $(seq 0 2); do
#    python distill_experiment.py with model_args.param=odo model_args.layer=features.6.0.conv1 max_lr=$max_lr min_lr=$min_lr ntrials=$ntrials nmaxepochs=$nmaxepochs optimizer=Adam model_args.structure_type=B model_args.nblocks=$nblocks dataset=imagenet teacher_model=resnet18 result_dir=$result_dir train_dir=/distillation/imagenet/activations
#done

#for exp in $(seq 6 6 18); do
#    python distill_experiment.py with model_args.nblocks=0 model_args.param=odo model_args.tied_weight=1 model_args.expansion=$exp model_args.diag_init=normal model_args.layer=features.6.0.conv1 max_lr=$max_lr min_lr=$min_lr ntrials=$ntrials nmaxepochs=$nmaxepochs optimizer=Adam model_args.structure_type=B dataset=imagenet teacher_model=resnet18 result_dir=$result_dir train_dir=/distillation/imagenet/activations
#

done

gcloud compute instances stop $(hostname) --zone us-west1-b -q
