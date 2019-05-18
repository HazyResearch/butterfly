result_dir=hyperband_results_cifar10
ntrials=50
nmaxepochs=20 

for nblocks in $(seq 0 5); do 
    python distill_experiment.py with model_args.param=odo model_args.layer=layer3.0.conv1 max_lr=1 min_lr=0.0001 ntrials=$ntrials nmaxepochs=$nmaxepochs optimizer=Adam model_args.structure_type=B model_args.nblocks=$nblocks dataset=cifar10 teacher_model=ResNet18 result_dir=$result_dir train_dir=/distillation/cifar10/activations

    python distill_experiment.py with model_args.param=odo model_args.layer=layer4.1.conv2 max_lr=1 min_lr=0.0001 ntrials=$ntrials nmaxepochs=$nmaxepochs optimizer=Adam model_args.structure_type=B model_args.nblocks=$nblocks dataset=cifar10 teacher_model=ResNet18 result_dir=$result_dir train_dir=/distillation/cifar10/activations

done
gcloud compute instances stop $(hostname) --zone us-west1-b -q
