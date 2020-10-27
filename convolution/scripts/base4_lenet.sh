#!/bin/bash
for complex in True False; do
    for base in 4 2; do
        for nblocks in 1 2 3 4 5 6; do
            python train.py wandb.group=base4-lenet model._target_=models.ButterfLeNet +model.nblocks=$nblocks +model.complex=$complex +model.base=$base 'train.optimizer.lr=[_sample_log_uniform,5e-3,2e-1]' 'train.optimizer.weight_decay=[_sample_log_uniform,1e-5,5e-4]' runner.hyperband=True runner.ntrials=30 runner.gpu_per_trial=0.3 &
            sleep 15m
        done
        wait
    done
done

python train.py wandb.group=base4-lenet model._target_=models.ButterfLeNet '+model.nblocks=[_grid,1,2,3,4,5,6]' +model.complex=True '+model.base=[_grid,2,4]' +model.init=fft train.optimizer.lr=0.01 '+train.lr_scheduler={_target_:lr_schedulers.LeNetScheduler,nepochs:100}' train.optimizer.weight_decay=1e-4 runner.ntrials=3 runner.gpu_per_trial=0.3

python train.py wandb.group=base4-lenet model._target_=models.ButterfLeNet '+model.nblocks=[_grid,1,2,3,4,5,6]' '+model.complex=[_grid,True,False]' '+model.base=[_grid,2,4]' +model.init=ortho train.optimizer.lr=0.01 '+train.lr_scheduler={_target_:lr_schedulers.LeNetScheduler,nepochs:100}' train.optimizer.weight_decay=1e-4 runner.ntrials=3 runner.gpu_per_trial=0.3


# See if zero-padding the filter weight makes a difference
python train.py wandb.group=base4-lenet model._target_=models.ButterfLeNet '+model.nblocks=[_grid,1,2,3,4,5,6]' +model.complex=True '+model.base=[_grid,2,4]' +model.init=fft +model.zero_pad=False train.optimizer.lr=0.01 '+train.lr_scheduler={_target_:lr_schedulers.LeNetScheduler,nepochs:100}' train.optimizer.weight_decay=1e-4 runner.ntrials=3 runner.gpu_per_trial=0.3

python train.py wandb.group=base4-lenet model._target_=models.ButterfLeNet '+model.nblocks=[_grid,1,2,3,4,5,6]' '+model.complex=[_grid,True,False]' '+model.base=[_grid,2,4]' +model.init=ortho +model.zero_pad=False train.optimizer.lr=0.01 '+train.lr_scheduler={_target_:lr_schedulers.LeNetScheduler,nepochs:100}' train.optimizer.weight_decay=1e-4 runner.ntrials=3 runner.gpu_per_trial=0.3

# Train for 200 epochs instead of 100
python train.py wandb.group=base4-lenet model._target_=models.ButterfLeNet '+model.nblocks=[_grid,1,2,3,4,5,6]' +model.complex=True '+model.base=[_grid,2,4]' +model.init=fft train.epochs=200 train.optimizer.lr=0.01 '+train.lr_scheduler={_target_:lr_schedulers.LeNetScheduler,nepochs:200}' train.optimizer.weight_decay=1e-4 runner.ntrials=3 runner.gpu_per_trial=0.3

python train.py wandb.group=base4-lenet model._target_=models.ButterfLeNet '+model.nblocks=[_grid,1,2,3,4,5,6]' '+model.complex=[_grid,True,False]' '+model.base=[_grid,2,4]' +model.init=ortho train.epochs=200 train.optimizer.lr=0.01 '+train.lr_scheduler={_target_:lr_schedulers.LeNetScheduler,nepochs:200}' train.optimizer.weight_decay=1e-4 runner.ntrials=3 runner.gpu_per_trial=0.3
