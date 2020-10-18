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
