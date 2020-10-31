python train.py wandb.group=cnn5 model=cnn5 train.optimizer.lr=0.01 '+train.lr_scheduler={_target_:lr_schedulers.LeNetScheduler,nepochs:100}' train.optimizer.weight_decay=1e-4 runner.ntrials=3 runner.gpu_per_trial=0.5

python train.py wandb.group=cnn5 model._target_=models.CNN5Butterfly '+model.nblocks=[_grid,1,2,3,4,5,6]' +model.complex=True '+model.base=[_grid,2,4]' +model.init=fft train.optimizer.lr=0.01 '+train.lr_scheduler={_target_:lr_schedulers.LeNetScheduler,nepochs:100}' train.optimizer.weight_decay=1e-4 runner.ntrials=3 runner.gpu_per_trial=0.5

python train.py wandb.group=cnn5 model._target_=models.CNN5Butterfly '+model.nblocks=[_grid,1,2,3,4,5,6]' '+model.complex=[_grid,True,False]' '+model.base=[_grid,2,4]' +model.init=ortho train.optimizer.lr=0.01 '+train.lr_scheduler={_target_:lr_schedulers.LeNetScheduler,nepochs:100}' train.optimizer.weight_decay=1e-4 runner.ntrials=3 runner.gpu_per_trial=0.5

