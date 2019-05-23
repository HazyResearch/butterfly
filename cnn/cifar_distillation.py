import os

layers=[
#    "layer4.1.conv2",
#    "layer4.1.conv1",
#    "layer4.0.shortcut.0",
#    "layer4.0.conv2",
#    "layer4.0.conv1",
    "layer3.1.conv2",
    "layer3.1.conv1",
    "layer3.0.shortcut.0",
    "layer3.0.conv2",
    "layer3.0.conv1",
#    "layer2.1.conv2",
#    "layer2.1.conv1",
#    "layer2.0.shortcut.0",
#    "layer2.0.conv2",
#    "layer2.0.conv1",
#    "layer1.1.conv2",
#    "layer1.1.conv1",
#    "layer1.0.conv2",
#    "layer1.0.conv1"
]

qsub_cmd = "qsub -V -r y -j y -b y -wd /home/mleszczy/learning-circuits/cnn"
resultdir = "/home/mleszczy/cifar_sweep/results_hyperband_variants"
traindir = "/distillation/cifar10/activations"
print(f"mkdir -p {resultdir}")

min_lr =1e-3
max_lr = 5e-1
grace_period=3
ntrials=20
nmaxepochs=10

#for layer in layers:
    # butterflies
#    for nblocks in range(0,6):
#        print(f"{qsub_cmd} -N {layer}_B_{nblocks} -o {resultdir}/{layer}_B_{nblocks}.log python distill_experiment.py with model_args.layer={layer} ntrials=50 nmaxepochs=20 max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=B model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")
#
#    # low rank
    #for nblocks in range(0,6):
     #   print(f"{qsub_cmd} -N {layer}_LR_{nblocks} -o {resultdir}/{layer}_LR_{nblocks}.log  python distill_experiment.py with model_args.layer={layer} ntrials=12 nmaxepochs=20 max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=LR model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

#    for rank in range(1,11):
#        print(f"{qsub_cmd} -N {layer}_LR_rank_{rank} -o {resultdir}/{layer}_LR_rank_{rank}.log  python distill_experiment.py with model_args.rank={rank} model_args.layer={layer} ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=LR dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

#    for nblocks in range(0,1):
#        print(f"{qsub_cmd} -N {layer}_B_{nblocks} -o {resultdir}/{layer}_B_{nblocks}.log python distill_experiment.py with model_args.layer={layer} model_args.param=obdobt ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=B model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

#    for nblocks in range(0,1):
#        print(f"{qsub_cmd} -N {layer}_B_{nblocks} -o {resultdir}/{layer}_B_{nblocks}.log python distill_experiment.py with model_args.layer={layer} model_args.param=svd ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=B model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

#    for nblocks in range(0,1):
#        print(f"{qsub_cmd} -N {layer}_B_{nblocks} -o {resultdir}/{layer}_B_{nblocks}.log python distill_experiment.py with model_args.layer={layer} model_args.param=ortho ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=B model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")
#    for nblocks in range(0,6):
#        print(f"{qsub_cmd} -N {layer}_B_{nblocks} -o {resultdir}/{layer}_B_{nblocks}.log python distill_experiment.py with model_args.layer={layer} model_args.param=odo ntrials=50 nmaxepochs=20 max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=B model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

#    for exp in range(6, 42, 6):
#        print(f"{qsub_cmd} -N {layer}_B_exp_{exp} -o {resultdir}/{layer}_B_exp_{exp}.log python distill_experiment.py with grace_period={grace_period} model_args.nblocks=0 model_args.param=odo model_args.tied_weight=1 model_args.expansion={exp} model_args.diag_init=normal model_args.layer={layer} ntrials=50 nmaxepochs=20 max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

#    for exp in range(6, 18, 6):
#        print(f"{qsub_cmd} -N {layer}_B_exp_{exp} -o {resultdir}/{layer}_B_exp_{exp}.log python distill_experiment.py with grace_period=1 model_args.nblocks=0 model_args.param=odo model_args.tied_weight=1 model_args.expansion={exp} model_args.diag_init=normal model_args.layer={layer} ntrials=12 nmaxepochs=5 max_lr=20000 min_lr=5000 optimizer=SGD model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

#    for exp in range(6, 12, 6):
#        print(f"{qsub_cmd} -N {layer}_B_exp_{exp} -o {resultdir}/{layer}_B_exp_{exp}.log python distill_experiment.py with grace_period=1 model_args.nblocks=0 model_args.param=odo model_args.tied_weight=1 model_args.expansion={exp} model_args.diag_init=normal model_args.layer={layer} ntrials=12 nmaxepochs=5 max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.ortho_init=0 model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

#    for nblocks in range(0,6):
#        print(f"{qsub_cmd} -N {layer}_B_{nblocks} -o {resultdir}/{layer}_B_{nblocks}.log python distill_experiment.py with model_args.layer={layer} model_args.param=odo ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} model_args.tied_weight=1 optimizer=Adam model_args.structure_type=B model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")
#
#    for nblocks in range(0,6):
#        print(f"{qsub_cmd} -N {layer}_B_{nblocks} -o {resultdir}/{layer}_B_{nblocks}.log python distill_experiment.py with model_args.layer={layer} model_args.param=regular ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} model_args.tied_weight=1 optimizer=Adam model_args.structure_type=B model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

    # diag init != normal
#    for exp in range(6, 18, 6):
#        print(f"{qsub_cmd} -N {layer}_B_exp_{exp} -o {resultdir}/{layer}_B_exp_{exp}.log python distill_experiment.py with model_args.nblocks=0 model_args.param=odo model_args.tied_weight=1 model_args.expansion={exp} model_args.layer={layer} ntrials=12 nmaxepochs=5 max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

    # block + expansion
#    for exp in range(6, 18, 6):
#        print(f"{qsub_cmd} -N {layer}_B_exp_{exp} -o {resultdir}/{layer}_B_exp_{exp}.log python distill_experiment.py with model_args.nblocks=1 model_args.diag_init=normal model_args.param=odo model_args.tied_weight=1 model_args.expansion={exp} model_args.layer={layer} ntrials=12 nmaxepochs=5 max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

    # 5 / 19
    # variant 1
#    for exp in range(4, 8):
#        print(f"{qsub_cmd} -N {layer}_B_exp_{exp} -o {resultdir}/{layer}_B_exp_{exp}.log python distill_experiment.py with model_args.nblocks=0 model_args.diag_init=normal model_args.param=odo model_args.tied_weight=1 model_args.expansion={exp} model_args.layer={layer} ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} grace_period={grace_period} optimizer=Adam model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

    # variant 2
#    for exp in range(1, 4):
#        print(f"{qsub_cmd} -N {layer}_B_exp_{exp} -o {resultdir}/{layer}_B_exp_{exp}.log python distill_experiment.py with model_args.nblocks=0 model_args.diag_init=normal model_args.param=odr model_args.tied_weight=1 model_args.expansion={exp} model_args.layer={layer} ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} grace_period={grace_period} optimizer=Adam model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

    # variant 3
#    for exp in range(1, 11):
#        print(f"{qsub_cmd} -N {layer}_B_exp_{exp} -o {resultdir}/{layer}_B_exp_{exp}.log python distill_experiment.py with model_args.nblocks=0 model_args.diag_init=normal model_args.param=opdo model_args.tied_weight=1 model_args.expansion={exp} model_args.layer={layer} ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} grace_period={grace_period} optimizer=Adam model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

    # variant 4
#    for nblocks in range(4, 8):
#        print(f"{qsub_cmd} -N {layer}_B_exp_{exp} -o {resultdir}/{layer}_B_exp_{exp}.log python distill_experiment.py with model_args.nblocks={nblocks} model_args.diag_init=normal model_args.param=odo model_args.tied_weight=1 model_args.expansion=1 model_args.layer={layer} ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr={max_lr} min_lr={min_lr} grace_period={grace_period} optimizer=Adam model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")

# sparsity
layer_param = {
          'layer3.1.conv1': [13788, 20682, 27576, 34470, 41364, 48258, 55152, 62046, 68940],
          'layer3.1.conv2': [13788, 20682, 27576, 34470, 41364, 48258, 55152, 62046, 68940],
          'layer3.0.conv1': [13788, 20682, 27576, 34470, 41364, 48258, 55152, 62046, 68940],
          'layer3.0.conv2': [13788, 20682, 27576, 34470, 41364, 48258, 55152, 62046, 68940],
          'layer3.0.shortcut.0': [1528, 2292, 3056, 3820, 4584, 5348, 6112, 6876, 7640]

}

for layer in layer_param:
    for nparams in layer_param[layer]:
        print(f"{qsub_cmd} -N {layer}_sparse_{nparams} -o {resultdir}/{layer}_sparse_{nparams}.log python distill_experiment.py with model_args.structure_type=sparse pretrained=cifar10_resnet18.pth.tar model_args.nparams={nparams} model_args.layer={layer} ntrials={ntrials} nmaxepochs={nmaxepochs} max_lr=1e-3 min_lr=1e-4 grace_period={grace_period} optimizer=Adam dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir}")
