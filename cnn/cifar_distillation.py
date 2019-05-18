import os 

layers=[
    "layer4.1.conv2",
#    "layer4.1.conv1",
#    "layer4.0.shortcut.0",
#    "layer4.0.conv2",
#    "layer4.0.conv1",
#    "layer3.1.conv2",
#    "layer3.1.conv1",
#    "layer3.0.shortcut.0",
#    "layer3.0.conv2",
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
iters = 5
resultdir = "/home/mleszczy/cifar_sweep/results_rel_mse_small"
traindir = "/distillation/cifar10/activations"
os.makedirs(resultdir, exist_ok=True)
max_lr=200
min_lr=85

for layer in layers:
    # butterflies
    print(f"{qsub_cmd} -N {layer}_B -o {resultdir}/{layer}_B.log python distill_experiment.py with model_args.layer={layer} max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=B dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir} iters={iters}")
#    for nblocks in range(1,6):
#        print(f"{qsub_cmd} -N {layer}_BBT_{nblocks} -o {resultdir}/{layer}_BBT_{nblocks}.log python distill_experiment.py with model_args.layer={layer} max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=BBT model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir} iters={iters}")

    # low rank
#    for nblocks in range(0,1):
#        print(f"{qsub_cmd} -N {layer}_LR_{nblocks} -o {resultdir}/{layer}_LR_{nblocks}.log  python distill_experiment.py with model_args.layer={layer} max_lr={max_lr} min_lr={min_lr} optimizer=Adam model_args.structure_type=LR model_args.nblocks={nblocks} dataset=cifar10 teacher_model=ResNet18 result_dir={resultdir} train_dir={traindir} iters={iters}")
