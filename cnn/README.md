Single node training: use `cifar_experiment.py`. For example:
```
python cifar_experiment.py with model=LeNet optimizer=SGD lr_decay=True weight_decay=True
```

Distributed training: we use Ray for CIFAR10 experiments. Instructions are in the README of the
`../learning_transforms` directory.

We modify [fastAI code](https://github.com/fastai/imagenet-fast/tree/master/imagenet_nv) for the ImageNet experiments. To run:
```
python -m multiproc imagenet_experiment.py --lr 0.4 --epochs 45 --small [DIR FOR IMAGENET DATA]
```
