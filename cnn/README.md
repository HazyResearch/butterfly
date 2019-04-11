Single node training: use `cifar_experiment.py`. For example:
```
python cifar_experiment.py with model=LeNet optimizer=SGD lr_decay=True weight_decay=True
```

Distributed training: we use Ray. Instructions are in the README of the
`../learning_transforms` directory.
