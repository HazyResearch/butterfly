See `run_exp.sh` for examples of how to run the experiments.
`run_exp.sh` also produces the numbers for recovering various transforms.
Large dimensions might take a while.
We use Hyperband to do hyperparameter tuning (e.g., learning rates) on 8 GPUs with early stopping, but I
suppose hand-tuning the hyperparameters can make it faster and require less
resource (at the cost of your time).

`inference.py` shows how to implement the BP fast multiplication given the
parameters of the BP model.

`speed_test.py` runs the speed comparison between BP, FFT,
DCT, DST, and dense matrix-vector multiply, and `speed_plot.py` plots the results.

To compile Cython extension (only necessary for speed benchmark): `python setup.py build_ext --inplace`

## Distributed training (experimental)
We use Ray for distributed training. Single node training works, but using
multiple nodes can make hyperparameter tuning faster.

For single node training, see `run_exp.sh`.

For distributed training:
1. Fill in the host names of the workers in `../ray.sh`.
2. Start the cluster with ray:
```
cd ../  # Must be in the learning-circuits directory
./ray.sh start
```
3. Run the training job as usual, e.g. `python learning_transforms.py with target=dft model=BP size=8`.
4. When you're done with the cluster, shut it down:
```
cd ../  # Must be in the learning-circuits directory
./ray.sh stop
```

There are some quirks to distributed training:
- If you change the code in the `../butterfly` directory, the workers might not
  see it since they keep a cached version (I think). So you might need to shut
  down and restart the cluster (i.e. `./ray.sh stop`, then `/.ray.sh start`).
- The mode where all trials are stopped early if there's one trial getting loss
  below 1e-8 sometimes hangs when training on CUDA. I don't know why.

