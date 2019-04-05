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

To compile Cython extension: `python setup.py build_ext --inplace`

To compile and install C++ extension for Pytorch:
```
cd butterfly/factor_multiply
python setup.py install
```
