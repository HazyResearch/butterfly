Code to accompany the papers [Learning Fast Algorithms for Linear Transforms
Using Butterfly Factorizations](https://arxiv.org/abs/1903.05895) and [Kaleidoscope: An Efficient, Learnable Representation For All Structured Linear Maps](https://openreview.net/forum?id=BkgrBgSYDS).

## Requirements
python>=3.6  
pytorch>=1.7  
numpy  
scipy

## Usage

2020-08-03: The new interface to butterfly C++/CUDA code is in `csrc` and
`torch_butterfly`.
It is tested in `tests/test_torch_butterfly.py` (which also shows example
usage).

To install:
```
python setup.py install
```
That is, use the `setup.py` file in this root directory.

The file `torch_butterfly/special.py` shows how to construct butterfly matrices
that performs FFT, inverse FFT, circulant matrix multiplication,
Hadamard transform, and torch.nn.Conv1d with circular padding. The tests in
`tests/test_special.py` show that these butterfly matrices exactly perform
those operations.

## Old interface

Note: this interface is being rewritten. Only use this if you need some feature
that's not supported in the new interface.

* The module `Butterfly` in `butterfly/butterfly.py` can be used as a drop-in
replacement for a `nn.Linear` layer. The files in `butterfly` directory are all
that are needed for this use.

The butterfly multiplication is written in C++ and CUDA as PyTorch extension.
To install it:
```
cd butterfly/factor_multiply
python setup.py install
cd butterfly/factor_multiply_fast
python setup.py install
```
Without the C++/CUDA version, butterfly multiplication is still usable, but is
quite slow. The variable `use_extension` in `butterfly/butterfly_multiply.py`
controls whether to use the C++/CUDA version or the pure PyTorch version.

For training, we've had better results with the Adam optimizer than SGD.

* The directory `learning_transforms` contains code to learn the transforms
  as presented in the paper. This directory is presently being developed and
  refactored.


