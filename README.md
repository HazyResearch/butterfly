Code to accompany the paper <a href="https://arxiv.org/abs/1903.05895">Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations.

## Requirements
Python 3.6+  
PyTorch >=1.1  
Numpy

## Usage

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


