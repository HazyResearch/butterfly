Code to accompany the paper <a href="https://arxiv.org/abs/1903.05895">Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations.

## Requirements
Python 3.6+  
PyTorch >=1.0  
Numpy

## Usage

* The module `Butterfly` in `butterfly/butterfly.py` can be used as a drop-in
replacement for a `nn.Linear` layer. The files in `butterfly` directory are all
that are needed for this use.

For training, we've had better results with the Adam optimizer than SGD.

* The directory `learning_transforms` contains code to learn the transforms
  as presented in the paper. This directory is presently being developed and
  refactored.


