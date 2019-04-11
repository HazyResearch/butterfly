#!/bin/bash
# DFT
python learning_transforms.py with target=dft model=BP size=8
python learning_transforms.py with target=dft model=BP size=16
python learning_transforms.py with target=dft model=BP size=32 ntrials=40
python learning_transforms.py with target=dft model=BP size=64 ntrials=60
python learning_transforms.py with target=dft model=BP size=128 ntrials=80
python learning_transforms.py with target=dft model=BP size=256 ntrials=160
python learning_transforms.py with target=dft model=BP size=512 ntrials=160
python learning_transforms.py with target=dft model=BP size=1024 ntrials=320

# DCT - BPP
python learning_transforms.py with target=dct model=BPP size=8
python learning_transforms.py with target=dct model=BPP size=16
python learning_transforms.py with target=dct model=BPP size=32 ntrials=40
python learning_transforms.py with target=dct model=BPP size=64 ntrials=60
python learning_transforms.py with target=dct model=BPP size=128 ntrials=80
python learning_transforms.py with target=dct model=BPP size=256 ntrials=160
python learning_transforms.py with target=dct model=BPP size=512 ntrials=160
python learning_transforms.py with target=dct model=BPP size=1024 ntrials=320

# DST - BPP
python learning_transforms.py with target=dst model=BPP size=8
python learning_transforms.py with target=dst model=BPP size=16
python learning_transforms.py with target=dst model=BPP size=32 ntrials=40
python learning_transforms.py with target=dst model=BPP size=64 ntrials=60
python learning_transforms.py with target=dst model=BPP size=128 ntrials=80
python learning_transforms.py with target=dst model=BPP size=256 ntrials=160
python learning_transforms.py with target=dst model=BPP size=512 ntrials=160
python learning_transforms.py with target=dst model=BPP size=1024 ntrials=320

# Hadamard
python learning_transforms.py with target=hadamard model=BP size=8
python learning_transforms.py with target=hadamard model=BP size=16
python learning_transforms.py with target=hadamard model=BP size=32 ntrials=40
python learning_transforms.py with target=hadamard model=BP size=64 ntrials=60
python learning_transforms.py with target=hadamard model=BP size=128 ntrials=80
python learning_transforms.py with target=hadamard model=BP size=256 ntrials=160
python learning_transforms.py with target=hadamard model=BP size=512 ntrials=160
python learning_transforms.py with target=hadamard model=BP size=1024 ntrials=320

# Hartley
python learning_transforms.py with target=hartley model=BP size=8
python learning_transforms.py with target=hartley model=BP size=16
python learning_transforms.py with target=hartley model=BP size=32 ntrials=40
python learning_transforms.py with target=hartley model=BP size=64 ntrials=60
python learning_transforms.py with target=hartley model=BP size=128 ntrials=80
python learning_transforms.py with target=hartley model=BP size=256 ntrials=160
python learning_transforms.py with target=hartley model=BP size=512 ntrials=160
python learning_transforms.py with target=hartley model=BP size=1024 ntrials=320

# Convolution
python learning_transforms.py with target=convolution model=BPBP size=8
python learning_transforms.py with target=convolution model=BPBP size=16
python learning_transforms.py with target=convolution model=BPBP size=32 ntrials=40
python learning_transforms.py with target=convolution model=BPBP size=64 ntrials=60
python learning_transforms.py with target=convolution model=BPBP size=128 ntrials=80
python learning_transforms.py with target=convolution model=BPBP size=256 ntrials=160
python learning_transforms.py with target=convolution model=BPBP size=512 ntrials=160
python learning_transforms.py with target=convolution model=BPBP size=1024 ntrials=320

# Randn
python learning_transforms.py with target=randn model=BP size=8
python learning_transforms.py with target=randn model=BP size=16
python learning_transforms.py with target=randn model=BP size=32 ntrials=40
python learning_transforms.py with target=randn model=BP size=64 ntrials=60
python learning_transforms.py with target=randn model=BP size=128 ntrials=80
python learning_transforms.py with target=randn model=BP size=256 ntrials=160
python learning_transforms.py with target=randn model=BP size=512 ntrials=160

# Legendre
python learning_transforms.py with target=legendre model=BP size=8
python learning_transforms.py with target=legendre model=BP size=16
python learning_transforms.py with target=legendre model=BP size=32 ntrials=40
python learning_transforms.py with target=legendre model=BP size=64 ntrials=60
python learning_transforms.py with target=legendre model=BP size=128 ntrials=80
python learning_transforms.py with target=legendre model=BP size=256 ntrials=160
python learning_transforms.py with target=legendre model=BP size=512 ntrials=160
python learning_transforms.py with target=legendre model=BP size=1024 ntrials=320
