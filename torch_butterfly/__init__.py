import importlib
from pathlib import Path

import torch

__version__ = '0.0.0'

for library in ['_version', '_butterfly']:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        # need str(Path) otherwise it can't find it
        library, [str(Path(__file__).absolute().parent)]).origin)

def check_cuda_version():
    if torch.version.cuda is not None:  # pragma: no cover
        cuda_version = torch.ops.torch_butterfly.cuda_version()

        if cuda_version == -1:
            major = minor = 0
        elif cuda_version < 10000:
            major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
        else:
            major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
        t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

        if t_major != major or t_minor != minor:
            raise RuntimeError(
                f'Detected that PyTorch and torch_butterfly were compiled with '
                f'different CUDA versions. PyTorch has CUDA version '
                f'{t_major}.{t_minor} and torch_butterfly has CUDA version '
                f'{major}.{minor}. Please reinstall the torch_butterfly that '
                f'matches your PyTorch install.')

check_cuda_version()
from .butterfly import Butterfly  # noqa
from .multiply import butterfly_multiply  # noqa
from . import special

__all__ = [
    'Butterfly',
    'butterfly_multiply',
    '__version__',
]
