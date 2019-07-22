import numpy as np
import torch

from butterfly.butterfly import Butterfly
from butterfly.projection import *


if __name__ == '__main__':
    n = 8
    matrix_type = 'butterfly'
    np.random.seed(1)
    torch.manual_seed(1)

    if matrix_type == 'butterfly':
        b = Butterfly(n, n, bias=False, tied_weight=False, increasing_stride=False)
        M = b(torch.eye(n)).detach().cpu().numpy()
    elif matrix_type == 'random':
        M = np.random.randn(n, n)
    else:
        raise ValueError('Unknown matrix type')
