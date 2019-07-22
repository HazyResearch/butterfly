import numpy as np
import torch

from butterfly.butterfly import Butterfly
from butterfly.projection import *


if __name__ == '__main__':
    n = 16
    matrix_type = 'butterfly'
    project_onto = 'B'
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    if matrix_type == 'butterfly':
        b = Butterfly(n, n, bias=False, tied_weight=False, increasing_stride=False)
        M = b(torch.eye(n)).detach().cpu().numpy()
    elif matrix_type == 'random':
        M = np.random.randn(n, n)
    else:
        raise ValueError('Unknown matrix type')
    M_tensor = torch.tensor(M).float()

    b_sos = butterfly_SOS(M, project_onto=project_onto)[0]
    sos_result = b_sos(torch.eye(n))
    print('SOS error:', torch.norm(sos_result - M_tensor))

    b_gd = butterfly_GD(M, project_onto=project_onto, print_every=100, lr=5e-2, tol=1e-4)[0].cpu()
    gd_result = b_gd(torch.eye(n))
    print('GD error:', torch.norm(gd_result - M_tensor))
