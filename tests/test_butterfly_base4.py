import math
import unittest

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torch_butterfly


class ButterflyBase4Test(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_butterfly_imul(self):
        batch_size = 10
        device = 'cpu'
        for in_size, out_size in [(7, 15), (15, 7)]:
            for complex in [False, True]:
                for increasing_stride in [True, False]:
                    for init in ['randn', 'ortho', 'identity']:
                        for nblocks in [1, 2, 3]:
                            for scale in [0.13, 2.75]:
                                b = torch_butterfly.ButterflyBase4(in_size, out_size, False,
                                                                   complex, increasing_stride,
                                                                   init, nblocks=nblocks).to(device)
                                dtype = torch.float32 if not complex else torch.complex64
                                input = torch.randn(batch_size, in_size, dtype=dtype, device=device)
                                output = b(input)
                                with torch.no_grad():
                                    b *= scale
                                output_scaled = b(input)
                                self.assertTrue(torch.allclose(output * scale, output_scaled,
                                                               self.rtol, self.atol),
                                                (output.shape, device, (in_size, out_size), complex, init, nblocks))

if __name__ == "__main__":
    unittest.main()
