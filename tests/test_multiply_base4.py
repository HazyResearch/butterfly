import math
import unittest

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torch_butterfly.multiply import butterfly_multiply_torch
from torch_butterfly.multiply_base4 import butterfly_multiply_base4_torch
from torch_butterfly.multiply_base4 import twiddle_base2_to_base4


class MultiplyBase4Test(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_multiply_base4(self):
        batch_size = 10
        nstacks = 2
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                log_n = int(math.log2(n))
                for nblocks in [1, 2, 3, 4]:
                    for complex in [False, True]:
                        for increasing_stride in [True, False]:
                            dtype = torch.float32 if not complex else torch.complex64
                            # complex randn already has the correct scaling of stddev=1.0
                            scaling = 1 / math.sqrt(2)
                            twiddle = torch.randn((nstacks, nblocks, log_n, n // 2, 2, 2),
                                                  dtype=dtype, device=device) * scaling
                            input = torch.randn((batch_size, nstacks, n), dtype=dtype,
                                                device=twiddle.device)
                            output2 = butterfly_multiply_torch(twiddle, input, increasing_stride)
                            twiddle4, twiddle2 = twiddle_base2_to_base4(twiddle, increasing_stride)
                            output4 = butterfly_multiply_base4_torch(twiddle4, twiddle2, input,
                                                                    increasing_stride)
                            self.assertTrue(torch.allclose(output2, output4,
                                                        rtol=self.rtol, atol=self.atol),
                                            ((output2 - output4).abs().max().item(),
                                            n, nblocks, complex, increasing_stride))


if __name__ == "__main__":
    unittest.main()
