# import math
import unittest

# import numpy as np

import torch
import torch_butterfly


class ButterflySpecialTest(unittest.TestCase):

    def test_fft(self):
        batch_size = 10
        n = 16
        for normalized in [False, True]:
            for br_first in [True, False]:
                input = torch.randn(batch_size, n, dtype=torch.complex64)
                b = torch_butterfly.special.fft(n, normalized=normalized, br_first=br_first)
                out = b(input)
                out_torch = torch.view_as_complex(torch.fft(torch.view_as_real(input), signal_ndim=1, normalized=normalized))
                self.assertTrue(torch.allclose(out, out_torch))

    def test_ifft(self):
        batch_size = 10
        n = 16
        for normalized in [False, True]:
            for br_first in [True, False]:
                input = torch.randn(batch_size, n, dtype=torch.complex64)
                b = torch_butterfly.special.ifft(n, normalized=normalized, br_first=br_first)
                out = b(input)
                out_torch = torch.view_as_complex(torch.ifft(torch.view_as_real(input), signal_ndim=1, normalized=normalized))
                self.assertTrue(torch.allclose(out, out_torch))


if __name__ == "__main__":
    unittest.main()
