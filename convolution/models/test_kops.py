import unittest

import torch
import torch.nn as nn

from kops import KOP2d


class KOP2dTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-4
        self.atol = 1e-5

    def test_fft_init(self):
        batch_size = 10
        in_ch, out_ch = 3, 6
        for in_size in [(32, 32), (16, 16), (32, 16), (16, 32)]:
            for nblocks in [1, 2, 3]:
                for base in [2, 4]:
                    kop = KOP2d(in_size, in_ch, out_ch, 5, init='fft', nblocks=nblocks, base=base)
                    x = torch.randn(batch_size, in_ch, *in_size)
                    conv = nn.Conv2d(in_ch, out_ch, 5, padding=2, padding_mode='circular',
                                     bias=False)
                    with torch.no_grad():
                        conv.weight.copy_(kop.weight.flip([-1, -2]))
                    self.assertTrue(torch.allclose(kop(x), conv(x), self.rtol, self.atol))


if __name__ == "__main__":
    unittest.main()
