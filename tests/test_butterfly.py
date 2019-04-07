import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import numpy as np

import torch

from butterfly import Butterfly

class ButterflyTest(unittest.TestCase):

    def test_butterfly(self):
        batch_size = 10
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for in_size, out_size in [(7, 15), (15, 7)]:
                for complex in [False, True]:
                    for tied_weight in [True, False]:
                        for ortho_init in [False, True]:
                            b = Butterfly(in_size, out_size, True, complex, tied_weight, ortho_init).to(device)
                            input = torch.randn((batch_size, in_size) + (() if not complex else (2,)), device=device)
                            output = b(input)
                            self.assertTrue(output.shape == (batch_size, out_size) + (() if not complex else (2,)),
                                            (output.shape, device, (in_size, out_size), complex, tied_weight, ortho_init))
                            if ortho_init:
                                twiddle_np = b.twiddle.detach().to('cpu').numpy()
                                if complex:
                                    twiddle_np = twiddle_np.view('complex64').squeeze(-1)
                                twiddle_np = twiddle_np.reshape(-1, 2, 2)
                                twiddle_norm = np.linalg.norm(twiddle_np, ord=2, axis=(1, 2))
                                self.assertTrue(np.allclose(twiddle_norm, 1),
                                                (twiddle_norm, device, (in_size, out_size), complex, tied_weight, ortho_init))

if __name__ == "__main__":
    unittest.main()
