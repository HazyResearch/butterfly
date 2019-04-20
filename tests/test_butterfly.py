import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import numpy as np

import torch

from butterfly import Butterfly
from butterfly.butterfly import ButterflyBmm


class ButterflyTest(unittest.TestCase):

    def test_butterfly(self):
        batch_size = 10
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for in_size, out_size in [(7, 15), (15, 7)]:
                for complex in [False, True]:
                    for tied_weight in [True, False]:
                        for increasing_stride in [True, False]:
                            for ortho_init in [False, True]:
                                for param in ['regular'] if complex else ['regular', 'ortho']:
                                    b = Butterfly(in_size, out_size, True, complex, tied_weight, increasing_stride, ortho_init, param).to(device)
                                    input = torch.randn((batch_size, in_size) + (() if not complex else (2,)), device=device)
                                    output = b(input)
                                    self.assertTrue(output.shape == (batch_size, out_size) + (() if not complex else (2,)),
                                                    (output.shape, device, (in_size, out_size), complex, tied_weight, ortho_init))
                                    if ortho_init and param == 'regular':
                                        twiddle_np = b.twiddle.detach().to('cpu').numpy()
                                        if complex:
                                            twiddle_np = twiddle_np.view('complex64').squeeze(-1)
                                        twiddle_np = twiddle_np.reshape(-1, 2, 2)
                                        twiddle_norm = np.linalg.norm(twiddle_np, ord=2, axis=(1, 2))
                                        self.assertTrue(np.allclose(twiddle_norm, 1),
                                                        (twiddle_norm, device, (in_size, out_size), complex, tied_weight, ortho_init))

    def test_butterfly_bmm(self):
        batch_size = 10
        matrix_batch = 3
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for in_size, out_size in [(7, 15), (15, 7)]:
                for complex in [False, True]:
                    for tied_weight in [True, False]:
                        for increasing_stride in [True, False]:
                            for ortho_init in [False, True]:
                                for param in ['regular'] if complex else ['regular', 'ortho']:
                                    b_bmm = ButterflyBmm(in_size, out_size, matrix_batch, True, complex, tied_weight, increasing_stride, ortho_init, param).to(device)
                                    input = torch.randn((batch_size, matrix_batch, in_size) + (() if not complex else (2,)), device=device)
                                    output = b_bmm(input)
                                    self.assertTrue(output.shape == (batch_size, matrix_batch, out_size) + (() if not complex else (2,)),
                                                    (output.shape, device, (in_size, out_size), complex, tied_weight, ortho_init))
                                    # Check that the result is the same as looping over butterflies
                                    if param == 'regular':
                                        output_loop = []
                                        for i in range(matrix_batch):
                                            b = Butterfly(in_size, out_size, True, complex, tied_weight, increasing_stride, ortho_init)
                                            b.twiddle = torch.nn.Parameter(b_bmm.twiddle[i * b_bmm.nstack:(i + 1) * b_bmm.nstack])
                                            b.bias = torch.nn.Parameter(b_bmm.bias[i])
                                            output_loop.append(b(input[:, i]))
                                        output_loop = torch.stack(output_loop, dim=1)
                                        self.assertTrue(torch.allclose(output, output_loop),
                                                        (output.shape, device, (in_size, out_size), complex, tied_weight, ortho_init))
                                    if ortho_init and param == 'regular':
                                        twiddle_np = b_bmm.twiddle.detach().to('cpu').numpy()
                                        if complex:
                                            twiddle_np = twiddle_np.view('complex64').squeeze(-1)
                                        twiddle_np = twiddle_np.reshape(-1, 2, 2)
                                        twiddle_norm = np.linalg.norm(twiddle_np, ord=2, axis=(1, 2))
                                        self.assertTrue(np.allclose(twiddle_norm, 1),
                                                        (twiddle_norm, device, (in_size, out_size), complex, tied_weight, ortho_init))

if __name__ == "__main__":
    unittest.main()
