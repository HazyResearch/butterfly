import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import numpy as np

import torch

from butterfly import Butterfly
from butterfly.butterfly import ButterflyBmm

from butterfly.butterfly_multiply import butterfly_ortho_mult_tied


class ButterflyTest(unittest.TestCase):

    def test_butterfly(self):
        batch_size = 10
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for in_size, out_size in [(7, 15), (15, 7)]:
                for complex in [False, True]:
                    for tied_weight in [True, False]:
                        for increasing_stride in [True, False]:
                            for ortho_init in [False, True]:
                                for param in ['regular'] if complex else ['regular', 'ortho', 'odo', 'odr', 'opdo', 'obdobt', 'svd']:
                                    for nblocks in [0, 1, 2, 3] if param in ['regular', 'ortho', 'odo', 'odr', 'opdo', 'obdobt'] else [0]:
                                        for expansion in [1, 2]:
                                            for double in [False, True]:
                                                if param in ['obdobt', 'svd'] and tied_weight:
                                                    continue
                                                if nblocks > 0 and complex:
                                                    continue
                                                if not (nblocks > 0 and tied_weight and param in ['odo', 'odr', 'opdo']):  # Special case
                                                    if nblocks > 0 and (tied_weight or param not in ['regular', 'ortho', 'odo', 'odr', 'opdo', 'obdobt']):
                                                        continue
                                                b = Butterfly(in_size, out_size, True, complex, tied_weight, increasing_stride, ortho_init, param, nblocks=nblocks, expansion=expansion).to(device)
                                                input = torch.randn((batch_size, in_size) + (() if not complex else (2,)), device=device)
                                                output = b(input)
                                                self.assertTrue(output.shape == (batch_size, out_size) + (() if not complex else (2,)),
                                                                (output.shape, device, (in_size, out_size), complex, tied_weight, ortho_init, nblocks))
                                                if ortho_init and param == 'regular':
                                                    twiddle_np = b.twiddle.detach().to('cpu').numpy()
                                                    if complex:
                                                        twiddle_np = twiddle_np.view('complex64').squeeze(-1)
                                                    twiddle_np = twiddle_np.reshape(-1, 2, 2)
                                                    twiddle_norm = np.linalg.norm(twiddle_np, ord=2, axis=(1, 2))
                                                    self.assertTrue(np.allclose(twiddle_norm, 1),
                                                                    (twiddle_norm, device, (in_size, out_size), complex, tied_weight, ortho_init))

    def test_butterfly_expansion(self):
        batch_size = 1
        device = 'cpu'
        in_size, out_size = (16, 16)
        expansion = 4
        b = Butterfly(in_size, out_size, bias=False, tied_weight=True, param='odo', expansion=expansion, diag_init='normal').to(device)
        input = torch.randn((batch_size, in_size), device=device)
        output = b(input)
        terms = []
        for i in range(expansion):
            temp = butterfly_ortho_mult_tied(b.twiddle[[i]], input.unsqueeze(1), False)
            temp = temp * b.diag[i]
            temp = butterfly_ortho_mult_tied(b.twiddle1[[i]], temp, True)
            terms.append(temp)
        total = sum(terms)
        self.assertTrue(torch.allclose(output, total))


    def test_butterfly_bmm(self):
        batch_size = 10
        matrix_batch = 3
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for in_size, out_size in [(7, 15), (15, 7)]:
                for complex in [False, True]:
                    for tied_weight in [True, False]:
                        for increasing_stride in [True, False]:
                            for ortho_init in [False, True]:
                                for param in ['regular'] if complex else ['regular', 'ortho', 'odo', 'odr', 'opdo', 'obdobt', 'svd']:
                                    for nblocks in [0, 1, 2, 3] if param in ['regular', 'ortho', 'odo', 'odr', 'opdo', 'obdobt'] else [0]:
                                        for expansion in [1, 2]:
                                            for double in [False, True]:
                                                if param in ['obdobt', 'svd'] and tied_weight:
                                                    continue
                                                if nblocks > 0 and complex:
                                                    continue
                                                if not (nblocks > 0 and tied_weight and param in ['odo', 'odr', 'opdo']):  # Special case
                                                    if nblocks > 0 and (tied_weight or param not in ['regular', 'ortho', 'odo', 'odr', 'opdo', 'obdobt']):
                                                        continue
                                                b_bmm = ButterflyBmm(in_size, out_size, matrix_batch, True, complex, tied_weight, increasing_stride, ortho_init, param, expansion=expansion).to(device)
                                                input = torch.randn((batch_size, matrix_batch, in_size) + (() if not complex else (2,)), device=device)
                                                output = b_bmm(input)
                                                self.assertTrue(output.shape == (batch_size, matrix_batch, out_size) + (() if not complex else (2,)),
                                                                (output.shape, device, (in_size, out_size), complex, tied_weight, ortho_init))
                                                # Check that the result is the same as looping over butterflies
                                                if param == 'regular':
                                                    output_loop = []
                                                    for i in range(matrix_batch):
                                                        b = Butterfly(in_size, out_size, True, complex, tied_weight, increasing_stride, ortho_init, expansion=expansion)
                                                        b.twiddle = torch.nn.Parameter(b_bmm.twiddle[i * b_bmm.nstack:(i + 1) * b_bmm.nstack])
                                                        b.bias = torch.nn.Parameter(b_bmm.bias[i])
                                                        output_loop.append(b(input[:, i]))
                                                    output_loop = torch.stack(output_loop, dim=1)
                                                    self.assertTrue(torch.allclose(output, output_loop),
                                                                    ((output - output_loop).abs().max().item(), output.shape, device, (in_size, out_size), complex, tied_weight, ortho_init))
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
