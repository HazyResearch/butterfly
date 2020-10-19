import math
import unittest

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torch_butterfly


class ButterflyTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_multiply(self):
        for batch_size, n in [(10, 4096), (8192, 512)]:  # Test size smaller than 1024 and large batch size for race conditions
        # for batch_size, n in [(10, 64)]:
        # for batch_size, n in [(1, 2)]:
            log_n = int(math.log2(n))
            nstacks = 2
            nblocks = 3
            for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            # for device in ['cuda']:
                for complex in [False, True]:
                # for complex in [False]:
                    for increasing_stride in [True, False]:
                    # for increasing_stride in [True]:
                        if batch_size > 1024 and (device == 'cpu'):
                            continue
                        dtype = torch.float32 if not complex else torch.complex64
                        # complex randn already has the correct scaling of stddev=1.0
                        scaling = 1 / math.sqrt(2)
                        twiddle = torch.randn((nstacks, nblocks, log_n, n // 2, 2, 2), dtype=dtype, requires_grad=True, device=device) * scaling
                        input = torch.randn((batch_size, nstacks, n), dtype=dtype, requires_grad=True, device=twiddle.device)
                        output = torch_butterfly.butterfly_multiply(twiddle, input, increasing_stride)
                        output_torch = torch_butterfly.multiply.butterfly_multiply_torch(twiddle, input, increasing_stride)
                        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                        ((output - output_torch).abs().max().item(), device, complex, increasing_stride))
                        grad = torch.randn_like(output_torch)
                        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                        ((d_input - d_input_torch).abs().max().item(), device, complex, increasing_stride))
                        # if device == 'cuda' and batch_size > 1024 and not complex and increasing_stride:
                        #     print((d_twiddle - d_twiddle_torch).abs().mean(dim=(0, 2, 3, 4)))
                        #     print(((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().mean(dim=(0, 2, 3, 4)))
                        #     i = ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().argmax()
                        #     print(d_twiddle.flatten()[i])
                        #     print(d_twiddle_torch.flatten()[i])
                        #     print(d_twiddle.flatten()[i-5:i+5])
                        #     print(d_twiddle_torch.flatten()[i-5:i+5])
                        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                       atol=self.atol * (10 if batch_size > 1024 else 1)),
                                        (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(),
                                         (batch_size, n), device, complex, increasing_stride))

    def test_input_padding_output_slicing(self):
        batch_size = 10
        nstacks = 2
        nblocks = 3
        for n in [32, 4096]:
            log_n = int(math.log2(n))
            for input_size in [n // 2 - 1, n - 4, 2 * n + 7]:
                for output_size in [None, n // 2 - 2, n - 5]:
                    for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
                        for complex in [False, True]:
                            for increasing_stride in [True, False]:
                                dtype = torch.float32 if not complex else torch.complex64
                                # complex randn already has the correct scaling of stddev=1.0
                                scaling = 1 / math.sqrt(2)
                                twiddle = torch.randn((nstacks, nblocks, log_n, n // 2, 2, 2), dtype=dtype, requires_grad=True, device=device) * scaling
                                input = torch.randn((batch_size, nstacks, input_size), dtype=dtype, requires_grad=True, device=twiddle.device)
                                output = torch_butterfly.butterfly_multiply(twiddle, input, increasing_stride, output_size)
                                output_torch = torch_butterfly.multiply.butterfly_multiply_torch(twiddle, input, increasing_stride, output_size)
                                self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                                ((output - output_torch).abs().max().item(), device, complex, increasing_stride))
                                grad = torch.randn_like(output_torch)
                                d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                                d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                                self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                                ((d_input - d_input_torch).abs().max().item(), device, complex, increasing_stride))
                                self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                            atol=self.atol * (10 if batch_size > 1024 else 1)),
                                                (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(),
                                                 (d_twiddle - d_twiddle_torch).abs().max().item(),
                                                (batch_size, n), device, complex, increasing_stride))



if __name__ == "__main__":
    unittest.main()
