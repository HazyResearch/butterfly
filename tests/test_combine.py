import copy
import unittest

import torch

import torch_butterfly


class ButterflyCombineTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_diagonal_butterfly(self):
        batch_size = 10
        for in_size, out_size in [(9, 15), (15, 9)]:
            for nblocks in [1, 2, 3]:
                for complex in [False, True]:
                    for increasing_stride in [True, False]:
                        for diag_first in [True, False]:
                            dtype = torch.float32 if not complex else torch.complex64
                            input = torch.randn(batch_size, in_size, dtype=dtype)
                            b = torch_butterfly.Butterfly(in_size, out_size, bias=False,
                                                          complex=complex,
                                                          increasing_stride=increasing_stride,
                                                          nblocks=nblocks)
                            twiddle_copy = b.twiddle.clone()
                            diagonal = torch.randn(in_size if diag_first else out_size,
                                                   dtype=dtype)
                            out = b(input * diagonal) if diag_first else b(input) * diagonal
                            for inplace in [True, False]:
                                b_copy = copy.deepcopy(b)  # otherwise inplace would modify b
                                bd = torch_butterfly.combine.diagonal_butterfly(
                                    b_copy, diagonal, diag_first, inplace)
                                out_bd = bd(input)
                                self.assertTrue(torch.allclose(out_bd, out, self.rtol, self.atol))
                                if not inplace:
                                    self.assertTrue(torch.allclose(b.twiddle, twiddle_copy,
                                                                   self.rtol, self.atol))

    def test_butterfly_kronecker(self):
        batch_size = 10
        n1 = 16
        n2 = 32
        for complex in [False, True]:
            for increasing_stride in [True, False]:
                dtype = torch.float32 if not complex else torch.complex64
                input = torch.randn(batch_size, n2, n1, dtype=dtype)
                b1 = torch_butterfly.Butterfly(n1, n1, bias=False, complex=complex,
                                               increasing_stride=increasing_stride)
                b2 = torch_butterfly.Butterfly(n2, n2, bias=False, complex=complex,
                                               increasing_stride=increasing_stride)
                b_tp = torch_butterfly.combine.TensorProduct(b1, b2)
                out_tp = b_tp(input)
                b = torch_butterfly.combine.butterfly_kronecker(b1, b2)
                out = b(input.reshape(batch_size, n2 * n1)).reshape(batch_size, n2, n1)
                self.assertTrue(torch.allclose(out, out_tp, self.rtol, self.atol))


if __name__ == "__main__":
    unittest.main()
