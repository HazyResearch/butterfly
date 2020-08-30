import copy
import itertools
import math
import unittest

import numpy as np

import torch

import torch_butterfly
from torch_butterfly.permutation import perm_vec_to_mat, invert, matrix_to_butterfly_factor


class ButterflyPermutationTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_matrix_to_butterfly_factor(self):
        num_repeats = 10
        for n in [2, 16, 64]:
            for _ in range(num_repeats):
                log_n = int(math.ceil(math.log2(n)))
                for log_k in range(1, log_n + 1):
                    b = torch_butterfly.Butterfly(n, n, bias=False, init='identity')
                    factor = torch.randn(n//2, 2, 2)
                    b.twiddle[0, 0, log_k - 1].copy_(factor)
                    matrix = b(torch.eye(n)).t()
                    factor_out = matrix_to_butterfly_factor(matrix.detach().numpy(), log_k,
                                                            pytorch_format=True, check_input=True)
                    self.assertTrue(torch.allclose(factor_out, factor))

    def test_modular_balance(self):
        num_repeats = 50
        for n in [2, 16, 64]:
            for _ in range(num_repeats):
                v = np.random.permutation(n)
                Rinv_perms, L_vec = torch_butterfly.permutation.modular_balance(v)
                self.assertTrue(torch_butterfly.permutation.is_modular_balanced(L_vec))
                v2 = v
                for p in Rinv_perms:
                    v2 = v2[p]
                self.assertTrue(np.all(v2 == L_vec))
                lv2 = L_vec
                for p in reversed(Rinv_perms):
                    lv2 = lv2[torch_butterfly.permutation.invert(p)]
                self.assertTrue(np.all(lv2 == v))
                R_perms = [perm_vec_to_mat(invert(p)) for p in reversed(Rinv_perms)]
                mat = perm_vec_to_mat(v, left=False)
                for p in reversed(R_perms):
                    mat = mat @ p.T
                self.assertTrue(np.allclose(mat, perm_vec_to_mat(L_vec)))

    def test_perm2butterfly_slow(self):
        num_repeats = 50
        for n in [2, 13, 38]:
            for increasing_stride in [False, True]:
                for complex in [False, True]:
                    for _ in range(num_repeats):
                        v = torch.randperm(n)
                        b = torch_butterfly.permutation.perm2butterfly_slow(v, complex,
                                                                            increasing_stride)
                        input = torch.arange(n, dtype=torch.float32)
                        if complex:
                            input = input.to(torch.complex64)
                        self.assertTrue(torch.allclose(input[v], b(input)))

    def test_perm2butterfly(self):
        num_repeats = 50
        for n in [2, 13, 38]:
            for increasing_stride in [False, True]:
                for complex in [False, True]:
                    for _ in range(num_repeats):
                        v = torch.randperm(n)
                        b = torch_butterfly.permutation.perm2butterfly(v, complex,
                                                                       increasing_stride)
                        input = torch.arange(n, dtype=torch.float32)
                        if complex:
                            input = input.to(torch.complex64)
                        self.assertTrue(torch.allclose(input[v], b(input)))


if __name__ == "__main__":
    unittest.main()
