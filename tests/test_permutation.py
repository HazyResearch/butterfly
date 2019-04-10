import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import numpy as np

import torch

from butterfly.permutation import Permutation, FixedPermutation, PermutationFactor


class PermutationTest(unittest.TestCase):

    def test_permutation(self):
        batch_size = 10
        size = 16
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for complex in [False, True]:
                for share_logit in [False, True]:
                    for increasing_stride in [False, True]:
                        perm = Permutation(size, share_logit, increasing_stride).to(device)
                        input = torch.randn((batch_size, size) + (() if not complex else (2,)), device=device)
                        output = perm(input)
                        self.assertTrue(output.shape == (batch_size, size) + (() if not complex else (2,)),
                                        (output.shape, device, (size, size), complex, share_logit, increasing_stride))
                        self.assertTrue(perm.argmax().dtype == torch.int64)
                        fixed_perm = FixedPermutation(perm.argmax())
                        output = fixed_perm(input)
                        self.assertTrue(output.shape == (batch_size, size) + (() if not complex else (2,)),
                                        (output.shape, device, (size, size), complex, share_logit, increasing_stride))

    def test_permutation_single(self):
        batch_size = 10
        size = 16
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for complex in [False, True]:
                perm = PermutationFactor(size).to(device)
                input = torch.randn((batch_size, size) + (() if not complex else (2,)), device=device)
                output = perm(input)
                self.assertTrue(output.shape == (batch_size, size) + (() if not complex else (2,)),
                                (output.shape, device, (size, size), complex))
                self.assertTrue(perm.argmax().dtype == torch.int64)
                fixed_perm = FixedPermutation(perm.argmax())
                output = fixed_perm(input)
                self.assertTrue(output.shape == (batch_size, size) + (() if not complex else (2,)),
                                (output.shape, device, (size, size), complex))

if __name__ == "__main__":
    unittest.main()
