import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import numpy as np

import torch

from cnn.models.sparse_conv import SparseConv2d

from unittest.mock import patch, Mock  
import torch.nn as nn

class SparseTest(unittest.TestCase): 

    @patch.object(SparseConv2d, '__init__') 
    def test_prune(self, test_init):
        test_init.return_value  = None 
        m = SparseConv2d() 
        m.weight = torch.tensor([[10,2,-9], [4,5,12]])
        m.nparams = 1
        mask = m._sparse_projection()
        torch.testing.assert_allclose(mask, torch.tensor([[0,0,0],[0,0,1]]))
        m.nparams = 3 
        mask = m._sparse_projection() 
        torch.testing.assert_allclose(mask, torch.tensor([[1,0,1],[0,0,1]]))

    @patch.object(SparseConv2d, '__init__')
    def test_forward(self, test_init): 
        test_init.return_value = None 
        m = SparseConv2d()
        m.weight = torch.tensor([[4,5,-9], [12,2,3]]) 
        m.mask = torch.tensor([[0,0,1], [1,0,0]])
        x = torch.tensor([3,4]) 
        with patch.object(nn.Conv2d, 'forward') as test_conv2d_forward: 
            test_conv2d_forward.return_value = None  
            m.forward(x) 
        torch.testing.assert_allclose(m.weight, torch.tensor([[0,0,-9],[12,0,0]]))

if __name__ == "__main__":
    unittest.main()
