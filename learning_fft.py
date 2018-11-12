import numpy as np
import torch
from torch import nn
from torch import optim

from butterfly import Butterfly, ButterflyProduct
from complex_utils import complex_matmul


# Hadamard matrix for n = 4
size = 4
M0 = Butterfly(size, diagonal=2, diag=torch.tensor([1.0, 1.0, -1.0, -1.0], requires_grad=True), subdiag=torch.ones(2, requires_grad=True), superdiag=torch.ones(2, requires_grad=True))
M1 = Butterfly(size, diagonal=1, diag=torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True), subdiag=torch.tensor([1.0, 0.0, 1.0], requires_grad=True), superdiag=torch.tensor([1.0, 0.0, 1.0], requires_grad=True))
H = M0.matrix() @ M1.matrix()
from scipy.linalg import hadamard
assert torch.allclose(H, torch.tensor(hadamard(4), dtype=torch.float))

size = 16
H = torch.tensor(hadamard(size), dtype=torch.float)


model = ButterflyProduct(size)
# model = Butterfly(size, diagonal=1)
optimizer = optim.Adam(model.parameters(), lr=0.03)
# optimizer = optim.LBFGS(model.parameters(), lr=0.5)
for i in range(15000):
    # def closure():
    #     optimizer.zero_grad()
    #     y = model.matrix()
    #     loss = nn.functional.mse_loss(y, H)
    #     loss.backward(retain_graph=True)
    #     return loss
    # optimizer.step(closure)

    optimizer.zero_grad()
    # x = torch.randn(64, size)
    # y = model(x)
    # loss = nn.functional.mse_loss(y, x @ H.t())
    y = model.matrix()
    loss = nn.functional.mse_loss(y, H)
    # y = model.butterflies[0].matrix()
    # loss = nn.functional.mse_loss(y, H)
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        y = model.matrix()
        loss = nn.functional.mse_loss(y, H)
        print(f'Loss: {loss.item()}')


size = 4
DFT = torch.fft(torch.stack((torch.eye(size), torch.zeros((size, size))), dim=-1), 1)
P = torch.stack((torch.tensor([[1., 0., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 0., 1.]]),
                 torch.zeros((size, size))), dim=-1)
M0 = Butterfly(size,
               diagonal=2,
               complex=True,
               diag=torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], requires_grad=True),
               subdiag=torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True),
               superdiag=torch.tensor([[1.0, 0.0], [0.0, -1.0]], requires_grad=True))
M1 = Butterfly(size,
               diagonal=1,
               complex=True,
               diag=torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], requires_grad=True),
               subdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True),
               superdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True))
assert torch.allclose(complex_matmul(M0.matrix(), complex_matmul(M1.matrix(), P)), DFT)
D = complex_matmul(DFT, P.transpose(0, 1))
assert torch.allclose(complex_matmul(M0.matrix(), M1.matrix()), D)

model = ButterflyProduct(size, complex=True)
optimizer = optim.Adam(model.parameters(), lr=0.03)
# optimizer = optim.LBFGS(model.parameters(), lr=0.5)
for i in range(15000):
    optimizer.zero_grad()
    # x = torch.randn(64, size)
    # y = model(x)
    # loss = nn.functional.mse_loss(y, x @ D.t())
    y = model.matrix()
    loss = nn.functional.mse_loss(y, D)
    # y = model.butterflies[0].matrix()
    # loss = nn.functional.mse_loss(y, D)
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        y = model.matrix()
        loss = nn.functional.mse_loss(y, D)
        print(f'Loss: {loss.item()}')
