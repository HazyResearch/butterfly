import numpy as np

n = 4
# x = np.random.randn(n)
x = np.arange(2, n+2)
V = np.vander(x, increasing=True)
D = np.diag(x)
D_inv = np.diag(1 / x)
Z0 = np.diag(np.ones(n-1), -1)
G = np.array(x ** n)[:, None]
H = np.array([0, 0, 0, 1])[:, None]
assert np.allclose(D @ V - V @ Z0 - G @ H.T, 0)
G = np.array(x ** (n-1))[:, None]
V - D_inv @ V @ Z0 - G @ H.T
A_power = [np.linalg.matrix_power(D_inv, i) for i in range(n)]
B_power = [np.linalg.matrix_power(Z0.T, i) for i in range(n)]
A_power_G = np.hstack([a @ G for a in A_power])
B_power_H = np.hstack([b @ H for b in B_power])
A_power_G @ B_power_H.T

v = np.random.randn(n)
result_slow = V @ v
A_power_G @ v[::-1]

result_slow = V.T @ v
(V * v[:, None]).sum(axis=0)
