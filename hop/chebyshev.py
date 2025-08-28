import numpy as np
from numpy.polynomial import chebyshev as cheb


# Chebyshev-Gauss-Lobatto nodes
def chebyshev_points(N):
    return cheb.chebpts2(N+1)


def chebyshev_D(N):

    j = np.arange(N+1)
    theta = np.pi * j / N

    assert(N > 0)
    nodes = chebyshev_points(N)
    c = np.ones(N + 1) 
    c[0]=2.0
    c[-1]=2.0
    c = c * ((-1) ** np.arange(N+1))
    x = np.tile(nodes, (N + 1,1))
    dX = x - x.T
    D = (np.outer(c, 1/c)) / (dX + np.eye(N + 1))
    return theta, D - np.diag(np.sum(D, axis=1))


def clenshaw_curtis(N):
    k = np.arange(0, N+1)
    x = np.cos(np.pi * k / N)  # nodes
    w = np.zeros(N+1)

    for j in range(N+1):
        if j == 0 or j == N:
            factor = 1
        else:
            factor = 2
        s = 0
        for m in range(1, N//2+1):
            b = 2 if (2*m != N) else 1
            s += b * np.cos(2*m*j*np.pi/N) / (4*m*m - 1)
        w[j] = (2.0/N) * (1 - s)
    return x, w