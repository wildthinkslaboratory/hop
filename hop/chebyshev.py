import numpy as np


# Chebyshev-Gauss-Lobatto nodes ordered from [-1,...,0,...,1]
def chebyshev_points(N):
   return -np.cos(np.pi * np.arange(N+1) / N)


# Chebyshev D matrix for nodes ordered from [-1,...,0,...,1]
def chebyshev_D(N):
    assert(N > 0)
    nodes = chebyshev_points(N)
    c = np.ones(N + 1) 
    c[0]=2.0
    c[-1]=2.0
    c = c * ((-1) ** np.arange(N+1))
    x = np.tile(nodes, (N + 1,1))
    dX = x - x.T
    D = (np.outer(c, 1/c)) / (dX + np.eye(N + 1))
    return np.diag(np.sum(D, axis=1)) - D


# return the width of each chebyshev segment
def chebyshev_segments(N, T):
    nodes = chebyshev_points(N)
    diffs = []
    for i in range(len(nodes) - 1):
        diffs.append((nodes[i+1] - nodes[i]) * T / 2)
    return diffs

    
# Clenshaw-Curtis quadrature weights
def weights(N):
    j = np.arange(N+1)
    theta = np.pi * j / N

    w = np.zeros(N+1)
    v = np.ones(N-1)
    if (N % 2) == 0:
        w[0] = 1/(N**2-1)
        w[N] = w[0]
        for k in range(1, int(N/2)):
            v = v - 2*np.cos(2*k*theta[1:N])/(4*k**2-1)
        v = v - np.cos(N * theta[1:N])/(N**2-1)
    else:
        w[0] = 1/(N**2)
        w[N] = w[0]
        for k in range(1, int(((N-1)/2)+1)):
            v = v - 2*np.cos(2*k*theta[1:N])/(4*k**2-1)

    w[1:N]=2*v/N
    return w





