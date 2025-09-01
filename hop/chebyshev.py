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
    print(c)
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
    print(w)


def weights_paper(N):
    w = np.zeros(N+1)
    if (N % 2) == 0:
        for i in range(N+1):
            if i == 0 or i == N:
                w[i] = 1 / (N**2 - 1)
            else:
                s = i
                if i > N//2:
                    s = N-i
                for j in range(N//2 + 1):
                    w[i] +=  1/(1-4*j**2) * np.cos(2 * np.pi * j* s/N)
    else:
        for i in range(N+1):
            if i == 0 or i == N:
                w[i] = 1 / N**2
            else:
                pass
    return w





def clenshaw_curtis_compute(n):
  n += 1
  if n == 1:
    x = np.zeros(n)
    w = np.zeros(n)
    w[0] = 2.0
  else:
    theta = np.zeros(n)
    for i in range(n):
      theta[i] = float ( n - 1 - i ) * np.pi / float ( n - 1 )

    x = np.cos ( theta )
    w = np.zeros ( n )

    for i in range ( 0, n ):

      w[i] = 1.0

      jhi = ( ( n - 1 ) // 2 )

      for j in range ( 0, jhi ):

        if ( 2 * ( j + 1 ) == ( n - 1 ) ):
          b = 1.0
        else:
          b = 2.0

        w[i] = w[i] - b * np.cos ( 2.0 * float ( j + 1 ) * theta[i] ) \
             / float ( 4 * j * ( j + 2 ) + 3 )

    w[0] = w[0] / float ( n - 1 )
    for i in range ( 1, n - 1 ):
      w[i] = 2.0 * w[i] / float ( n - 1 )
    w[n-1] = w[n-1] / float ( n - 1 )

  return x, w


print(weights_paper(6))
print(weights(6))
print(clenshaw_curtis_compute(6))