import numpy as np
import numpy.random as rd
import math
import scipy as sp
import sys
from scipy.spatial import distance

def gen_data(m, pf, n, k, p, L, G, max_r, u_density=1.0):
  """Randomly generate dictionary U and activity matrix W. Randomly generate
  matrix of fixed measurements Af and 3d-array of measurement matrices Av.
  Generate observations Yf and Yv."""

  Af = gen_a(pf, G)
  U = gen_dict(max_r, L, G, u_density)
  W, Supp = gen_W(n, k, p, L)
  Yf = compress(Af, U, W)
  pv = m  - pf

  Yv = np.empty((m, n))
  Av = np.empty(shape = (n, m, G))

  for i in range(0, n): # get signals for variable measurements
    Ai = gen_a(pv, G)
    Ai = np.vstack([Af, Ai])
    Av[i] = Ai
    Yi = compress(Ai, U, W[:,i])
    Yv[:,i] = Yi

  return (Yf, Yv, Af, Av, U, W)

def gen_a(m,G):
  """Generate fixed measurement matrix A. A is m x G, where G is
  the number of genes."""

  A = rd.standard_normal((m, G))
  A = A/np.linalg.norm(A,axis=0)

  return A

def gen_dict(max_corr, L, G, u_density):
  """Generate dictionary U with L modules on G genes. Max_corr is the unsigned
  maximum threshold correlation between any two column vectors. U is G x L."""

  U = np.empty((G, L))
  k = int(u_density * G)
  S = rand_subset(G, k)
  v = rd.uniform(0, 1, G)
  v = sparsify(v, S)
  v = v/np.linalg.norm(v)
  U[:,0] = v

  for i in range(1,L):
    r_max = 1

    while r_max > max_corr:
      S = rand_subset(G, k)
      v = rd.uniform(0, 1, G)
      v = sparsify(v, S)
      v = v/np.linalg.norm(v)
      r_max = max(abs(1 - distance.cdist(U[:,0:i].T, [v], 'correlation')))

    U[:,i] = v

  return U


def sparsify(w, S):
  """Sparsify vector w of length L of support pattern S.
  S is a set of indices from 0 to L-1 inclusive."""

  L = len(w)
  ind = [i for i in range(0, L) if i not in S]
  for i in ind:
    w[i] = 0

  return w

def gen_W(n, k, p, L):
  """Randomly generate module activity matrix and list of support patterns Supp.
  Each column vector is k-sparse. L = number of modules, p = minimum number of
  samples per support pattern, and n = number of total samples. W is L x n."""

  W = np.empty((L, n))
  ind = 0
  s = int(math.floor(n/p)) # max number of support patterns
  Supp = []

  for i in range(0, s):
    # for later: more realistic to have a range of k?
    S = rand_subset(L, k)
    Supp.append(S)

    for j in range(0, p): # Generate p vectors for each support pattern S.
      w = rd.randn(L)
      w = sparsify(w,S)
      W[:,ind] = w
      ind = ind + 1

  while ind < n: # Generate remaining samples
    w = rd.randn(L)
    w = sparsify(w,S)
    W[:,ind] = w
    ind = ind + 1

  return (W, Supp)

def rand_subset(L, k):
  """Randomly select subset of k elements from L, without replacement."""

  if not 0 <= k <= L:
    raise ValueError('Must have 0 <= k <= L.')

  ind = list(range(0, L))
  S = []

  for n in range(0, k):
    i = rd.randint(0 , L - n)
    s = ind.pop(i)
    S.append(s)

  return S

def compress(A,U, W):
  """Generate observation matrix Y (m x n) based on measurement matrix
  A, randomly generated dict U with L modules, and W. Y = AUW."""

  return A.dot(U).dot(W)
