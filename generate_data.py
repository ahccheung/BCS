import numpy as np
import numpy.random as rd
import math
import scipy as sp
import sys

### Requires numpy version 1.12 or higher

def gen_data(m, pf, n, k, p, L, G, theta):
  """Randomly generate dictionary U and activity matrix W. Randomly generate
  matrix of fixed measurements Af and list of variable measurement matrices Av.
  Generate observations Yf and Yv."""

  Af = gen_a(pf, G)
  U = gen_dict(theta, L, G)
  W, Supp = gen_W(n, k, p, L)
  Yf = compress(Af, U, W) # pfxn matrix of fixed composite measurements.
  pv = m  - pf

  Yv = np.empty((pv, n)) # observations from variable measurements.
  Av = [] # list of n matrices for variable measurements

  for i in range(0, n): # get signals for variable measurements
    Ai = gen_a(pv, G)
    Av.append(Ai)
    Yi = compress(Ai, U, W[:,i])
    Yi = Yi.getA1()
    Yv[:,i] = Yi

  return (Yf, Yv, Af, Av, U, W)

def gen_a(m,G):
  """Generate fixed measurement matrix A. A is m x G, where G is
  the number of genes."""

  A = rd.standard_normal((m, G))
  A = A/np.linalg.norm(A,axis=0)

  return np.matrix(A)

def gen_dict(theta, L, G):
  """Generate dictionary U with L modules on G genes. Theta is a parameter that
  denotes the max angle between a reference col vector and any other vector.
  U is G x L."""

  U = np.empty((G, L))
  v = rd.uniform(0, 1, G) # Generate random vector. Another distribution?
  v = v/np.linalg.norm(v)
  U[:,0] = v

  for i in range(1, L):
    angle = rd.uniform(0, theta) # Choose an angle in between 0 and theta
    vp = angle_vec(v, angle) # Get unit vector at appropriate angle
    U[:,i] = vp

  return np.matrix(U)

def angle_vec(v, theta):
  """Construct random vector at angle theta from v. Assumes v has norm 1."""

  p = len(v)
  w = rd.uniform(0, 1, p)
  w = w - np.vdot(v, w) * v
  w = w / np.linalg.norm(w) # w = unit length vector orthogonal to v
  z = np.cos(theta) * v + np.sin(theta) * w # rotate to appropriate angle

  return z/np.linalg.norm(z)

def gen_w(L, S):
  """Generate random sparse vector w with length L of support pattern S.
  S is a set of indices from 0 to L-1 inclusive. Entries of w are generated
  from standard normal dist."""

  w = rd.randn(L)
  ind = [i for i in range(0, L) if i not in S]
  for i in ind:
    w[i] = 0 # Set indices not in support to 0

  return w

def gen_W(n, k, p, L):
  """Randomly generate module activity matrix and list of support patterns Supp.
  Each column vector is k-sparse. L = number of modules, p = minimum number of
  samples per support pattern, and n = number of total samples. W is L x n."""

  W = np.empty((L, n))
  ind = 0
  s = int(math.floor(n/p)) # max number of support patterns
  Supp = []# list of support patterns.

  for i in range(0, s):
    # for later: more realistic to have a range of k?
    S = rand_subset(L, k) # Randomly find support pattern
    Supp.append(S)
    # for later: check support pattern hasn't already been generated before.
    # not too important if L is big enough

    for j in range(0, p): # Generate p vectors for each support pattern S.
      w = gen_w(L, S)
      W[:,ind] = w
      ind = ind + 1

  while ind < n: # Generate remaining samples
    w = gen_w(L, S)
    W[:,ind] = w
    ind = ind + 1

  return (np.matrix(W), Supp)

def module_supp(W):
"""Count number of times each module is activated."""

  return np.count_nonzero(W, axis = 1)

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

  return A * U * W
