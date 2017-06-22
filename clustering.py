import numpy as np
import numpy.random as rd
import math
import scipy as sp
from sklearn.cluster import SpectralClustering
import generate_data
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score

def main(Yf, Yv, Af, Av, U, W):
  """Cluster samples based on fixed measurements. Compute the diversity of each
  cluster."""

  c = num_clusters(Yf.shape[1])
  cY = cluster_lab(Yf)
  clus = sort_cluster(cY, W, c)
  cdiv = map(diversity, clus)
  print cdiv

def sim_mat(Y):
  """Produces similarity matrix from data matrix Y."""

  S = (Y/np.linalg.norm(Y,axis=0)).T
  dS = 1 - S.dot(S.T)
  dS = np.exp(-np.square(dS)/2.)

  return dS

def num_clusters(n):
  """Returns number of clusters based on number of samples."""

  return max(5,min(20,n/50))

def cluster_lab(Y):
  """Returns cluster labels from composite measurements Y."""

  n = Y.shape[1]
  c = num_clusters(n)
  dY = sim_mat(Y)
  lY = SpectralClustering(n_clusters=c,affinity='precomputed').fit_predict(dY)

  return lY

def sort_cluster(lY,W, n_clus):
  """Return list of vectors for each cluster."""

  clusters = []
  n_samples = len(lY)

  for n in range(0, n_clus): #initialize empty lists for each cluster
    clusters.append([])

  for j in range(0, n_samples): # sort observations to appropriate cluster

    c = lY[j] # observation j is in cluster c.
    clusters[c].append(W[:,j])

  return clusters

def cluster_supp(lY, W, n_clus):
  """For set of clusters, compute support of each cluster."""

  Supp = []
  clusters = cluster_vec(lY, W, n_clus)

  for c in range(0, num_clus): # compute support of each cluster.
    Supp.append(compute_supp(clusters[c]))

  return(Supp)

def compute_supp(W):
  """Takes a list of vectors and computes the support of them."""

  # Add later: check list is non-empty.

  N = len(W)
  S = set()
  for n in range(0, N):
    ind = vec_supp(W[n]) # support of nth vector
    S = set().union(S, ind)

  return S

def vec_supp(w):
  """Returns the support of a vector w."""

  L = len(w)
  ind = [l for l in range(0, L) if w[l]!= 0]

  return ind

def diversity(C):
  """Computes the diversity of a cluster C of vectors. C is a list of vectors in
  the cluster."""

  count = count_vec(C) # vector of counts of each module
  entropy = sp.stats.entropy(count)

  return np.exp(entropy)

def module_cluster_supp(lY, W, num_clus):
  """Return the max # of times each module is supported in a single cluster."""

  L = W.shape[0]
  clusters = sort_cluster(lY, W, num_clus)
  counts = np.empty((L, num_clus))

  for i in range(0, num_clus):
    counts[:,i] = count_vec(clusters[i]) #Get counts for each module and cluster

  supp = np.apply_along_axis(max, 1, counts) #max count for each module

  return supp


def count_vec(C):
  """Produces vector of counts based on how many non-zero entries there are in
  the cluster. C is a list of vectors in each cluster."""

  L = len(C[0])
  count = np.zeros(L)

  for w in C:
    ind = vec_supp(w) # indices of non-zero entries
    for i in ind:
      count[i] += 1

  return count

def test(m, pf, n, k, p, L, G, theta,ntries):
  for i in range(0, ntries):
    Yf, Yv, Af, Av, U, W = generate_data.main(m, pf, n, k, p, L, G, theta)
    main(Yf, Yv, Af, Av, U, W)
