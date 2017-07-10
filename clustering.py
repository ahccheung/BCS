from __future__ import division
import numpy as np
import numpy.random as rd
import math
import scipy as sp
from sklearn.cluster import SpectralClustering
import generate_data
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score

def main(Yf, W):
  """Cluster samples based on fixed measurements. Compute the diversity of each
  cluster."""

  c = num_clusters(Yf.shape[1])
  lY = cluster_lab(Yf)
  clus = sort_cluster(lY, W, c)
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

  return int(max(5,min(20,n/50)))

def cluster_lab(Y):
  """Returns cluster labels from composite measurements Y."""

  n = Y.shape[1]
  c = num_clusters(n)
  dY = sim_mat(Y)
  lY = SpectralClustering(n_clusters=c,affinity='precomputed').fit_predict(dY)

  return lY

def sort_cluster(lY,W, n_clus):
  """Return matrix of vectors for each cluster."""

  clusters = []
  n_samples = len(lY)

  for n in range(0, n_clus):
    clusters.append([])
    cidx = np.where(lY == n)[0]
    clusters[n] = W[:,cidx]

  #for j in range(0, n_samples): # sort observations to appropriate cluster
    #c = lY[j]
    # clusters[c].append(W[:,j])

  return clusters

def cluster_supp(lY, W, n_clus):
  """For set of clusters, compute support of each cluster."""

  Supp = []
  clusters = sort_cluster(lY, W, n_clus)

  for c in range(0, num_clus): # compute support of each cluster.
    Supp.append(compute_supp(clusters[c]))

  return(Supp)

def compute_supp(W):
  """Takes a matrix of vectors and computes the support of them."""

  # Add later: check list is non-empty.

  N = W.shape[1]
  S = set()
  for n in range(0, N):
    ind = vec_supp(W[:,n]) # support of nth vector
    S = set().union(S, ind)

  return S

def vec_supp(w):
  """Returns the support of a vector w."""

  L = len(w)
  ind = [l for l in range(0, L) if w[l]!= 0]

  return ind

def diversity(C):
  """Computes the diversity of a cluster C of vectors. C is a matrix of vectors
  in the cluster."""

  count = count_vec(C) # vector of counts of each module
  entropy = sp.stats.entropy(count)

  return np.exp(entropy)

def module_cluster_supp(lY, W, num_clus):
  """Return the max # of times each module is supported in a single cluster."""

  L = W.shape[0]
  clusters = sort_cluster(lY, W, num_clus)
  counts = np.empty((L, num_clus))
  counts = counts.astype(int)

  for i in range(0, num_clus):
    counts[:,i] = count_vec(clusters[i])

  supp = np.apply_along_axis(max, 1, counts) #max count for each module

  return supp

def module_supp_ratio(lY, W, num_clus):

  clus_supp = module_cluster_supp(lY, W, num_clus)
  mod_supp = generate_data.count_vec(W)
  L = W.shape[0]
  ratios = [clus_supp[i]/mod_supp[i] for i in range(0,L) if mod_supp[i] != 0]
  return ratios

def count_vec(C):
  """Produces vector of counts based on how many non-zero entries there are in
  each row of a matrix."""

  counts = np.apply_along_axis(np.count_nonzero, 1, C)

  return counts.astype(int)
