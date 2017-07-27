from __future__ import division
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
from sklearn.cluster import SpectralClustering, AffinityPropagation
import clustering
from generate_data import gen_data
import csv

def avg_max_corr(U, Uhat):
  corrs = [np.max(1 - distance.cdist(Uhat.T[j].reshape(1, Uhat.shape[0]),U.T)) for j in range(Uhat.shape[1])]
  return np.mean(corrs)

def dict_sparsity(U):
  counts = np.apply_along_axis(np.count_nonzero, 0, U)
  G = U.shape[0]

  return (min(counts)/G, max(counts)/G, np.mean(counts)/G)

def analyze_bcs(Yf, lY, U0,W0,U1,W1, U,W, Wcs):

  X = U.dot(W)
  p_wy, s_wy = compare_distances(W, Yf)
  p_xy, s_xy = compare_distances(X, Yf)
  p_wx, s_wx = compare_distances(W, X)

  lY = cluster_lab(Yf)
  ami_yx = compare_clusters(Yf, X)
  ami_xw = compare_clusters(X, W)
  ami_yw = compare_clusters(Yf, W)

  x = [p_wy, s_wy, p_xy, s_xy, p_wx, s_wx, ami_yx, ami_xw, ami_yw]
  x = x + bcs_metrics(U0,W0,U1,W1,U,W,Wcs)

  return x

def bcs_metrics(U0,W0,U1,W1,U,W,Wcs):

  X = U.dot(W)
  X0 = U0.dot(W0)
  X1= U1.dot(W1)
  Xcs = U.dot(Wcs)
  init_fit = estimate_diff(X0, X)
  bcs_fit = estimate_diff(X1,X)
  corrs= abs(1 - distance.cdist(U1.T, U.T, 'correlation'))
  minc = np.amin(corrs)
  maxc = np.amax(corrs)

  cs_fit = estimate_diff(Xcs, X)

  return [init_fit, bcs_fit, cs_fit, minc, maxc]

def feasible_param(m,pf, n, k, p, L, G, r):
  """Check that parameter values make sense."""

  params = [m, pf, n, k, p, L, G, r]

  if any(c == 0 for c in params):
    print "Some parameters are 0!"
    return False

  if pf > m:
    print "Number of fixed measurements more than total measurements!"
    return False

  if k > L:
    print "Sparsity is greater than number of modules!"
    return False

  if p > n:
    print "More samples per support pattern than samples!"
    return False

  if L > G:
    print "More modules than genes!"
    return False

  return True

def data_metrics(Yf,lY, U, W):

  X = U.dot(W)
  n = Yf.shape[1]
  c = clustering.num_clusters(n)

  ami_wy = compare_clusters(Yf, W)
  ami_xy = compare_clusters(Yf, X)
  ami_wx = compare_clusters(X, W)

  # Correlation of distances
  p_wy, s_wy = compare_distances(W, Yf)
  p_xy, s_xy = compare_distances(X, Yf)
  p_wx, s_wx = compare_distances(W, X)


  mod_supp_ratio = clustering.module_supp_ratio(lY, W, c)
  minsupp = min(mod_supp_ratio)
  avgsupp = np.mean(mod_supp_ratio)
  maxsupp = max(mod_supp_ratio)

  metrics= [ami_wy, ami_xy, ami_wx, p_wy, p_xy, p_wx, minsupp, maxsupp, avgsupp]
  return metrics

def analyze_clustering(Yf, U, W):

  n = Yf.shape[1]
  X = U.dot(W)
  c = clustering.num_clusters(n)

  # Obtain clusters based on fixed measurements
  cY = clustering.cluster_lab(Yf)
  clusY = clustering.sort_cluster(cY, W, c)

  # Obtain clusters based on gene expression
  cX = clustering.cluster_lab(X)
  clusX = clustering.sort_cluster(cX, W, c)

  # Diversity metrics on the clusters from measurements
  divY = map(clustering.diversity, clusY)
  mindivY = min(divY)
  maxdivY = max(divY)
  avgdivY = np.mean(divY)

  # Diversity metrics on the clusters from gene expression
  divX = map(clustering.diversity, clusX)
  mindivX = min(divX)
  maxdivX = max(divX)
  avgdivX = np.mean(divX)

  # Compare clusters from W, X, and Yf
  ami_yw = compare_clusters(Yf, W)
  ami_yx = compare_clusters(Yf, X)
  ami_xw = compare_clusters(X, W)

  # Correlation of distances
  p_wy, s_wy = compare_distances(W, Yf)
  p_xy, s_xy = compare_distances(X, Yf)
  p_wx, s_wx = compare_distances(W, X)

  mod_supp_ratio = clustering.module_supp_ratio(cY, W, c)
  minsupp = min(mod_supp_ratio)
  avgsupp = np.mean(mod_supp_ratio)
  maxsupp = max(mod_supp_ratio)

  metrics = [mindivY, maxdivY, avgdivY, mindivX, maxdivX, avgdivX, ami_yw,
  ami_yx, ami_xw, p_xy, s_xy, p_wx, s_wx, p_wy, s_wy, minsupp, avgsupp, maxsupp]

  return metrics

def estimate_diff(Xhat, X):
  return (1- np.linalg.norm(Xhat - X)**2/np.linalg.norm(X)**2)

def compare_distances(X,Y):
  """Get Pearson and Spearman correlations between pairwise distance of columns
  in X and Y. X and Y must have the same number of columns."""

  if X.shape[1] != Y.shape[1]:
    raise ValueError('X and Y must have the same number of columns.')

  dist_x = distance.pdist(X.T,'euclidean')
  dist_y = distance.pdist(Y.T,'euclidean')

  p = (1 - distance.correlation(dist_x,dist_y))
  spear = spearmanr(dist_x,dist_y)

  return p,spear[0]

def corr(X,Y):
  """Get Pearson and Spearman correlations of X & Y flattened, and Pearson corr
  across rows and columns of X and Y. X and Y should have the same dimension."""

  if X.shape != Y.shape :
    raise ValueError('X and Y should have the same dimension.')

  flatX = X.flatten()
  flatY = Y.flatten()
  p = (1 - distance.correlation(flatX, flatY))
  spear = spearmanr(flatX, flatY)

  # Calculate correlation across rows and columns.
  pg = corr_along_axis(X, Y, 0)
  ps = corr_along_axis(X, Y, 1)

  return p, spear[0], pg, ps

def corr_along_axis(X, Y, axis):
  """Compute avg Pearson corr of X and Y along axis (0 for rows, 1 for cols)."""

  dist = np.zeros(X.shape[axis])

  if axis == 1: #correlation across cols. Transpose matrices and do across rows.
    X = X.T
    Y = Y.T

  for i in range(X.shape[0]):
    dist[i] = 1 - distance.correlation(X[i],Y[i])

  p = (np.average(dist[np.isfinite(dist)]))

  return p

def compare_clusters(X,Y):
  """Get adjusted mutual information score of clusters from samples X and Y."""

  lX = clustering.cluster_lab(X)
  lY = clustering.cluster_lab(Y)

  return adjusted_mutual_info_score(lX,lY)
