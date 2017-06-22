import numpy as np
import itertools
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
from sklearn.cluster import SpectralClustering, AffinityPropagation
import clustering
from generate_data import gen_data

def test_parameters(filename):

  G = 1000
  ms = [int(np.floor(x)) for x in [G/200, G/100, G/50, G/20, G/10]]
  ns = [int(np.floor(x)) for x in [G/20, G/10, G/5, G/2]]
  Ls = [int(np.floor(G/2))]; Gs = [G]; thetas =

  params = itertools.product(ms, ns, ks, ps, Ls, Gs, thetas)

  for par in params:
    m, n, k, p, L, G, theta = par
    pf = m
    ks = [int(np.floor(x)) for x in [L/100, L/20, L/10, L/4]]
    ps = [int(np.floor(x)) for x in [n/40, n/20, n/10, n/5]]

    kp = itertools.product(ks, ps)

    for (k, p) in kp:
      print m, pf, n, k, p, L, G, theta

      if feasible_param(m, pf, n, k, p, L, G, theta):

        metrics = analyze_clustering(m, pf, n, k,  p, L, G, theta)
        row = list(par) + metrics
        # write row to csv

def feasible_param(m,pf, n, k, p, L, G, theta):
  """Check that parameter values make sense."""

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

def analyze_clustering(m, pf, n, k, p, L, G, theta):

  Yf, Yv, Af, Av, U, W = gen_data(m, pf, n, k, p, L, G, theta)
  X = U*W
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

  metrics = [mindivY, maxdivY, avgdivY, mindivX, maxdivX, avgdivX, ami_yw,
  ami_yx, ami_xw, p_xy, s_xy, p_wx, s_wx, p_wy, s_wy]

  return metrics

def compare_distances(X,Y):
  """Get Pearson and Spearman correlations between pairwise distance of columns
  in X and Y. X and Y must have the same number of columns."""

  if X.shape[1] != Y.shape[1]:
    raise ValueError('X and Y must have the same number of columns.')

  X = X.getA()
  Y = Y.getA()
  dist_x = distance.pdist(X.T,'euclidean')
  dist_y = distance.pdist(Y.T,'euclidean')

  p = (1 - distance.correlation(dist_x,dist_y))
  spear = spearmanr(dist_x,dist_y)

  return p,spear[0]

def correlations(X,Y):
  """Get Pearson and Spearman correlations of X & Y flattened, and Pearson corr
  across rows and columns of X and Y. X and Y should have the same dimension."""

  if X.shape != Y.shape :
    raise ValueError('X and Y should have the same dimension.')

  flatX = X.getA1()
  flatY = Y.getA1()
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
