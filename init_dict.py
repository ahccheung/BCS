import numpy as np
import numpy.random as rd
import math
import scipy as sp
import spams
from scipy.stats import entropy
from sparse_optimization import SparseOptimization
from pred_analysis import *
from generate_data import gen_data
from clustering import cluster_lab
from scipy.spatial import distance
import csv
from sklearn.linear_model import OrthogonalMatchingPursuit as omp

def test_cs(m, n, k, p, L, G, r, lda):

  pf = m//5

  Yf, Yv, Af, Av, U, W = gen_data(m, pf, n, k, p, L, G, r)
  X = U.dot(W)

  p_wy, s_wy = compare_distances(W, Yf)
  p_xy, s_xy = compare_distances(X, Yf)
  p_wx, s_wx = compare_distances(W, X)

  lY = cluster_lab(Yf)
  ami_yx = compare_clusters(Yf, X)
  ami_xw = compare_clusters(X, W)
  ami_yw = compare_clusters(Yf, W)

  Uinit, Winit, Ub, Wb = run_bcs(lY, Yv, Av, lda, U, W, pf)
  X0 = Uinit.dot(Winit)
  Xbcs = Ub.dot(Wb)
  init_err = estimate_diff(X0, X)
  bcs_err = estimate_diff(Xbcs,X)
  corrs= abs(1 - distance.cdist(Ub.T, U.T, 'correlation'))
  minc = np.amin(corrs)
  maxc = np.amax(corrs)

  Wcs = run_cs(Yv, Av, U, W, False)
  Xcs = U.dot(Wcs)
  cs_err = estimate_diff(Xcs, X)

  x = [p_wy, s_wy, p_xy, s_xy, p_wx, s_wx, ami_yx, ami_xw, ami_yw, init_err,
  bcs_err, minc, maxc, cs_err]

  return x

def run_test(filename, ms, ks, ldas):
  csvfile = open(filename, 'wb')
  writer = csv.writer(csvfile)

  par = ['pf', 'n', 'k', 'p', 'L', 'G', 'r', 'lda']
  metrics = ['p_wy', 's_wy', 'p_xy', 's_xy', 'p_wx', 's_wx', 'ami_yx', 'ami_xw',
  'ami_yw', 'init_err',  'bcs_err','minc', 'maxc', 'cs_err']
  row = par + metrics
  writer.writerow(row)

  G = 1000
  n = 200
  L = 100
  p = 10
  #ms = [G, G//2, G//5, G//10, G//20, G//40]
  #ks = [1, 2, 5, 8]
  r = 0.2
  #ldas = [5, 25, 45, 65, 85]
  params = itertools.product(ms, ks, ldas)
  for (m, k, lda) in params:
    pf = m//5
    for i in range(0, 5):
      print "m, n, k, p, L, G, r, lda "
      print m, n, k, p, L, G, r, lda
      par = [m, n, k, p, L, G, r, lda]
      metrics = test_cs(m, n, k, p, L, G, r, lda)
      row = par + metrics
      writer.writerow(row)

  csvfile.close()


def dict_from_clusters(lY, Av, Yv, lda, U, W, pf):
  """Learn init dictionary and module activity U0, W0 based on clusters of var
  observations/measurements Yv and Av. lY are cluster labels for the samples."""

  n = Av.shape[0]
  m = Av.shape[1]
  G = Av.shape[2]
  pv = m - pf
  U0 = np.zeros((G, 0))
  W0 = np.zeros((0, n))
  X  = U.dot(W)
  dict_lda = lda

  for c in set(lY):
    cidx = np.where(lY == c)[0]
    if n > 1000:
      d = max(5,len(cidx)/20)
    else:
      d = max(5,len(cidx)/10)

    a = Av[cidx]
    y = Yv[:,cidx]
    u,wc = get_cluster_modules(a,y,d,pf,dict_lda)
    U0 = np.hstack([U0,u])
    w = np.zeros((wc.shape[0],n))
    w[:,cidx] = wc
    W0 = np.vstack([W0,w])
    xhat = u.dot(wc)
    pearson,spearman,gene_pearson,sample_pearson = correlations(X[:,cidx],xhat)
    var_fit = 1-np.linalg.norm(X[:,cidx]-xhat)**2/np.linalg.norm(X[:,cidx])**2
    uent = np.average([np.exp(entropy(u)) for u in U0.T])
    #print c,X.shape[0],len(cidx), pearson,spearman,gene_pearson,sample_pearson,var_fit,uent

  return U0, W0

def refine(Yv, Av, U0, W0, X, lda, pf):

  G = X.shape[0]
  n = X.shape[1]
  U = U0
  W = W0

  for _ in range(5):
    U = cDL(Yv, Av, W, U, lda, pf, sample_average_loss=False)
    W = get_W(Yv, Av , U, n, k=10)
    Xhat = U.dot(W)
    Xhat[(Xhat < 0)] = 0
    pearson,spearman,gene_pearson,sample_pearson= correlations(X,Xhat)
    var_fit = 1-np.linalg.norm(X - Xhat)**2/np.linalg.norm(X)**2
    uent = np.average([np.exp(entropy(u)) for u in U.T])
    #print G, n,pearson,spearman,gene_pearson,sample_pearson,var_fit,uent

  return U, W

def run_cs(Yv, Av, U, W,nonneg):
  n = Av.shape[0]
  X  = U.dot(W)

  W0 = get_W(Yv, Av , U, n, k=20)
  Xhat = U.dot(W0)
  if nonneg:
    Xhat[(Xhat < 0)] = 0

  pearson,spearman,gene_pearson,sample_pearson= correlations(X,Xhat)
  var_fit = 1-np.linalg.norm(X - Xhat)**2/np.linalg.norm(X)**2
  #print  n,pearson,spearman,gene_pearson,sample_pearson,var_fit

  print 'Fit: %f' % (estimate_diff(Xhat, X))

  return W0

def run_bcs(lY, Yv, Av, lda, U, W, pf):

  X = U.dot(W)

  U0, W0 =dict_from_clusters(lY, Av, Yv, lda, U, W, pf)
  Lhat0 =U0.shape[1]
  Xhat0 = U0.dot(W0)
  print 'init fit: %f' % (estimate_diff(Xhat0 , X))

  U1, W1 = refine(Yv, Av, U0, W0, X, lda, pf)
  Lhat1 =U1.shape[1]
  Xhat1 = U1.dot(W1)
  print 'final fit: %f,' % (estimate_diff(Xhat1, X))

  return U0, W0, U1, W1

def get_cluster_modules(Av, Yv, d, pf, lda, maxItr=5):
  """Learn U and W from a single cluster."""

  G = Av.shape[2]
  n = Yv.shape[1]

  k = min(d, 20)
  U = rd.random((G,d))
  U = U/np.linalg.norm(U,axis=0)
  W = get_W(Yv, Av, U, n ,k=k)

  for itr in range(maxItr):
    U = cDL(Yv, Av, W, U, lda, pf, sample_average_loss=False)
    W = get_W(Yv, Av, U, n, k=k)

  return U, W


def get_W(Yv, Av, U, n, k=5):
  """Use OMP to produce estimate of W, fixing U."""

  L = U.shape[1]
  W = np.empty((L, n))
  for i in range(0, n):
    Wi = sparse_decode(Yv[:,i:i+1],Av[i].dot(U),k,mink=min(5,k-1))
    W[:,i] = Wi

  return W

def sparse_decode(Yv,AU,k,worstFit=1.,mink=4, threads=4):
  while k > mink:

    W = omp(k, tol=None, fit_intercept=True, normalize=True, precompute='auto').fit(AU, Yv)
    W = W.coef_.T
    fit = 1 - np.linalg.norm(Yv - AU.dot(W))**2/np.linalg.norm(Yv)**2
    if fit < worstFit:
      break
    else:
      k -= 1

  return W

def cDL(Y,Av,W,U,lda1,pf,maxItr=40,with_prints=False,nonneg=True,forceNorm=True,sample_average_loss=False):

  snl = SparseOptimization()
  snl.Y = Y.flatten()[:,np.newaxis]
  snl.Ynorm = np.linalg.norm(Y)
  snl.U = U
  def get_yhat(U):
    uw = U.reshape(snl.U.shape).dot(W)
    yhat = np.zeros(Y.shape)
    for i in range(yhat.shape[1]):
      yhat[:,i] = Av[i].dot(uw[:,i])
    return yhat.flatten()[:,np.newaxis]
  def proximal_optimum(U,delta,nonneg=False,forceNorm=False):
    Z = U.reshape(snl.U.shape)
    if delta > 0:
      z = (Z - delta*np.sign(Z))*(abs(Z) > delta)
    else:
      z = Z
    if nonneg:
      z[(z < 0)] = 0
    elif hasattr(snl,'prox_bounds'):
      z = np.maximum(z,self.prox_bounds[0])
      z = np.minimum(z,self.prox_bounds[1])
    if forceNorm:
      z = z/np.linalg.norm(z,axis=0)
      z[np.isnan(z)] = 0
    return z.flatten()[:,np.newaxis]
  if sample_average_loss:
    def grad_U(U,resid):
      r = resid.reshape(Y.shape)
      wgrad = np.zeros(U.shape)
      for i in range(r.shape[1]):
        wgrad += np.outer(Av[i].T.dot(r[:,i]),W[:,i]).flatten()[:,np.newaxis]
      return wgrad
    def get_resid(Yhat):
      resid = (Yhat.reshape(Y.shape) - Y)
      resid_0 = (resid[pf:]**2).sum(0)**.5 + 1e-3
      resid[pf:] = resid[pf:]/resid_0/Y.shape[1]
      resid_1 = (resid[:pf]**2).sum(1)**.5 + 1e-3
      resid[:pf] = (resid[:pf].T/resid_1/Y.shape[0]).T
      return resid.flatten()[:,np.newaxis]*snl.Ynorm
    def simple_loss(U,lda1):
      Yhat = get_yhat(U).reshape(Y.shape)
      loss = np.average(((Yhat[pf:] - Y[pf:])**2).sum(0)**.5)
      loss += np.average(((Yhat[:pf] - Y[:pf])**2).sum(1)**.5)
      return loss*snl.Ynorm + lda1*abs(U).sum()
  else:
    def grad_U(U,resid):
      r = resid.reshape(Y.shape)
      wgrad = np.zeros(U.shape)
      for i in range(r.shape[1]):
        wgrad += np.outer(Av[i].T.dot(r[:,i]),W[:,i]).flatten()[:,np.newaxis]
      return wgrad
    def get_resid(Yhat):
      return Yhat - snl.Y
    def simple_loss(U,lda1):
      Yhat = get_yhat(U)
      loss = 0.5*np.linalg.norm(Yhat - snl.Y)**2
      return loss + lda1*abs(U).sum()

  snl.get_Yhat = get_yhat
  snl.get_grad = grad_U
  snl.get_resid = get_resid
  snl.simple_loss = simple_loss
  snl.proximal_optimum = proximal_optimum
  lda = lda1*np.linalg.norm(grad_U(U.flatten()[:,np.newaxis],snl.Y).reshape(U.shape))/np.product(U.shape)*(np.log(U.shape[1])/Y.shape[1])**.5
  U1 = snl.nonlinear_proxGrad(lda,U.flatten()[:,np.newaxis],maxItr=maxItr,with_prints=with_prints,fa_update_freq=1e6,nonneg=nonneg,forceNorm=forceNorm)
  snl = None

  return U1.reshape(U.shape)
