import numpy as np
import numpy.random as rd
import math
import scipy as sp
import spams
from scipy.stats import entropy
from sparse_optimization import SparseOptimization
import pred_analysis as pa
from generate_data import gen_data
from clustering import cluster_lab
from scipy.spatial import distance
import csv
from sklearn.linear_model import OrthogonalMatchingPursuit as omp


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
      d = max(5,len(cidx)//20)
    else:
      d = max(5,len(cidx)//10)
    a = Av[cidx]
    y = Yv[:,cidx]
    u,wc = get_cluster_modules(a,y,d,pf,dict_lda)
    U0 = np.hstack([U0,u])
    w = np.zeros((wc.shape[0],n))
    w[:,cidx] = wc
    W0 = np.vstack([W0,w])

  return U0, W0

def refine(Yv, Av, U0, W0, X, lda, pf, niter):

  G = X.shape[0]
  n = X.shape[1]
  U = U0
  W = W0
  X0 = U0.dot(W0)
  fit0 = pa.estimate_diff(X0, X)
  fits = [fit0]
  for _ in range(niter):
    U = cDL(Yv, Av, W, U, lda, pf, sample_average_loss=False)
    W = get_W(Yv, Av , U, n, k=10)
    Xhat = U.dot(W)
    fit = pa.estimate_diff(Xhat, X)
    fits.append(fit)

  return U, W, fits

def run_cs(Yv, Av, U, W,nonneg):
  n = Av.shape[0]
  X  = U.dot(W)

  W0 = get_W(Yv, Av , U, n, k=20)
  Xhat = U.dot(W0)
  if nonneg:
    Xhat[(Xhat < 0)] = 0

  print 'Fit: %f' % (pa.estimate_diff(Xhat, X))

  return W0

def run_bcs(lY, Yv, Av, lda, U, W, pf, niter):

  X = U.dot(W)

  U0, W0 =dict_from_clusters(lY, Av, Yv, lda, U, W, pf)
  Lhat0 =U0.shape[1]
  Xhat0 = U0.dot(W0)
  print 'init fit: %f' % (pa.estimate_diff(Xhat0 , X))

  U1, W1, fits = refine(Yv, Av, U0, W0, X, lda, pf, niter)
  Lhat1 =U1.shape[1]
  Xhat1 = U1.dot(W1)
  print 'final fit: %f,' % (pa.estimate_diff(Xhat1, X))

  return U0, W0, U1, W1, fits

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
