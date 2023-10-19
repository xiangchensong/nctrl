import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy import optimize
from sklearn.metrics import accuracy_score
from .munkres import Munkres
from itertools import permutations
def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    # print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort

def compute_mcc_tdrl(mus_train, ys_train, correlation_fn):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}
  result = np.zeros(mus_train.shape)
  result[:ys_train.shape[0],:ys_train.shape[1]] = ys_train
  for i in range(len(mus_train) - len(ys_train)):
    result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
  corr_sorted, sort_idx, mu_sorted = correlation(mus_train, result, method=correlation_fn)
  mcc = np.mean(np.abs(np.diag(corr_sorted)[:len(ys_train)]))
  return mcc
def compute_mcc_aapo(z_est, z, correlation_fn):
    dim = z_est.shape[1]

    # calculate correlations
    if correlation_fn == "Pearson":
        corr = np.corrcoef(z, z_est, rowvar=False)
        corr = corr[:dim, dim:]
    elif correlation_fn == "Spearman":
        corr, pvals = sp.stats.spearmanr(z, z_est)
        corr = corr[:dim, dim:]

    # sort variables to try find matching components
    ridx, cidx = sp.optimize.linear_sum_assignment(-np.abs(corr))

    # calc with best matching components
    mean_abs_corr = np.mean(np.abs(corr[ridx, cidx]))
    s_est_sorted = z_est[:, cidx]
    return mean_abs_corr #, s_est_sorted, cidx

def compute_mcc(z_est, z, correlation_fn):
    return compute_mcc_tdrl(z_est, z, correlation_fn)

def compute_acc(cs_true, cs_est, C):
    """compute the clustering accuracy"""

    # compute the linear assignment
    cs_true,cs_est = cs_true.flatten(),cs_est.flatten()
    cm = np.zeros((C,C))
    for i in range(C):
        for j in range(C):
            est_i_idx = (cs_est == i).astype(int)
            true_j_idx = (cs_true == j).astype(int)
            cm[i,j] = -np.sum(est_i_idx == true_j_idx)
    _, matchidx = optimize.linear_sum_assignment(cm)
    cs_est_trans = np.array([matchidx[s] for s in cs_est])
    return np.mean((cs_est_trans == cs_true).astype(int)), matchidx

def compute_min_A_err(A, A_est):
    min_A_err = np.inf
    min_permutation = None
    for p in permutations(range(A.shape[0])):
        p = np.array(p)
        A_p = A[p,:][:,p]
        A_err = np.abs(A_p - A_est).mean()
        if A_err < min_A_err:
            min_A_err = A_err
            min_permutation = p
    return min_A_err, min_permutation

def translate_c(c_est,p):
    c_p = np.zeros(c_est.shape)
    for i,c in enumerate(c_est):
        c_p[i] = p[c]
    return c_p