import numpy as np
from utils import *

def pca(mat):
    """
    mat is a N-by-D matrix, N is #data, D is dimension (N>D)
    return eig_vals, eig_vecs sorted by abs(eig_vals) in descending order
    eig_vecs[0] is the eigenvector with the largest eigenvalue
    """
    if mat.shape[0] >= mat.shape[1]:
        eig_vals, eig_vecs = np.linalg.eig(np.cov(mat.T))
        eig_pairs = zip(np.abs(eig_vals), eig_vecs.T)
        eig_pairs.sort(reverse=True)
        eig_vals, eig_vecs = zip(*eig_pairs)
        eig_vals = np.asarray(eig_vals)
        eig_vecs = np.asarray(eig_vecs)
        info("attribute matrix shape: %s" % (mat.shape,))
        info("Eigenvalues \n%s" % (np.asarray(eig_vals),))
        info("Cum Explained Variance \n%s" % (
                np.cumsum(eig_vals / np.sum(eig_vals)),))
    else:
        info("#data < dimension. PCA has complex eigenvalues. ignore for now")
        eig_vals, eig_vecs = None, None
    return eig_vals, eig_vecs
