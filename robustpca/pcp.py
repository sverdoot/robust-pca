import numpy as np
import scipy as sp
import scipy.linalg

from tqdm import tqdm

from.utils import time_printer


def shrinkage(mat: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(mat) * np.maximum(np.abs(mat) - thresh, np.zeros(mat.shape))

def sv_thresholding(mat: np.ndarray, thresh: float) -> np.ndarray:
    U, s, V = sp.linalg.svd(mat, full_matrices=False)
    s = shrinkage(s, thresh)
    return np.dot(U, np.dot(np.diag(s), V))
    # return U @ np.diag(s) @ V


class PCP:
    def __init__(self):
        pass

    @staticmethod
    def term_criteria(D, L, S, tol=1e-3):
        diff = np.linalg.norm(D - L - S, ord='fro') / np.linalg.norm(D, ord='fro')
        if np.linalg.norm(D - L - S, ord='fro') / np.linalg.norm(D, ord='fro') < tol:
            return True, diff
        else:
            return False, diff

    @staticmethod
    def default_mu(data_mat):
        return .25 / np.abs(data_mat).mean()

    @time_printer
    def decompose(self, data_mat, mu, max_iter=1e4, tol=1e-7, verbose=False):
        n, m = data_mat.shape
        lamda = 1. / (max(n, m))**.5
        mu_inv = mu ** (-1)
        S = np.zeros(data_mat.shape)
        Y = np.zeros(data_mat.shape)
        L = np.zeros(data_mat.shape)

        it = 0
        while not self.term_criteria(data_mat, L, S, tol=tol)[0] and it < max_iter:
            L = sv_thresholding(data_mat - S + mu_inv * Y, mu_inv)
            S = shrinkage(data_mat - L + mu_inv * Y, lamda * mu_inv)
            Y = Y + mu * (data_mat - L - S)
            it += 1

        if verbose:
            print(f'Iteration: {it}, error: {self.term_criteria(data_mat, L, S, tol=tol)[1]}, terminating alg.')
        
        return L, S