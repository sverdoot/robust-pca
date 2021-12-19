from typing import Union

import numpy as np
import scipy as sp
import scipy.linalg
from tqdm import tqdm

from .utils import time_printer


def shrinkage(mat: np.ndarray, thresh: Union[np.ndarray, float]) -> np.ndarray:
    return np.sign(mat) * np.maximum(np.abs(mat) - thresh, np.zeros(mat.shape))


def sv_thresholding(mat: np.ndarray, thresh: float) -> np.ndarray:
    U, s, V = sp.linalg.svd(mat, full_matrices=False)
    s = shrinkage(s, thresh)
    return U @ np.diag(s) @ V


class PCP:
    """Robust PCA (Principal Component Percuit) via Augmented Lagrangian Multipliers"""

    def __init__(self):
        pass

    @staticmethod
    def term_criteria(D, L, S, tol=1e-3):
        diff = np.linalg.norm(D - L - S, ord="fro") / np.linalg.norm(D, ord="fro")
        if diff < tol:
            return True, diff
        else:
            return False, diff

    @staticmethod
    def default_mu(data_mat):
        return 0.25 / np.abs(data_mat).mean()

    @time_printer
    def decompose(self, data_mat, mu, max_iter=1e4, tol=1e-7, verbose=False):
        n, m = data_mat.shape
        lamda = 1.0 / (max(n, m)) ** 0.5
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
            print(
                f"Iteration: {it}, error: {self.term_criteria(data_mat, L, S, tol=tol)[1]}, terminating alg."
            )

        return L, S


class StablePCP:
    def __init__(self) -> None:
        pass

    @staticmethod
    def default_mu(data_mat, sigma):
        return 1.0 / ((2 * np.max(data_mat.shape) ** 0.5 * sigma))
        # return ((2 * np.max(data_mat.shape) ** .5 * sigma))

    @staticmethod
    def term_criteria(L, S, L_prev, S_prev, tol=1e-3):
        diff = (
            np.linalg.norm(L - L_prev, ord="fro") ** 2 + np.linalg.norm(S - S_prev, ord="fro") ** 2
        )
        if diff < tol:
            return True, diff
        else:
            return False, diff

    @time_printer
    def decompose(self, data_mat, mu, max_iter=1e4, tol=1e-7, verbose=False):
        n, m = data_mat.shape
        lamda = 1.0 / (max(n, m)) ** 0.5
        mu_inv = mu ** (-1)
        S = np.zeros(data_mat.shape)
        L = np.zeros(data_mat.shape)

        L_prev = L
        S_prev = S

        it = 0
        while (
            not self.term_criteria(L, S, L_prev, S_prev, tol=tol)[0] and it < max_iter
        ) or it == 0:
            L_prev = L
            S_prev = S

            L = sv_thresholding(data_mat - S, mu_inv)
            S = shrinkage(data_mat - L, lamda * mu_inv)
            it += 1

        if verbose:
            print(
                f"Iteration: {it}, diff: {self.term_criteria(L, S, L_prev, S_prev, tol=tol)[1]}, terminating alg."
            )

        return L, S


class CompressedPCP:
    def __init__(self):
        pass

    @staticmethod
    def term_criteria(Y, L, S, C, tol=1e-3):
        diff = np.linalg.norm(Y - L - S @ C, ord="fro") / np.linalg.norm(Y, ord="fro")
        if diff < tol:
            return True, diff
        else:
            return False, diff

    @staticmethod
    def default_mu(Y):
        return 0.25 / np.mean(np.abs(Y))

    @time_printer
    def decompose(self, Y, C, mu, d, max_iter=1e4, tol=1e-7, verbose=False, lamda=None):
        n, m = Y.shape
        lamda = lamda if lamda else 1.0 / (max(n, m)) ** 0.5
        mu_inv = mu ** (-1)
        S = np.zeros((n, d))
        P = np.zeros((n, m))

        it = 0
        while (not self.term_criteria(Y, P, S, C, tol=tol)[0] and it < max_iter) or it == 0:
            P = sv_thresholding(Y - S @ C, mu_inv)
            pinv = np.linalg.pinv(C @ C.T)
            thresh = lamda * (np.ones((n, d)) @ pinv)
            thresh = np.maximum(np.zeros((n, d)), lamda * (np.ones((n, d)) @ pinv))
            S = shrinkage((Y - P) @ C.T @ pinv, thresh)
            it += 1

        if verbose:
            print(
                f"Iteration: {it}, error: {self.term_criteria(Y, P, S, C, tol=tol)[1]}, terminating alg."
            )

        return P, S
