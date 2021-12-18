import numpy as np
import scipy as sp
import scipy.linalg

from .utils import time_printer


def thresholding(mat: np.ndarray, threshold: float) -> np.ndarray:
    # new_mat = np.copy(mat)
    # new_mat[np.abs(new_mat) < threshold] = 0
    # return new_mat
    return np.sign(mat) * np.maximum(np.abs(mat) - threshold, np.zeros(mat.shape))


def best_approximator(mat: np.ndarray, rank: float) -> np.ndarray:
    U, s, V = sp.linalg.svd(mat, full_matrices=False)
    return U[:, :rank] @ np.diag(s[:rank]) @ V[:rank, :]


class IRCUR:
    def __init__(self) -> None:
        pass

    @staticmethod
    def term_criteria(data_mat, L, S, I, J, tol=1e-3):
        diff = np.linalg.norm((data_mat - L - S)[I, :], ord="fro") + np.linalg.norm(
            (data_mat - L - S)[:, J], ord="fro"
        )
        diff /= np.linalg.norm((data_mat)[I, :], ord="fro") + np.linalg.norm(
            (data_mat)[:, J], ord="fro"
        )

        if diff < tol:
            return True, diff
        else:
            return False, diff

    @time_printer
    def decompose(
        self,
        data_mat,
        rank,
        nr,
        nc,
        initial_threshold,
        tol=1e-5,
        thresholding_decay=0.65,
        resample=True,
        max_iter=1e4,
        verbose=False,
    ):
        n, m = data_mat.shape
        nr = min(nr, n)
        nc = min(nc, m)
        L = np.zeros(data_mat.shape)
        S = np.zeros(data_mat.shape)
        I = np.random.choice(np.arange(n), nr, replace=True)
        J = np.random.choice(np.arange(m), nc, replace=True)
        it = 0

        while not self.term_criteria(data_mat, L, S, I, J, tol=tol)[0] and it < max_iter:
            if resample:
                I = np.random.choice(np.arange(n), nr, replace=True)
                J = np.random.choice(np.arange(m), nc, replace=True)

            threshold = thresholding_decay ** it * initial_threshold
            S[:, J] = thresholding(data_mat - L, threshold)[:, J]
            S[I, :] = thresholding(data_mat - L, threshold)[I, :]

            C = (data_mat - S)[:, J]
            R = (data_mat - S)[I, :]
            U = best_approximator((data_mat - S)[I, :][:, J], rank)
            L = C @ sp.linalg.pinv(U) @ R

            it += 1

        if verbose:
            print(
                f"Iteration: {it}, diff: {self.term_criteria(data_mat, L, S, I, J, tol=tol)[1]}, terminating alg."
            )

        return L, S
