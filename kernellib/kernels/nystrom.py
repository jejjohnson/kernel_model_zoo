import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise_kernels
from scipy.stats import chi
from numba import jit


class RandomizedNystrom(BaseEstimator, TransformerMixin):
    """Approximation of a kernel map using a subset of
    training data. Utilizes the randomized svd for the
    kernel decomposition to speed up the computations.


    Author: J. Emmanuel Johnson
    Email : jemanjohnson34@gmail.com
            emanjohnson91@gmail.com
    Date  : December, 2017
    """

    def __init__(
        self,
        kernel="rbf",
        sigma=1.0,
        n_components=100,
        k_rank=1,
        random_state=None,
        **kwargs
    ):
        self.kernel = kernel
        self.sigma = sigma
        self.n_components = n_components
        self.k_rank = k_rank
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit estimator to the data"""
        X = check_array(X)
        rnd = check_random_state(self.random_state)

        # gamma parameter for the kernel matrix
        self.gamma = 1 / (2 * self.sigma ** 2)

        n_samples = X.shape[0]
        if self.n_components > n_samples:
            n_components = n_samples
        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)

        indices = rnd.permutation(n_samples)
        basis_indices = indices[:n_components]
        basis = X[basis_indices]

        basis_kernel = pairwise_kernels(basis, metric=self.kernel, gamma=self.gamma)

        # Randomized SVD
        U, S, V = randomized_svd(
            basis_kernel,
            n_components=self.k_rank,
            random_state=self.random_state,
            flip_sign=True,
        )

        S = np.maximum(S, 1e-12)

        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = indices

        return self

    def transform(self, X):
        """Apply the feature map to X."""
        X = check_array(X)

        embedded = pairwise_kernels(
            X, self.components_, metric=self.kernel, gamma=self.gamma
        )

        return np.dot(embedded, self.normalization_.T)

    def compute_kernel(self, X):

        L = self.transform(X)

        return np.dot(L, L.T)
