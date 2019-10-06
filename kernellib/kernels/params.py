import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, check_array
from sklearn.metrics.pairwise import rbf_kernel
from scipy import stats
from typing import Optional
from scipy.spatial.distance import pdist


class RandomFourierFeatures(BaseEstimator, TransformerMixin):
    """Random Fourier Features Kernel Matrix Approximation
    Author: J. Emmanuel Johnson
    Email : jemanjohnson34@gmail.com
            emanjohnson91@gmail.com
    Date  : 3rd - August, 2018
    """

    def __init__(self, n_components=50, gamma=None, random_state=None):
        self.gamma = gamma
        # Dimensionality D (number of MonteCarlo samples)
        self.n_components = n_components
        self.rng = check_random_state(random_state)
        self.fitted = False

    def fit(self, X, y=None):
        """ Generates MonteCarlo random samples """
        X = check_array(X, ensure_2d=True, accept_sparse="csr")

        n_features = X.shape[1]

        # Generate D iid samples from p(w)
        self.weights = (2 * gamma) * self.rng.normal(
            size=(n_features, self.n_components)
        )

        # Generate D iid samples from Uniform(0,2*pi)
        self.bias = 2 * np.pi * self.rng.rand(self.n_components)

        # set fitted flag
        self.fitted = True
        return self

    def transform(self, X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        if not self.fitted:
            raise NotFittedError(
                "RBF_MonteCarlo must be fitted beform computing the feature map Z"
            )
        # Compute feature map Z(x):
        Z = np.dot(X, self.weights) + self.bias[np.newaxis, :]

        np.cos(Z, out=Z)

        Z *= np.sqrt(2 / self.n_components)

        return Z

    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        if not self.fitted:
            raise NotFittedError(
                "RBF_MonteCarlo must be fitted beform computing the kernel matrix"
            )
        Z = self.transform(X)

        return np.dot(Z, Z.T)


def get_param_grid(
    init_sigma: float = 1.0,
    factor: int = 2,
    n_grid_points: int = 20,
    estimate_params: Optional[dict] = None,
) -> dict:

    if init_sigma is None:
        init_sigma = 1.0

    # create bounds for search space (logscale)
    init_space = 10 ** (-factor)
    end_space = 10 ** (factor)

    # create param grid
    param_grid = np.logspace(
        np.log10(init_sigma * init_space),
        np.log10(init_sigma * end_space),
        n_grid_points,
    )

    return param_grid


def estimate_sigma(
    X: np.ndarray,
    subsample: Optional[int] = None,
    method: str = "mean",
    random_state: Optional[int] = None,
) -> float:
    """A function to provide a reasonable estimate of the sigma values
    for the RBF kernel using different methods. 
    Parameters
    ----------
    X : array, (n_samples, d_dimensions)
        The data matrix to be estimated.
    method : str {'mean'} default: 'mean'
        different methods used to estimate the sigma for the rbf kernel
        matrix.
        * Mean
        * Median
        * Silverman
        * Scott - very common for density estimation
    random_state : int, (default: None)
        controls the seed for the subsamples drawn to represent
        the data distribution
    Returns
    -------
    sigma : float
        The estimated sigma value
        
    Resources
    ---------
    - Original MATLAB function: https://goo.gl/xYoJce
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
           : juan.johnson@uv.es
    Date   : 6 - July - 2018
    """
    X = check_array(X, ensure_2d=True)

    rng = check_random_state(random_state)

    # subsampling
    [n_samples, d_dimensions] = X.shape

    if subsample is not None:
        X = rng.permutation(X)[:subsample, :]

    if method == "mean":
        sigma = np.mean(pdist(X) > 0)

    elif method == "median":
        sigma = np.median(pdist(X) > 0)

    elif method == "silverman":
        sigma = np.power(
            n_samples * (d_dimensions + 2.0) / 4.0, -1.0 / (d_dimensions + 4)
        )

    elif method == "scott":
        sigma = np.power(n_samples, -1.0 / (d_dimensions + 4))

    else:
        raise ValueError('Unrecognized mode "{}".'.format(method))

    return sigma


def gamma_to_sigma(gamma: float) -> float:
    """Transforms the gamma parameter into sigma using the 
    following relationship:
       
                         1
        sigma =  -----------------
                 sqrt( 2 * gamma )
    """
    return 1 / np.sqrt(2 * gamma)


def sigma_to_gamma(sigma: float) -> float:
    """Transforms the sigma parameter into gamma using the 
    following relationship:
       
                      1
         gamma = -----------
                 2 * sigma^2
    """
    return 1 / (2 * sigma ** 2)
