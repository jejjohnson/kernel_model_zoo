import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, check_array
from sklearn.metrics.pairwise import rbf_kernel
from scipy import stats
from typing import Optional
from scipy.spatial.distance import pdist, squareform


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
    method: str = "median",
    percent: Optional[float] = 0.15,
    scale: float = 1.0,
    random_state: Optional[int] = None,
) -> float:
    """A function to provide a reasonable estimate of the sigma values
    for the RBF kernel using different methods. 
    Parameters
    ----------
    X : array, (n_samples, d_dimensions)
        The data matrix to be estimated.

    method : str, default: 'median'
        different methods used to estimate the sigma for the rbf kernel
        matrix.
        * Mean
        * Median
        * Silverman
        * Scott - very common for density estimation

    percent : float, default=0.15
        The kth percentage of distance chosen

    scale : float, default=None
        Option to scale the sigma chosen. Typically used with the
        median or mean method as they are data dependent.
    
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
    # print(method, percent)
    if method == "mean" and percent is None:
        sigma = np.mean(pdist(X))

    elif method == "mean" and percent is not None:
        kth_sample = int(percent * n_samples)
        sigma = np.mean(np.sort(squareform(pdist(X)))[:, kth_sample])

    elif method == "median" and percent is None:
        sigma = np.median(pdist(X))

    elif method == "median" and percent is not None:
        kth_sample = int(percent * n_samples)
        sigma = np.median(np.sort(squareform(pdist(X)))[:, kth_sample])

    elif method == "silverman":
        sigma = np.power(
            n_samples * (d_dimensions + 2.0) / 4.0, -1.0 / (d_dimensions + 4)
        )

    elif method == "scott":
        sigma = np.power(n_samples, -1.0 / (d_dimensions + 4))

    else:
        raise ValueError('Unrecognized mode "{}".'.format(method))

    # scale the sigma by a factor
    if scale is not None:
        sigma *= scale

    # return sigma
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
