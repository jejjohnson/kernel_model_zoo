import numpy as np
from typing import Optional, Callable, Union
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel, linear_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from scipy.spatial.distance import pdist
from scipy import stats
from ..kernels.utils import estimate_sigma, get_param_grid, sigma_to_gamma


class HSIC(BaseEstimator):
    """Hilbert-Schmidt Independence Criterion (HSIC). This is
    a method for measuring independence between two variables.
    
    Parameters
    ----------
    alpha : {float, array-like}, shape = [n_targets]
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.  Alpha corresponds to
        ``(2*C)^-1`` in other linear models such as LogisticRegression or
        LinearSVC. If an array is passed, penalties are assumed to be specific
        to the targets. Hence they must correspond in number.
    
    kernel : string or callable, default="linear"
        Kernel mapping used internally. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number. Set to "precomputed" in
        order to pass a precomputed kernel matrix to the estimator
        methods instead of samples.
    
    gamma_X : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Used only by the X parameter. 
        Interpretation of the default value is left to the kernel; 
        see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    gamma_Y : float, default=None
        The same gamma parameter as the X. If None, the same gamma_X will be
        used for the Y.
    
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    
    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.
    
    scorer : str, default='ctka'
        The method to score how well the sigma fits the two datasets.
        
        The following options are:
        * 'ctka': centered target kernel alignment
        * 'tka' : target kernel alignment
        * 'hsic': the hsic value
    
    random_state : str
    Attributes
    ----------
    hsic_value : float
        The HSIC value is scored after fitting.
        
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 14-Feb-2019
    
    Resources
    ---------
    Original MATLAB Implementation : 
        http:// isp.uv.es/code/shsic.zip
    Paper :
        Sensitivity maps of the Hilbert–Schmidt independence criterion
        Perez-Suay et al., 2018
    """

    def __init__(
        self,
        gamma_X: float = 1.0,
        gamma_Y: Optional[None] = None,
        kernel: Union[Callable, str] = "rbf",
        degree: float = 3,
        coef0: float = 1,
        kernel_params: Optional[dict] = None,
        random_state: Optional[int] = None,
        scorer: str = "tka",
        subsample: Optional[int] = None,
        bias: bool = True,
    ):
        self.gamma_X = gamma_X
        self.gamma_Y = gamma_Y
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.scorer = scorer
        self.subsample = subsample
        self.bias = bias

    def fit(self, X, Y):

        # Check sizes of X, Y
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)

        # Check samples are the same
        assert X.shape[0] == Y.shape[0]

        self.n_samples = X.shape[0]
        self.dx_dimensions = X.shape[1]
        self.dy_dimensions = Y.shape[1]

        # subsample data if necessary
        if self.subsample is not None:
            X = self.rng.permutation(X)[: self.subsample, :]
            Y = self.rng.permutation(Y)[: self.subsample, :]

        self.X_train_ = X
        self.Y_train_ = Y

        # Calculate Kernel Matrices
        K_x = self.compute_kernel(X, gamma=self.gamma_X)
        K_y = self.compute_kernel(Y, gamma=self.gamma_Y)

        # Center Kernel
        # H = np.eye(n_samples) - (1 / n_samples) * np.ones(n_samples)
        # K_xc = K_x @ H
        if self.scorer.lower() in ["hsic", "ctka"]:
            K_x = KernelCenterer().fit_transform(K_x)
            K_y = KernelCenterer().fit_transform(K_y)

        # Compute HSIC value
        self.hsic_value = self._calculate_hsic(K_x, K_y)

        return self

    def compute_kernel(self, X, Y=None, gamma=1.0):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _calculate_hsic(self, K_x, K_y):
        if self.scorer == "tka":
            return kernel_alignment(K_x, K_y, center=False)

        elif self.scorer == "ctka":
            return kernel_alignment(K_x, K_y, center=False)

        elif self.scorer == "hsic":

            if self.bias:
                self.hsic_bias = 1 / (self.n_samples - 1) ** 2
            else:
                self.hsic_bias = 1 / (self.n_samples ** 2)

            return self.hsic_bias * np.sum(K_x * K_y)
        else:
            raise ValueError(f"Unrecognized scorer: {self.scorer}")

    def score(self, X, y=None):
        """This is not needed. It's only needed to comply with sklearn API.
        
        We will use the target kernel alignment algorithm as a score
        function. This can be used to find the best parameters."""

        return self.hsic_value


def get_gamma_init(
    X: np.ndarray, Y: np.ndarray, method: str, percent: Optional[float] = None
) -> float:
    """Get Gamma initializer
    
    Parameters
    ----------
    method : str,
        the initialization method
        
    percent : float
        if using the Belkin method, this uses a percentage
        of the kth nearest neighbour
    
    Returns
    -------
    gamma_init : float
        the initial gamma value
    """

    # initialize sigma
    sigma_init_X = estimate_sigma(X, method=method, percent=percent)
    sigma_init_Y = estimate_sigma(Y, method=method, percent=percent)

    # mean of the two
    sigma_init = np.mean([sigma_init_X, sigma_init_Y])

    # convert sigma to gamma
    gamma_init = sigma_to_gamma(sigma_init)

    # return initial gamma value
    return gamma_init


def train_rbf_hsic(
    X: np.ndarray,
    Y: np.ndarray,
    scorer: str = "ckta",
    n_gamma: int = 100,
    factor: int = 1,
    sigma_est: str = "mean",
    verbose=0,
    n_jobs=-1,
    cv=2,
) -> dict:

    # Estimate Sigma
    sigma_x = estimate_sigma(X, method=sigma_est)
    sigma_y = estimate_sigma(Y, method=sigma_est)

    # init overall sigma is mean between two
    init_sigma = np.mean([sigma_x, sigma_y])

    # get sigma parameter grid
    sigmas = get_param_grid(init_sigma, factor, n_gamma)

    gammas = sigma_to_gamma(sigmas)
    init_gamma = sigma_to_gamma(init_sigma)
    param_grid = {"gamma": gammas}

    # Get HSIC model
    clf_hsic = HSIC(
        gamma=init_gamma, kernel="rbf", scorer=scorer, subsample=None, bias=True
    )

    # print(n_jobs, verbose)

    clf_grid = GridSearchCV(
        clf_hsic, param_grid, iid=False, n_jobs=n_jobs, verbose=verbose, cv=cv
    )

    clf_grid.fit(X, Y)

    # print results
    if verbose:
        print(
            f"Best {clf_grid.best_estimator_.scorer.upper()} score: {clf_grid.best_score_:.5f}"
        )
        print(f"gamma: {clf_grid.best_estimator_.gamma:.3f}")

    return clf_grid.best_estimator_


def train_hsic(
    X: np.ndarray,
    Y: np.ndarray,
    clf_hsic: BaseEstimator,
    param_grid: dict,
    verbose=None,
    n_jobs=-1,
    cv=2,
) -> dict:

    clf_grid = GridSearchCV(
        clf_hsic, param_grid, iid=False, n_jobs=n_jobs, verbose=verbose, cv=cv
    )

    clf_grid.fit(X, Y)

    # print results
    if verbose:
        print(
            f"Best {clf_grid.best_estimator_.scorer.upper()} score: {clf_grid.best_score_:.3e}"
        )
        print(f"gamma: {clf_grid.best_estimator_.gamma:.3f}")

    return clf_grid.best_estimator_


def kernel_alignment(K_x: np.array, K_y: np.array, center: bool = False) -> float:
    """Gives a target kernel alignment score: how aligned the kernels are. Very
    useful for measures which depend on comparing two different kernels, e.g.
    Hilbert-Schmidt Independence Criterion (a.k.a. Maximum Mean Discrepency)
    
    Note: the centered target kernel alignment score is the same function
          with the center flag = True.
    
    Parameters
    ----------
    K_x : np.array, (n_samples, n_samples)
        The first kernel matrix, K(X,X')
    
    K_y : np.array, (n_samples, n_samples)
        The second kernel matrix, K(Y,Y')
        
    center : Bool, (default: False)
        The option to center the kernels (independently) before hand.
    
    Returns
    -------
    kta_score : float,
        (centered) target kernel alignment score.
    """

    # center kernels
    if center:
        K_x = KernelCenterer().fit_transform(K_x)
        K_y = KernelCenterer().fit_transform(K_y)

    # target kernel alignment
    return np.sum(K_x * K_y) / np.linalg.norm(K_x) / np.linalg.norm(K_y)

