from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVR


class RFFRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self, n_components=10, gamma=1.0, alpha=1.0, krr_kwargs=None, random_state=None
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.krr_kwargs = krr_kwargs
        self.random_state = random_state

    def fit(self, X, y):
        # fit RFF Model
        self.pipeline = Pipeline(
            [
                (
                    "rff",
                    RBFSampler(
                        n_components=self.n_components,
                        gamma=self.gamma,
                        random_state=self.random_state,
                    ),
                ),
                (
                    "krr",
                    KernelRidge(
                        alpha=self.alpha,
                        kernel="linear",
                        gamma=None,
                        kernel_params=None,
                    ),
                ),
            ]
        )

        # fit LR model
        self.pipeline.fit(X, y)

        return self

    def predict(self, X):
        return self.pipeline.predict(X)


class RFFSVR(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_components=10,
        gamma=1.0,
        epsilon=0.0,
        C=1.0,
        tol=1e-4,
        loss="epsilon_insensitive",
        svr_kwargs=None,
        random_state=None,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.epsilon = epsilon
        self.C = C
        self.tol = tol
        self.loss = loss
        self.svr_kwargs = svr_kwargs
        self.random_state = random_state

    def fit(self, X, y):
        # fit RFF Model
        self.pipeline = Pipeline(
            [
                (
                    "rff",
                    RBFSampler(
                        n_components=self.n_components,
                        gamma=self.gamma,
                        random_state=self.random_state,
                    ),
                ),
                (
                    "svr",
                    LinearSVR(
                        epsilon=self.epsilon,
                        C=self.C,
                        tol=self.tol,
                        loss=self.loss,
                        **self.svr_kwargs,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        # fit LR model
        self.pipeline.fit(X, y)

        return self

    def predict(self, X):
        return self.pipeline.predict(X)


class RFFSGD(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=10, gamma=1.0, random_state=123, sgd_kwargs=None):

        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.sgd_kwargs = sgd_kwargs

    def fit(self, X, y):
        # fit RFF Model
        self.pipeline = Pipeline(
            [
                (
                    "rff",
                    RBFSampler(
                        n_components=self.n_components,
                        gamma=self.gamma,
                        random_state=self.random_state,
                    ),
                ),
                ("sgd", SGDRegressor(**self.sgd_kwargs)),
            ]
        )

        # fit LR model
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
