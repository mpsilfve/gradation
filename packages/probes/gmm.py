from ..utils.utils_iter import map_list
import numpy as np
from sklearn.mixture import GaussianMixture
from .gmm_prior import GaussianMixtureWithPrior

# TODO: modify this to use your GMM.

class GaussianMixtureDiscriminantAnalysis:
    def __init__(self, gmm1=None, gmm2=None, cov_type='diag', n_components=3, classes=None):
        self.gmm1 = gmm1
        self.gmm2 = gmm2
        self.cov_type = cov_type #NOTE: this will be ignored for now
        self.n_components = n_components
        self.classes_ = classes

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)

        x1s = X[y==1] 
        x2s = X[y==0]

        # self.gmm1 = GaussianMixture(covariance_type=self.cov_type, n_components=self.n_components)
        # self.gmm2 = GaussianMixture(covariance_type=self.cov_type, n_components=self.n_components)
        self.gmm1 = GaussianMixtureWithPrior(self.n_components, x1s.shape[1])
        self.gmm2 = GaussianMixtureWithPrior(self.n_components, x1s.shape[1])

        self.gmm1.fit(x1s)
        self.gmm2.fit(x2s)

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        preds = log_probs.argmax(axis=1) 
        preds = np.array(map_list(lambda x: 1 if x == 0 else 0, preds))
        return self.classes_[preds]

    def predict_log_proba(self, X):
        log_prob_c1 = self.gmm1.score_samples(X)
        log_prob_c2 = self.gmm2.score_samples(X)

        log_probs = np.column_stack((log_prob_c1, log_prob_c2))
        return log_probs