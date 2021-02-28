from scipy.stats import multivariate_normal
from ..utils.utils_iter import map_list
import numpy as np
from functools import partial

class QDA():
    def __init__(self, post_means=None, post_covs=None,classes=None):
        self.prior_means = None # hyperparameter, will be set to empirical mean
        self.prior_mean_conf = 0.01 # hyperparameter 
        self.prior_covs = None # hyperparameter, will be set to a diagonal matrix based on the data
        self.prior_cov_conf = None # hyperparameter, will be set to vector dimensionality + 2 

        self.post_means = post_means # will be set to the mode of the posterior
        self.post_covs = post_covs # will be set to the mode of the posterior
        self.classes_ = classes

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        x1s = X[y==1] 
        x2s = X[y==0]
        num_items = [len(x1s), len(x2s)]

        def _calculate_uncentered_ssq(matrix):
            return np.apply_along_axis(lambda arr: np.outer(arr, arr), -1, matrix).sum(axis=0)

        empirical_covs = [np.cov(x1s, rowvar=False, bias=True), np.cov(x2s, rowvar=False, bias=True)]
        ssq_uncentereds = [_calculate_uncentered_ssq(x1s), _calculate_uncentered_ssq(x2s)]
        empirical_means = [x1s.mean(axis=0), x2s.mean(axis=0)]
        self.prior_means = empirical_means
        self.prior_cov_conf = x1s.shape[1] + 2
        # self.prior_covs = map_list(lambda m: np.diag(m.diagonal()), empirical_covs)
        self.prior_covs = map_list(lambda m: np.diag(m.diagonal()), empirical_covs)

        post_mean_confs = map_list(lambda N: N + self.prior_mean_conf, num_items)
        self.post_means = map_list(lambda point: (self.prior_mean_conf * point[0] + point[1] * point[0])/point[2], 
            zip(empirical_means, num_items, post_mean_confs))
        post_cov_confs = map_list(lambda N: N + self.prior_cov_conf, num_items)
        
        # def _update_cov(prior_cov, emp_cov, prior_mean_conf, post_mean_conf, prior_mean, post_mean):
        def _update_cov(prior_cov, ssq_uncentered, prior_mean_conf, post_mean_conf, prior_mean, post_mean):
            # return prior_cov + emp_cov + (prior_mean_conf * np.outer(prior_mean, prior_mean)) - (post_mean_conf * np.outer(post_mean, post_mean))  
            return prior_cov + ssq_uncentered + (prior_mean_conf * np.outer(prior_mean, prior_mean)) - (post_mean_conf * np.outer(post_mean, post_mean))  
        
        cov_params_zipped = zip(self.prior_covs, ssq_uncentereds, len(empirical_covs) * [self.prior_mean_conf], post_mean_confs, self.prior_means, self.post_means)
        post_covs_unscaled = map_list(lambda item: _update_cov(item[0], item[1], item[2], item[3], item[4], item[5]), cov_params_zipped)
        post_covs = map_list(lambda item: item[0]/(item[1] + x1s.shape[1] + 2), zip(post_covs_unscaled, post_cov_confs) )
        self.post_covs = post_covs

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        preds = log_probs.argmax(axis=1) 
        preds = np.array(map_list(lambda x: 1 if x == 0 else 0, preds))
        return self.classes_[preds]

    def predict_log_proba(self, X):
        """
        """
        mvns = map_list(lambda mean_cov_pair: multivariate_normal(mean_cov_pair[0], mean_cov_pair[1]), zip(self.post_means, self.post_covs))
        # NOTE: this is doesn't take into account the class probabiltiies 
        calculate_mvn_logpdf = lambda mvn, vec: mvn.logpdf(vec) 
        log_prob_c1 = np.apply_along_axis(partial(calculate_mvn_logpdf, mvns[0]), 1, X)
        log_prob_c2 = np.apply_along_axis(partial(calculate_mvn_logpdf, mvns[1]), 1, X)
        log_probs = np.column_stack((log_prob_c1, log_prob_c2))

        return log_probs
