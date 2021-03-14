from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from functools import partial
import pandas as pd
import numpy as np

# TODO: test on toy data.

def _get_gaussian_components(means, covs):
    """Returns an array of gaussian components

    Args:
        means (np.array): K x D matrix of gaussian means
        covs ([type]): K x D x D matrix of gaussian covariances
    """
    return [multivariate_normal(mean_cov[0], mean_cov[1]) for mean_cov in zip(means, covs)]

def _calculate_responsbilities(weights, gaussian_components, x):
    responsibiltiies_unnormed = np.array([mvn.pdf(x) * weight for (mvn, weight) in zip(gaussian_components, weights)])
    responsibilities = responsibiltiies_unnormed / sum(responsibiltiies_unnormed)
    return responsibilities

def _calculate_conditional_likelihoods( gaussian_components, x):
    likelihoods = np.array([mvn.pdf(x) for mvn in gaussian_components])
    return likelihoods

def _get_kmeans_covs(kmeans_labels, X, dims):
    # X is already in a dataframe
    frame = X
    frame['kmeans_label'] = kmeans_labels
    unrolled_covs = frame.groupby(['kmeans_label']).cov().unstack().values
    reshape_to_cov = lambda arr: arr.reshape(dims, dims)
    covs = np.apply_along_axis(reshape_to_cov, 1, unrolled_covs)
    return covs

class GaussianMixtureWithPrior:
    def __init__(self, K, dims):
        self.K = K
        self.dims = dims

        # self.weights = np.random.dirichlet(np.ones(10))
        self.weights_ = None

        self.means_ = None

        self.covs = None

        self.responsibilities = None

        self.lower_bound_ = -float('inf')

    def _calculate_updated_means(self, responsibilities, X):
        def update_mean(component_responsibilities):
            resp_diag = np.diag(component_responsibilities) 
            X_resp_scaled = resp_diag @ X
            return (X_resp_scaled.sum(axis=0)/sum(component_responsibilities))
        return np.array([update_mean(responsibilities[:, k]) for k in range(self.K)])

    def _calculate_updated_covs(self, responsibilities, X, means):
        def update_cov(component_responsibilities, component_mean):
            # resp_diag = np.diag(component_responsibilities) 
            empirical_scatter = np.zeros((X.shape[1], X.shape[1]))
            for i in range(X.shape[0]):
                empirical_scatter += component_responsibilities[i] * np.outer(X[i], X[i])
            empirical_scatter /= component_responsibilities.sum()
            empirical_scatter = empirical_scatter - np.outer(component_mean, component_mean)
            return empirical_scatter
        return np.array([update_cov(responsibilities[:, k], self.means_[k]) for k in range(self.K)])

    def _compute_ll(self, X): 
        gcs = _get_gaussian_components(self.means_, self.covs)
        conditional_likelihood_matrix_per_comp = np.apply_along_axis(partial(_calculate_conditional_likelihoods, gcs), 1, X)
        joint_likelihood_matrix_per_comp = (self.weights_) * conditional_likelihood_matrix_per_comp
        joint_likelihood_vec = joint_likelihood_matrix_per_comp.sum(axis=1)
        log_joint_likelihood_vec = np.log(joint_likelihood_vec)
        return log_joint_likelihood_vec.sum()

    def fit(self, X):
        ############# First step: initialize with k-means clustering
        # NOTE: need to check for ordering issues here.
        kmeans = KMeans(n_clusters=self.K).fit(X)
        self.means_ = kmeans.cluster_centers_
        self.covs = _get_kmeans_covs(kmeans.labels_, X.copy(), self.dims)
        num_points_per_cluster = pd.Series(kmeans.labels_).value_counts(sort=False).values
        self.weights_ = num_points_per_cluster / sum(num_points_per_cluster)

        likelihood_pair = [-float('inf'), None]
        while True:
            ############# 
            N = X.shape[0]
            #### Do E step: calculating responsibilities
            gcs = _get_gaussian_components(self.means_, self.covs)
            resps = np.vstack(X.apply(partial(_calculate_responsbilities, self.weights_, gcs), axis=1).values)# this is going to be an N X K array

            #### Do M step
            self.weights_ = resps.sum(axis=0) / N

            self.means_ = self._calculate_updated_means(resps, X)

            self.covs = self._calculate_updated_covs(resps, X.values, self.means_)

            observed_data_ll = self._compute_ll(X.values)
            self.lower_bound_ = observed_data_ll
            print(f"Current complete log likelihood: {observed_data_ll}")
            print(f"Current means: {self.means_}")
            likelihood_pair[1] = observed_data_ll

            # sanity check
            if likelihood_pair[1] < likelihood_pair[0]:
                assert False, "Model is getting worse!?"

            if likelihood_pair[1] - likelihood_pair[0] > 1e-3:
                likelihood_pair[0] = likelihood_pair[1]
                likelihood_pair[1] = None
                continue
            else:
                print("Converged")
                print(f"Means: {self.means_}")
                print(f"Observed data log-likelihood: {observed_data_ll}")
                break
            
