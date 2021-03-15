from scipy import linalg
import sys
from sklearn.mixture import GaussianMixture
import argparse

from packages.gradation_utils.utils import *
from packages.probes.gmm import *
from packages.probes.gmm import GaussianMixtureDiscriminantAnalysis
from packages.gradation_utils.evaluation import *
from packages.gradation_utils.visualizations import *
from packages.visualizations.visualize_gaussian import *
from packages.visualizations.probing_visualizations import *
from packages.constants.constants import *
from packages.pkl_operations.pkl_io import *
from packages.utils.utils_save import *

def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.
    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol

def get_marginal_parameters(gmm, dims, n_components, cov_type='diag'):
    ##### returns a gmm with the dims parameters only
    gmm_new = GaussianMixture(n_components=n_components, covariance_type=cov_type)
    gmm_new.weights_ = gmm.weights_
    gmm_new.means_ = np.array(map_list(lambda m: m[dims], gmm.means_))
    # new_weights_ = gmm.weights_
    # new_means_ = np.array(map_list(lambda m: m[dims], gmm.means_))

    # NOTE: lines below won't work if we don't use a diagonal covariance.
    # new_covs = np.array(map_list(lambda cov: cov[dims], gmm.covariances_))
    # new_precisions = np.array(map_list(lambda prec: prec[dims], gmm.precisions_))
    gmm_new.covariances_ = np.array(map_list(lambda cov: cov[dims], gmm.covariances_))
    # gmm_new.precisions_ = np.array(map_list(lambda prec: prec[dims], gmm.precisions_))
    # gmm_new.precisions_cholesky_ = _compute_precision_cholesky(gmm_new.covariances_, gmm.covariance_type)

    # gmm_new = GaussianMixture(n_components=n_components, weights_init=new_weights_, means_init=new_means_, precisions_init=new_precisions)
    return gmm_new

def obtain_probing_dimensions_gmm(train_frame, validation_frame, num_dims_to_select=10, n_components=2, cov_type='diag'):
    dims_selected = []
    best_likelihoods = []
    full_model = GaussianMixtureDiscriminantAnalysis(cov_type=cov_type, n_components=n_components)
    all_dims = map_list(str, ENCODER_DIMS)
    full_model.fit(train_frame[all_dims], train_frame[LABEL_COLUMN])
    y_test = validation_frame[LABEL_COLUMN]
    y_test = y_test.map(lambda x: 0 if x == full_model.classes_[0] else 1)
    labels = y_test.map(lambda x: 0 if x == full_model.classes_[0] else 1)
    label_inds = labels.map(lambda label: 0 if label == 1 else 1)
    for _ in range(num_dims_to_select):
        best_loglik = -float('inf')
        best_dim = -1
        set_dims_selected = set(dims_selected)
        for dim in range(len(ENCODER_DIMS)):
            if dim in set_dims_selected:
                continue
            curr_dims = np.array(dims_selected + [dim])

            ########################################### TODO: extract marginals
            gmm1, gmm2 = full_model.gmm1, full_model.gmm2 # gmm1 is those with labels 1, and gmm2 is those with labels 0. Confusing, I know.
            partial_gmm1 = get_marginal_parameters(gmm1, curr_dims, n_components)
            partial_gmm2 = get_marginal_parameters(gmm2, curr_dims, n_components)
            
            partial_model = GaussianMixtureDiscriminantAnalysis(partial_gmm1, partial_gmm2, cov_type, n_components, full_model.classes_)
            ########################################### TODO: extract marginals

            X_test = validation_frame[list(map_list(str, curr_dims))] 
            log_prob_both_classes = partial_model.predict_log_proba(X_test)
            log_prob_for_true_class = log_prob_both_classes[np.arange(len(y_test)), label_inds]

            # TODO: may want to use logsumexp below.
            curr_loglik = log_prob_for_true_class.sum()
            if curr_loglik > best_loglik:
                best_loglik = curr_loglik
                best_dim = dim
        dims_selected.append(best_dim)
        # print(f"Current dimensions selected: {dims_selected}")
    best_likelihoods.append(best_loglik)
    return dims_selected, best_likelihoods

def plot_gmm_contours(nc_to_dims, train_frame, validation_frame, cov_type):
    """Plot GMM contours for the different number of components.

    Args:
        nc_to_dims ({int: [int]}): Number of components to dimensions that were intrinsically selected.
    """
    i = 0
    for nc, dims in nc_to_dims.items():
        fig, axes = plt.subplots(1, 1)
        plot_gmm_contours_with_nc(train_frame, validation_frame, dims[0:2], nc, cov_type, axes)
        i+=1
    
def evaluate_gmm_probing_dims(train_frame, validation_frame, n_components_to_dims):
    for nc, dims in n_components_to_dims.items():
        evaluate_probing_dims(train_frame, validation_frame, dims, GaussianMixtureDiscriminantAnalysis, f'gmm_results_{nc}')

def main(args):
    if args.evaluate_gmm_probing_dims:
        train_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_train.csv')
        validation_frame = filter_validation_frame(pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv'))
        # train_frame, validation_frame = get_train_validation_frames()
        # train_frame = normalize_embeds(train_frame)
        # validation_frame = normalize_embeds(validation_frame)
        # intrins_dims_dlda =  [479, 385, 495, 257, 254, 250, 407, 239, 269, 265]
        # intrins_dims_qda = [479, 495, 464, 332, 380, 272, 239, 129, 172, 206]
        # intrins_dims_gmm = [219, 487, 380, 119, 397, 146, 378, 143, 472, 355]
        n_components_to_dims = {1: ([351, 487, 219, 174, 116, 439, 400, 499, 466, 61]), 2: ([219, 487, 380, 119, 397, 146, 378, 143, 472, 355]), 3: ([219, 487, 380, 119, 397, 146, 378, 472, 355, 143]), 4: ([219, 487, 397, 119, 380, 146, 355, 472, 231, 143]), 5: ([487, 380, 119, 397, 146, 219, 472, 355, 143, 231])}
        evaluate_gmm_probing_dims(train_frame, validation_frame, n_components_to_dims)
    elif args.obtain_probing_dimensions_gmm:
        train_frame, validation_frame = get_train_validation_frames()
        n_components_to_dims = {}
        for num_components in range(1,6):
            dims_selected, _= obtain_probing_dimensions_gmm(train_frame, validation_frame, n_components=num_components)
            n_components_to_dims[num_components] = dims_selected
        print(n_components_to_dims)
    elif args.plot_gmm_contours:
        n_components_to_dims = {1: ([351, 487, 219, 174, 116, 439, 400, 499, 466, 61]), 2: ([219, 487, 380, 119, 397, 146, 378, 143, 472, 355]), 3: ([219, 487, 380, 119, 397, 146, 378, 472, 355, 143]), 4: ([219, 487, 397, 119, 380, 146, 355, 472, 231, 143]), 5: ([487, 380, 119, 397, 146, 219, 472, 355, 143, 231])}
        train_frame, validation_frame = get_train_validation_frames(False)
        plot_gmm_contours(n_components_to_dims, train_frame, validation_frame, 'diag')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate_gmm_probing_dims', action='store_true')
    parser.add_argument('--obtain_probing_dimensions_gmm', action='store_true')
    parser.add_argument('--plot_gmm_contours', action='store_true')
    main(parser.parse_args())