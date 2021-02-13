import sys
from itertools import combinations
import csv
import pandas as pd
import argparse
import numpy as np
import pickle
import seaborn as sns 
import matplotlib.pyplot as plt
from os.path import split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
# from torch import load

from packages.probes.qda import QDA
from packages.visualizations.probing_visualizations import plot_dimensions
from packages.visualizations.visualize_gaussian import *
from packages.constants.constants import *
from packages.utils.utils_iter import map_list
from packages.probes.diagonal_lda import DiagonalLDA
from packages.pkl_operations.pkl_io import *
from packages.utils.utils_save import *

def obtain_penult_embed(rep_tensor):
    pen_index = len(rep_tensor) - 2
    pen_representation = rep_tensor[pen_index][0]
    assert len(pen_representation) == 500
    return pen_representation.numpy()

def build_validation_frame(validation_representations, validation_csv_path):
    validation_frame = pd.read_csv(validation_csv_path)
    validation_frame[ENCODER_DIMS] = list(map(obtain_penult_embed, validation_representations))
    store_csv_dynamic(validation_frame, 'gradation_w_encoder_state_validation')
    # TODO: add the dimensions to it.

def build_train_frame(train_representations, train_csv_path):
    train_frame = pd.read_csv(train_csv_path)
    train_frame[ENCODER_DIMS] = list(map(obtain_penult_embed, train_representations))
    store_csv_dynamic(train_frame, 'gradation_w_encoder_state_train')

# TODO: need to verify that the marginalization property holds true for diagonal LDA
def obtain_probing_dimensions_dlda(train_frame, validation_frame, num_dims_to_select=30):
    dims_selected = []
    best_likelihoods = []
    full_model = DiagonalLDA()
    full_model.fit(train_frame[map_list(str, ENCODER_DIMS)], train_frame[LABEL_COLUMN])
    y_test = validation_frame[LABEL_COLUMN]
    # y_test = y_test.map(lambda x: 0 if x == full_model.classes_[0] else 1)
    y_test = y_test.map(lambda y: 1 if y == 0 else 1)
    for _ in range(num_dims_to_select):
        best_loglik = -float('inf')
        best_dim = -1
        set_dims_selected = set(dims_selected)
        for dim in range(len(ENCODER_DIMS)):
            if dim in set_dims_selected:
                continue
            curr_dims = np.array(dims_selected + [dim])
            m1, m2, pev = full_model.centroid_one, full_model.centroid_two, full_model.pev
            partial_m1 = m1[curr_dims]
            partial_m2 = m2[curr_dims]
            partial_pev = pev[curr_dims]
            partial_model = DiagonalLDA(partial_m1, partial_m2, partial_pev, full_model.classes_, full_model.class_priors)

            X_test = validation_frame[list(map_list(str, curr_dims))] 
            
            log_prob_both_classes = partial_model.predict_log_proba(X_test)
            log_prob_for_true_class = log_prob_both_classes[np.arange(len(y_test)), y_test]
            curr_loglik = log_prob_for_true_class.sum()
            if curr_loglik > best_loglik:
                best_loglik = curr_loglik
                best_dim = dim
        dims_selected.append(best_dim)
        best_likelihoods.append(best_loglik)
    return dims_selected, best_likelihoods

def obtain_probing_dimensions_qda(train_frame, validation_frame, num_dims_to_select=10):
    dims_selected = []
    best_likelihoods = []
    full_model = QDA()
    all_dims = map_list(str, ENCODER_DIMS)
    full_model.fit(train_frame[all_dims], train_frame[LABEL_COLUMN])
    y_test = validation_frame[LABEL_COLUMN]
    y_test = y_test.map(lambda x: 0 if x == full_model.classes_[0] else 1)
    for _ in range(num_dims_to_select):
        best_loglik = -float('inf')
        best_dim = -1
        set_dims_selected = set(dims_selected)
        for dim in range(len(ENCODER_DIMS)):
            if dim in set_dims_selected:
                continue
            curr_dims = np.array(dims_selected + [dim])
            means, covs = full_model.post_means, full_model.post_covs
            partial_means = map_list(lambda mean: mean[curr_dims], means)
            partial_covs = map_list(lambda cov: cov[np.ix_(curr_dims, curr_dims)], covs)
            partial_model = QDA(partial_means, partial_covs, full_model.classes_)

            X_test = validation_frame[list(map_list(str, curr_dims))] 
            log_prob_both_classes = partial_model.predict_log_proba(X_test)
            log_prob_for_true_class = log_prob_both_classes[np.arange(len(y_test)), y_test]
            curr_loglik = log_prob_for_true_class.sum()
            if curr_loglik > best_loglik:
                best_loglik = curr_loglik
                best_dim = dim
        dims_selected.append(best_dim)
        print(dims_selected)
    best_likelihoods.append(best_loglik)
    return dims_selected, best_likelihoods

def print_data_stats(train_frame, validation_frame):
    num_train_gradation = len(train_frame[train_frame[LABEL_COLUMN]=='yes'])
    num_train_no_gradation = len(train_frame[train_frame[LABEL_COLUMN]=='no'])

    num_test_gradation = len(validation_frame[validation_frame[LABEL_COLUMN]=='yes'])
    num_test_no_gradation = len(validation_frame[validation_frame[LABEL_COLUMN]=='no'])
    print(f"Number of train examples with gradation: {num_train_gradation}")
    print(f"Number of train examples without gradation: {num_train_no_gradation}")
    print(f"Proportion of train examples with gradation: {num_train_gradation/(num_train_no_gradation + num_train_gradation):.3f}")

    print(f"Number of test examples with gradation: {num_test_gradation}")
    print(f"Number of test examples without gradation: {num_test_no_gradation}")
    print(f"Proportion of test examples with gradation: {num_test_gradation/(num_test_gradation + num_test_no_gradation):.3f}")


def plot_accs_and_lls(accs, lls, figname):
    fig, (axes) = plt.subplots(2,1, sharex=True)
    axes[0].plot(np.arange(len(lls)), lls, label='log-likelihood', marker='o')
    axes[1].plot(np.arange(len(accs)), accs, label='accuracy', marker='o')

    axes[1].set_xlabel("Number of dimensions")
    axes[0].set_ylabel("Log-likelihood")
    axes[1].set_ylabel("Accuracy")

    plt.tight_layout()
    store_pic_dynamic(plt, figname, 'results')

def evaluate_probing_dims(train_frame, validation_frame, intrinsic_dims, model, figname):
    intrinsic_dims = map_list(str, intrinsic_dims)
    # full_model.fit(train_frame[intrinsic_dims], train_frame[LABEL_COLUMN])
    y_test = validation_frame[LABEL_COLUMN]

    accs = []
    lls = []
    for i in range(1, len(intrinsic_dims)):
        partial_model = model()
        curr_dims = intrinsic_dims[0:i+1] 
        partial_model.fit(train_frame[curr_dims], train_frame[LABEL_COLUMN])
        X_test = validation_frame[curr_dims] 
        y_test_binarized = y_test.map(lambda x: 0 if x == 1 else 1)
        log_prob_both_classes = partial_model.predict_log_proba(X_test)
        log_prob_for_true_class = log_prob_both_classes[np.arange(len(y_test_binarized)), y_test_binarized]
        curr_loglik = log_prob_for_true_class.sum()
        predictions = partial_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        accs.append(acc)
        print(acc)
        lls.append(curr_loglik)
        if (i == 1) or (i == len(intrinsic_dims)-1):
            print(confusion_matrix(y_test, predictions))
    plot_accs_and_lls(accs, lls, figname)

def filter_validation_frame(frame):
    return frame[frame[LABEL_COLUMN].isin(set(['yes', 'no']))]

def main(args):
    if args.build_validation_frame:
        # validation_representations = load_pkl_from_path('models/gradation_10_step_3000.pt.treebank-nouns.tsv.nom2gen.valid.src.enc_states')
        validation_representations = load_pkl_from_path('models/gradation_3_step_3000.pt.treebank-nouns.tsv.nom2gen.valid.src.enc_states')
        # validation_csv_path = 'LSTM_hidden_states/data/treebank-nouns.tsv.nom2gen.valid.annotated.csv'
        build_validation_frame(validation_representations, VALIDATION_CSV_PATH)
    elif args.build_train_frame:
        train_representations = load_pkl_from_path('train_representations/gradation_3_step_3000.pt.treebank-nouns.tsv.nom2gen.train.src.enc_states')
        build_train_frame(train_representations, TRAIN_CSV_PATH)
    elif args.print_data_stats:
        train_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_train.csv')
        validation_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv')
        print_data_stats(train_frame, validation_frame)
    elif args.obtain_probing_dimensions_dlda:
        train_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_train.csv')
        validation_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv')
        best_dims, best_likelihoods = obtain_probing_dimensions_dlda(train_frame, validation_frame)
        print(f"Best dimensions: {best_dims}")
        print(f"Log likelihoods for corresponding dimensions: {best_likelihoods}")
    elif args.obtain_probing_dimensions_qda:
        train_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_train.csv')
        validation_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv')
        best_dims, best_likelihoods = obtain_probing_dimensions_qda(train_frame, validation_frame)
    elif args.evaluate_probing_dims:
        train_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_train.csv')
        validation_frame = filter_validation_frame(pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv'))
        intrins_dims_dlda =[385, 479, 495, 257, 407, 250, 239, 363, 473, 386]
        intrins_dims_qda = [495, 464, 479, 332, 239, 380, 272, 206, 129, 176]
        # evaluate_probing_dims(train_frame, validation_frame, intrins_dims_dlda, DiagonalLDA, 'dlda_results')
        evaluate_probing_dims(train_frame, validation_frame, intrins_dims_qda, QDA, 'qda_results')
    elif args.replicate_fig_1:
        train_frame = pd.read_csv('results/2021-02-11/gradation_w_encoder_state_train.csv')
        validation_frame = pd.read_csv('results/2021-02-11/gradation_w_encoder_state_validation.csv')
        gradation_frame = pd.concat([train_frame, validation_frame])
        print(gradation_frame)
        plot_dimensions(gradation_frame)
        # print()


if __name__ == '__main__':
    # state_dict= load_pkl_from_path('./representations')
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_validation_frame', action='store_true')
    parser.add_argument('--build_train_frame', action='store_true')
    parser.add_argument('--obtain_probing_dimensions_dlda', action='store_true')
    parser.add_argument('--obtain_probing_dimensions_qda', action='store_true')
    parser.add_argument('--print_data_stats', action='store_true')
    parser.add_argument('--evaluate_probing_dims', action='store_true')
    parser.add_argument('--replicate_fig_1', action='store_true')
    main(parser.parse_args())