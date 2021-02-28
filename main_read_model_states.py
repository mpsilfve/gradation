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
from sklearn.preprocessing import normalize

from packages.probes.qda import QDA
from packages.gradation_utils.utils import *
from packages.gradation_utils.visualizations import *
from packages.visualizations.probing_visualizations import *
from packages.visualizations.visualize_gaussian import *
from packages.constants.constants import *
from packages.utils.utils_iter import map_list
from packages.probes.diagonal_lda import DiagonalLDA
from packages.pkl_operations.pkl_io import *
from packages.utils.utils_save import *

def obtain_probing_dimensions_dlda(train_frame, validation_frame, num_dims_to_select=10):
    dims_selected = []
    best_likelihoods = []
    full_model = DiagonalLDA()
    full_model.fit(train_frame[map_list(str, ENCODER_DIMS)], train_frame[LABEL_COLUMN])
    y_test = validation_frame[LABEL_COLUMN]
    labels = y_test.map(lambda x: 0 if x == full_model.classes_[0] else 1)
    label_inds = labels.map(lambda label: 0 if label == 1 else 1)
    # y_test = y_test.map(lambda y: 1 if y == 0 else 1)
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
            log_prob_for_true_class = log_prob_both_classes[np.arange(len(label_inds)), label_inds]
            curr_loglik = log_prob_for_true_class.sum()
            if curr_loglik > best_loglik:
                best_loglik = curr_loglik
                best_dim = dim
        dims_selected.append(best_dim)
        print(f"Dimensions selected so far: {dims_selected}")
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
            means, covs = full_model.post_means, full_model.post_covs
            partial_means = map_list(lambda mean: mean[curr_dims], means)
            partial_covs = map_list(lambda cov: cov[np.ix_(curr_dims, curr_dims)], covs)
            partial_model = QDA(partial_means, partial_covs, full_model.classes_)

            X_test = validation_frame[list(map_list(str, curr_dims))] 
            log_prob_both_classes = partial_model.predict_log_proba(X_test)
            log_prob_for_true_class = log_prob_both_classes[np.arange(len(y_test)), label_inds]
            curr_loglik = log_prob_for_true_class.sum()
            if curr_loglik > best_loglik:
                best_loglik = curr_loglik
                best_dim = dim
        dims_selected.append(best_dim)
        print(f"Current dimensions selected: {dims_selected}")
    best_likelihoods.append(best_loglik)
    return dims_selected, best_likelihoods

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
        labels = y_test.map(lambda x: 0 if x == partial_model.classes_[0] else 1)
        label_inds = labels.map(lambda label: 0 if label == 1 else 1)
        X_test = validation_frame[curr_dims] 
        # y_test_binarized = y_test.map(lambda x: 0 if x == 1 else 1)
        log_prob_both_classes = partial_model.predict_log_proba(X_test)
        log_prob_for_true_class = log_prob_both_classes[np.arange(len(label_inds)), label_inds]
        curr_loglik = log_prob_for_true_class.mean()
        predictions = partial_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        accs.append(acc)
        print(acc)
        lls.append(curr_loglik)
        if (i == 0) or (i == len(intrinsic_dims)-1):
            print(confusion_matrix(y_test, predictions))
    plot_accs_and_lls(accs, lls, figname)

def filter_validation_frame(frame):
    return frame[frame[LABEL_COLUMN].isin(set(['yes', 'no']))]

def calculate_activation_differences(dims, validation_frame):
    gradation_frame = validation_frame[validation_frame[LABEL_COLUMN]=='yes']
    non_gradation_frame = validation_frame[validation_frame[LABEL_COLUMN]=='no']
    print(non_gradation_frame)
    print(gradation_frame)
    print(f'Format::Dimension:mean difference;(standard deviation for gradation examples, standard deviation for non-gradation examples)')
    for dim in dims:
        dim = str(dim)
        gradation_activations = gradation_frame[dim]
        non_gradation_activations = non_gradation_frame[dim]
        g_mean = gradation_activations.mean()
        ng_mean = non_gradation_activations.mean()
        g_std = gradation_activations.std()
        ng_std = non_gradation_activations.std()
        print(f"{dim}:{g_mean-ng_mean:.3f};({g_std:.3f},{ng_std:.3f})")

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
        train_frame = normalize_embeds(train_frame)
        validation_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv')
        validation_frame = normalize_embeds(validation_frame)
        best_dims, best_likelihoods = obtain_probing_dimensions_dlda(train_frame, validation_frame)
        print(f"Best dimensions: {best_dims}")
        print(f"Log likelihoods for corresponding dimensions: {best_likelihoods}")
    elif args.obtain_probing_dimensions_qda:
        train_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_train.csv')
        train_frame = normalize_embeds(train_frame)
        validation_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv')
        validation_frame = normalize_embeds(validation_frame)
        best_dims, best_likelihoods = obtain_probing_dimensions_qda(train_frame, validation_frame)
    elif args.evaluate_probing_dims:
        train_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_train.csv')
        validation_frame = filter_validation_frame(pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv'))
        # intrins_dims_dlda =  [479, 385, 495, 257, 254, 250, 407, 239, 269, 265]
        # intrins_dims_dlda_normed = [12, 132, 473, 450, 351, 383, 121, 285, 126, 459]   
        # paper_dims_dlda =  [56, 270, 284, 367, 306]
        # paper_dims_qda=  [56, 270, 284, 367, 306]
        # intrins_dims_qda = [479, 495, 464, 332, 380, 272, 239, 129, 172, 206]
        intrins_dims_qda_normed = [351, 487, 397, 219, 174, 102, 348, 439, 284, 228]  
        # evaluate_probing_dims(train_frame, validation_frame, intrins_dims_dlda_normed, DiagonalLDA, 'dlda_results')
        # evaluate_probing_dims(train_frame, validation_frame, intrins_dims_dlda, DiagonalLDA, 'dlda_results')
        evaluate_probing_dims(train_frame, validation_frame, intrins_dims_qda_normed, QDA, 'qda_results')
        # evaluate_probing_dims(train_frame, validation_frame, intrins_dims_qda_normed, QDA, 'qda_results')
        # evaluate_probing_dims(train_frame, validation_frame, paper_dims_qda, DiagonalLDA, 'dlda_paper_results')
    elif args.replicate_fig_1:
        train_frame = pd.read_csv('results/2021-02-11/gradation_w_encoder_state_train.csv')
        validation_frame = pd.read_csv('results/2021-02-11/gradation_w_encoder_state_validation.csv')
        gradation_frame = pd.concat([train_frame, validation_frame])
        plot_dimensions(gradation_frame)
    elif args.plot_contours:
        train_frame = pd.read_csv('results/2021-01-19/gradation_w_encoder_state_train.csv')
        validation_frame = filter_validation_frame(pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv'))
        dlda_dims = [12, 132]
        qda_dims = [351, 284]
        paper_dims = [56, 367]
        # plot_contours(train_frame, validation_frame, dlda_dims, qda_dims)
        plot_contours(train_frame, validation_frame, dlda_dims, qda_dims)
        # plot_contours(train_frame, validation_frame, paper_dims, paper_dims)
    elif args.calculate_activation_differences:
        validation_frame = filter_validation_frame(pd.read_csv('results/2021-01-19/gradation_w_encoder_state_validation.csv'))
        intrins_dims_dlda =  [479, 385, 495, 257, 254, 250, 407, 239, 269, 265]
        intrins_dims_qda = [479, 495, 464, 332, 380, 272, 239, 129, 172, 206]
        paper_dims = [56, 270, 284, 367, 306]
        # intrins_dims_dlda =  [56]
        # calculate_activation_differences(intrins_dims_dlda, validation_frame)
        # calculate_activation_differences(intrins_dims_qda, validation_frame)
        calculate_activation_differences(paper_dims, validation_frame)

if __name__ == '__main__':
    # state_dict= load_pkl_from_path('./representations')
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_validation_frame', action='store_true')
    parser.add_argument('--build_train_frame', action='store_true')
    parser.add_argument('--calculate_activation_differences', action='store_true')
    parser.add_argument('--obtain_probing_dimensions_dlda', action='store_true')
    parser.add_argument('--obtain_probing_dimensions_qda', action='store_true')
    parser.add_argument('--print_data_stats', action='store_true')
    parser.add_argument('--evaluate_probing_dims', action='store_true')
    parser.add_argument('--plot_contours', action='store_true')
    parser.add_argument('--replicate_fig_1', action='store_true')
    main(parser.parse_args())