import seaborn as sns
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt

from packages.visualizations.visualize_gaussian import *
from packages.probes.diagonal_lda import *
from packages.probes.qda import *
from packages.pkl_operations.pkl_io import *
from scipy.stats import multivariate_normal

np.random.seed(0)

def obtain_gt_mvns():
    corr_cov = np.array([[1, 0.8], [0.8, 1]])
    corr_dims_norm = multivariate_normal(-np.ones(2), corr_cov)

    uncorr_cov = np.eye(2)
    uncorr_dims_norm = multivariate_normal(np.ones(2), uncorr_cov)
    return corr_dims_norm, uncorr_dims_norm

def get_samples(norm_dists, size=100):
    sample_frames = []
    for i in range(len(norm_dists)):
        norm_dist = norm_dists[i]
        sample_frame = pd.DataFrame(data=norm_dist.rvs(size=size))
        sample_frame['label'] = len(sample_frame) * [i]
        sample_frames.append(sample_frame)
    complete_sample_frame = pd.concat(sample_frames)
    return complete_sample_frame

def draw_samples(sample_frame, ax, cs):
    sns.scatterplot(data=sample_frame, x=0, y=1, hue='label', ax=ax, palette=cs, hue_order=[0,1])

def test_dlda():
    corr_norm, uncorr_norm = obtain_gt_mvns()
    sample_frame = get_samples([corr_norm, uncorr_norm])
    cs = sns.color_palette('husl', 2)
    fig, ax = plt.subplots(1,1) 
    draw_samples(sample_frame, ax, cs)
    y = sample_frame['label']

    dlda = DiagonalLDA()
    dlda.fit(sample_frame[[0,1]], sample_frame['label'])
    for i in range(len(dlda.classes_)):
        if dlda.classes_[i] == 1:
            c = cs[1]
            m = dlda.centroid_one
        elif dlda.classes_[i] == 0:
            c = cs[0]
            m = dlda.centroid_two
        cov = np.diag(dlda.pev)
        ellipse = draw_gaussian_ellipse(m, cov, c)
        # ellipse_m2 = draw_gaussian_ellipse(m2, cov, cs[1])
        draw_ellipse(ellipse, ax)
        # draw_ellipse(ellipse_m2, ax)
    store_pic_dynamic(plt, 'samples_dlda', 'results')

    log_probs = dlda.predict_log_proba(sample_frame[[0,1]])
    labels = y.map(lambda x: 0 if x == dlda.classes_[0] else 1)
    label_inds = labels.map(lambda label: 0 if label == 1 else 1)

    log_prob_for_true_class = log_probs[np.arange(len(sample_frame['label'])), label_inds].mean()
    print(log_prob_for_true_class)

def test_qda():
    corr_norm, uncorr_norm = obtain_gt_mvns()
    sample_frame = get_samples([corr_norm, uncorr_norm])
    cs = sns.color_palette('husl', 2)
    fig, ax = plt.subplots(1,1) 
    draw_samples(sample_frame, ax, cs)

    qda = QDA()
    qda.fit(sample_frame[[0,1]], sample_frame['label'])
    for i in range(len(qda.classes_)):
        if qda.classes_[i] == 0:
            m = qda.post_means[1]
            cov = qda.post_covs[1]
            print(cov)
            # cov = np.array([[1, 0.8], [0.8, 1]])
            c = cs[0]
        elif qda.classes_[i] == 1:
            m = qda.post_means[0]
            cov = qda.post_covs[0]

            c = cs[1]
        ellipse = draw_gaussian_ellipse(m, cov, c)
        draw_ellipse(ellipse, ax)
    store_pic_dynamic(plt, 'samples_qda', 'results')
        
def main(args):
    if args.test_dlda:
        test_dlda()
    elif args.test_qda:
        test_qda()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dlda', action='store_true')
    parser.add_argument('--test_qda', action='store_true')
    # parser.add_argument('--', action='store_true')

    main(parser.parse_args())

