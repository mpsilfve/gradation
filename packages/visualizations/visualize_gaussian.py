import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.linalg as LA
import numpy as np

def draw_gaussian_ellipse(mean, cov, color):
    """

    Args:
        mean ([type]): [description]
        cov ([type]): [description]
    """
    s = 5.991 # 90% confidence interval band
    # https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix
    # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
    eigvals, eigvecs = LA.eig(cov) # TODO: these aren't sorted
    prin_eigval_i, min_eigval_i = np.argmax(eigvals), np.argmin(eigvals)
    principal_axis, minor_axis = eigvecs[prin_eigval_i], eigvecs[min_eigval_i]
    prin_length = 2 * np.sqrt(s) * np.sqrt(eigvals[prin_eigval_i]) 
    min_length = 2 * np.sqrt(s) * np.sqrt(eigvals[min_eigval_i]) 
    # angle = np.arctan2(principal_axis[1], principal_axis[0]) 
    angle = 180 * np.arctan2(principal_axis[1], principal_axis[0]) / np.pi
    # TODO: something isn't right about the angle addition
    return Ellipse((mean[0], mean[1]), prin_length, min_length, angle + 90, alpha=0.5, color=color, fill=False) 

# def plot_points(frame, ax):
#     sns.scatterplot(data=frame, x=0, y=1, hue='cluster_num', ax=ax)
#     return ax

# def plot_mixture_weights(arr, ax):
#     sns.barplot(x=np.arange(len(arr)), y=arr)
#     return ax 

def draw_ellipse(ellipse, ax):
    ax.add_patch(ellipse)
    return ax