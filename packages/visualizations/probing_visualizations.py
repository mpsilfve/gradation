import seaborn as sns

from packages.utils.utils_iter import map_list
from packages.pkl_operations.pkl_io import *
from ..constants.constants import *
from ..probes.diagonal_lda import DiagonalLDA
from ..probes.qda import QDA
from ..visualizations.visualize_gaussian import *

def plot_dlda_ellipses(train_frame, validation_frame, dimensions):
    assert len(dimensions) == 2
    cs = sns.color_palette('husl', 2)
    fig, ax = plt.subplots(1,1)
    model = DiagonalLDA()
    dimensions = map_list(str, dimensions)
    model.fit(train_frame[dimensions], train_frame[LABEL_COLUMN])
    m1 = model.centroid_one
    m2 = model.centroid_two
    cov = np.diag(model.pev)
    ellipse_m1 = draw_gaussian_ellipse(m1, cov, cs[0] )
    ellipse_m2 = draw_gaussian_ellipse(m2, cov, cs[1])
    draw_ellipse(ellipse_m1, ax)
    draw_ellipse(ellipse_m2, ax)
    sns.scatterplot(data=validation_frame, x=dimensions[0], y=dimensions[1], hue=LABEL_COLUMN, hue_order=['yes', 'no'],  ax=ax, palette=cs)
    store_pic_dynamic(plt, 'dlda_ellipses', 'results')

def plot_qda_ellipses(train_frame, validation_frame, dimensions):
    assert len(dimensions) == 2
    cs = sns.color_palette('husl', 2)
    fig, ax = plt.subplots(1,1)
    model = QDA()
    dimensions = map_list(str, dimensions)
    model.fit(train_frame[dimensions], train_frame[LABEL_COLUMN])
    m1 = model.post_means[0]
    m2 = model.post_means[1]
    cov1 = model.post_covs[0]
    cov2 = model.post_covs[1]
    ellipse_m1 = draw_gaussian_ellipse(m1, cov1, cs[0] )
    ellipse_m2 = draw_gaussian_ellipse(m2, cov2, cs[1])
    draw_ellipse(ellipse_m1, ax)
    draw_ellipse(ellipse_m2, ax)
    sns.scatterplot(data=validation_frame, x=dimensions[0], y=dimensions[1], hue=LABEL_COLUMN, hue_order=['yes', 'no'],  ax=ax, palette=cs)
    store_pic_dynamic(plt, 'qda_ellipses', 'results')

def plot_dimensions(gradation_frame, dims=[487,484]):
    fig, ax = plt.subplots(1,1, figsize=[6.4, 7.8])
    dims = map_list(str, dims)
    gradation_frame = gradation_frame[gradation_frame[LABEL_COLUMN].isin(set(['yes', 'no']))]
    # print(embed_2d_frame)
    # markers = {'k': '$k$', 't': '$t$', 'p': '$p$', '-': '.', '_': '.'}
    # markers = {'k': 'k', 't': 't', 'p': '$p$', '-': '.', '_': '.'}
    sizes = {'no': 5, 'yes': 30}
    sns.scatterplot(data=gradation_frame, x=dims[0], y=dims[1], hue=DIRECTION_LABEL, style=CONSONANT_LABEL, legend='brief', size=LABEL_COLUMN, sizes=sizes)
    ax.get_legend().remove()

    legend = fig.legend(loc='upper center', ncol=3, bbox_to_anchor = (.5, ax.get_ylim()[1] + .18))
        # fig.legend(loc='upper center', ncol=6)
    # plt.tight_layout(bbox)
    store_pic_dynamic(plt, 'model_3_fig_1', 'results')
    # 487 and 484