

def plot_contours(train_frame, validation_frame, dlda_dims, qda_dims):
    plot_dlda_ellipses(train_frame, validation_frame, dlda_dims)
    plot_qda_ellipses(train_frame, validation_frame, qda_dims)

def plot_accs_and_lls(accs, lls, figname):
    fig, (axes) = plt.subplots(2,1, sharex=True)
    axes[0].plot(np.arange(len(lls)), lls, label='log-likelihood', marker='o')
    axes[1].plot(np.arange(len(accs)), accs, label='accuracy', marker='o')
    plt.xticks(np.arange(len(lls)), np.arange(len(lls)) + 1)

    axes[1].set_xlabel("Number of dimensions")
    axes[0].set_ylabel("Log-likelihood")
    axes[1].set_ylabel("Accuracy")

    plt.tight_layout()
    store_pic_dynamic(plt, figname, 'results')