import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time
import numpy as np

# latent_vecs: FC之前的二维特征(b, feats)
# final_label: 标签
def t_sne(latent_vecs, final_label, config, label_names=['Benign', 'Malignant'],
          num_classes=2):
    fname = "tsne_" + str(time.time()) + '.pdf'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, num_classes))
    embeddings = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    xx = embeddings[:, 0]
    yy = embeddings[:, 1]

    # plot the images
    # if False:
    #     for i, (x, y) in enumerate(zip(xx, yy)):
    #         im = OffsetImage(X_sample[i], zoom=0.1, cmap='gray')
    #         ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    #         ax.add_artist(ab)
    #     ax.update_datalim(np.column_stack([xx, yy]))
    #     ax.autoscale()

    # plot the 2D data points
    for i in range(num_classes):
        ax.scatter(xx[final_label == i], yy[final_label == i], color=colors[i], label=label_names[i], s=10)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.set_yticks([])
    plt.axis('tight')
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    plt.savefig(config.tmp_dir + '/' + fname, format='pdf', dpi=600)
    # plt.show()

