from commons.utils import *
from PIL import Image
from sklearn.manifold import TSNE
from matplotlib.pyplot import cm
import matplotlib.mlab as mlab
from matplotlib.ticker import NullFormatter

# latent_vecs: FC之前的二维特征(b, feats)
# final_label: 标签, 'Sarcopenia'->1->蓝
def t_sne(latent_vecs, final_label, config, fold, epoch, label_names=['Asymptomatic', 'Sarcopenia'], num_classes=2):
    # fname = "t-SNE/" + str(fold) + str(epoch) + str(time.time()) + '.pdf'
    fname = "t-SNE" + str(fold) + str(epoch) + '.pdf'
    # fig = plt.figure()
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    plt_props()
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
        # ax.scatter(xx[final_label == i], yy[final_label == i], color=colors[i], label=label_names[i], s=50)
        ax.scatter(xx[final_label == i], yy[final_label == i], color=colors[i], s=30)

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.set_yticks([])
    plt.axis('tight')
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('off')
    # plt.legend(loc='best', scatterpoints=1, fontsize=16)
    plt.savefig(config.tmp_dir + '/' + "t-SNE_" + str(fold) + "_" + str(epoch) + '.png', dpi=100)
    # plt.savefig(config.tmp_dir + '/' + fname, format='pdf', dpi=600)
    # plt.show()
    plt.close(fig)

from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#使用Yooden法寻找最佳阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

#计算roc值
def ROC(label, y_prob):
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    roc_auc = auc(fpr, tpr)
    optimal_threshold, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_threshold, optimal_point

# 计算混淆矩阵
def calculate_metric(label, y_prob, optimal_threshold):
    p = []
    for i in y_prob:
        if i >= optimal_threshold:
            p.append(1)
        else:
            p.append(0)
    confusion = confusion_matrix(label, p)
    # print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Accuracy = (TP + TN) / float(TP + TN + FP + FN)
    Sensitivity = TP / float(TP + FN)
    Specificity = TN / float(TN + FP)
    return Accuracy, Sensitivity, Specificity

# AUC-ROC curve
# results = []
# roc_ = []
def auc_roc(probs, labels, config, model_name, fold, epoch):
    results = []
    roc_ = []
    probs = probs[:, 1]
    fpr, tpr, roc_auc, Optimal_threshold, optimal_point = ROC(labels, probs)
    Accuracy, Sensitivity, Specificity = calculate_metric(labels, probs, Optimal_threshold)
    result = [Optimal_threshold, Accuracy, Sensitivity, Specificity, roc_auc, model_name]
    results.append(result)
    roc_.append([fpr, tpr, roc_auc, model_name])

    # 绘制多组对比roc曲线
    color = ["darkorange", "navy", "red", "green", "yellow", "pink"]
    fig = plt.figure()
    # plt.figure(figsize=(10, 10))
    plt_props()
    lw = 2
    plt.plot(roc_[0][0], roc_[0][1], color=color[0], lw=lw, label=roc_[0][3] + ' (AUC = %0.3f)' % roc_[0][2])
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR (False Positive Rate)')
    plt.ylabel('TPR (True Positive Rate)')
    plt.title('AUC - ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(config.tmp_dir+'/'+'AUC-ROC curve_'+str(fold)+"_"+str(epoch)+".png", dpi=80)
    # plt.savefig(config.tmp_dir+'/'+'AUC-ROC curve'+str(fold)+str(epoch)+'.pdf', format='pdf', dpi=600)
    # plt.show()
    # plt.close()
    plt.close(fig)

def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img

def plt_props():
    plt.rcParams['font.size'] = 16
    # plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.style'] = 'normal'
    plt.rcParams['font.variant'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['figure.figsize'] = 8, 6
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 5

# def plot_sns_line(data_frames, y_aix, y_label, title, data_set, dashes):
#     sns.set(font='serif')
#     sns.set_style("whitegrid")
#     final_sns = pd.concat(data_frames, axis=0)
#     fig, ax = plt.subplots(dpi=500, figsize=(8, 6))
#     plt_props()
#     ax = sns.lineplot(x="per", y="data", hue="Method", style='Method',
#                       markers=False, dashes=dashes,
#                       data=final_sns)
#     # ax.lines[2].set_linestyle("--")
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
#     y_major_locator = MultipleLocator(5)
#     ax.yaxis.set_major_locator(y_major_locator)
#     ax.set_xlim(9.8, 50.4)
#     ax.set_ylim(y_aix[0], y_aix[1])
#     x_major_locator = MultipleLocator(10)
#     ax.xaxis.set_major_locator(x_major_locator)
#     ax.grid(visible=False)
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles=handles[1:], labels=labels[1:], ncol=2, loc='lower right')
#     # leg = ax.legend(handles=handles[1:], labels=labels[1:], ncol=2, loc='lower right')
#     # leg_lines = leg.get_lines()
#     # leg_lines[2].set_linestyle("--")
#     fig.tight_layout()
#     plt.xlabel('% of Labeled Data')
#     plt.ylabel(y_label)
#     plt.title(title + ' performance')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(join('../log/', title + "_line_seg_" + data_set + ".jpg"))
#     plt.close()

# def plot_error_bar(res_stat, y_aix, y_label, title, m, data_set, nn=3):
#     sns.set(font='serif')
#     sns.set_style("whitegrid")
#     if nn == 3:
#         isic_ratios = [100. * item for item in [0.05, 0.10, 0.15]]
#     if nn == 6:
#         isic_ratios = [100. * item for item in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
#     if nn == 9:
#         isic_ratios = [100. * item for item in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]]
#     fig, ax = plt.subplots(dpi=500, figsize=(8, 6))
#     plt_props()
#     for key in m.keys():
#         print(key)
#         if key == 'Full':
#             ax.errorbar(isic_ratios, np.max(np.array(res_stat[key]) * 100, axis=0), label=key, marker=m[key],
#                         yerr=0, capsize=2, linestyle='--')
#             # ax.errorbar(isic_ratios, np.mean(np.array(res_stat[key]) * 100, axis=0), label=key, marker=m[key],
#             #             yerr=np.std(np.array(res_stat[key]) * 100, axis=0), capsize=2, linestyle='--')
#         else:
#             ax.errorbar(isic_ratios, np.mean(np.array(res_stat[key]) * 100, axis=0), label=key, marker=m[key],
#                         yerr=np.std(np.array(res_stat[key]) * 100, axis=0), capsize=2)
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
#     y_major_locator = MultipleLocator(5)
#     ax.yaxis.set_major_locator(y_major_locator)
#     ax.set_xlim(isic_ratios[0]-0.5, isic_ratios[-1]+0.5)
#     ax.set_ylim(y_aix[0], y_aix[1])
#     x_major_locator = MultipleLocator(10)
#     ax.xaxis.set_major_locator(x_major_locator)
#     ax.legend(ncol=2, loc='lower right')
#     ax.grid(visible=False)
#     plt.xlabel('% of Labeled Data')
#     plt.ylabel(y_label)
#     plt.title(title + ' performance')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(join('../log/', title + "_error_seg_" + data_set + ".jpg"))
#     plt.close()