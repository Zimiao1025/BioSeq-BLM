import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_roc_curve(cv_labels, cv_prob, file_path):
    """Plot ROC curve."""
    # Receiver Operating Characteristic
    tpr_list = []
    auc_list = []
    fpr_array = []
    tpr_array = []
    thresholds_array = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(cv_labels)):
        fpr, tpr, thresholds = roc_curve(cv_labels[i], cv_prob[i])
        fpr_array.append(fpr)
        tpr_array.append(tpr)
        thresholds_array.append(thresholds)
        tpr_list.append(interp(mean_fpr, fpr, tpr))
        tpr_list[-1][0] = 0.0
        try:
            roc_auc = auc(fpr, tpr)
        except ZeroDivisionError:
            roc_auc = 0.0
        auc_list.append(roc_auc)

    plt.figure(0)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.7)
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)
    plt.plot(mean_fpr, mean_tpr, color='navy',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.7)
    std_tpr = np.std(tpr_list, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=.3,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.title('Receiver Operating Characteristic', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    figure_name = file_path + 'cv_roc.png'
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The Receiver Operating Characteristic of cross-validation can be found:')
        print(full_path)
        print('\n')
    return mean_auc


def plot_roc_ind(ind_labels, ind_prob, file_path):
    fpr_ind, tpr_ind, thresholds_ind = roc_curve(ind_labels, ind_prob)
    try:
        ind_auc = auc(fpr_ind, tpr_ind)
    except ZeroDivisionError:
        ind_auc = 0.0
    plt.figure(0)
    plt.plot(fpr_ind, tpr_ind, lw=2, alpha=0.7, color='red',
             label='ROC curve (area = %0.2f)' % ind_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    figure_name = file_path + 'ind_roc.png'
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The Receiver Operating Characteristic of independent test can be found:')
        print(full_path)
        print('\n')
    return ind_auc


def plot_pr_curve(cv_labels, cv_prob, file_path):
    precisions = []
    auc_list = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)
    for i in range(len(cv_labels)):
        precision, recall, _ = precision_recall_curve(cv_labels[i], cv_prob[i])
        recall_array.append(recall)
        precision_array.append(precision)
        precisions.append(interp(mean_recall, recall[::-1], precision[::-1])[::-1])
        try:
            roc_auc = auc(recall, precision)
        except ZeroDivisionError:
            roc_auc = 0.0
        auc_list.append(roc_auc)

    plt.figure(0)
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(auc_list)
    plt.plot(mean_recall, mean_precision, color='navy',
             label=r'Mean PRC (AUPRC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.7)
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.3,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.title('Precision-Recall Curve', fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend(loc="lower left")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    figure_name = file_path + 'cv_prc.png'
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The Precision-Recall Curve of cross-validation can be found:')
        print(full_path)
        print('\n')
    return mean_auc


def plot_pr_ind(ind_labels, ind_prob, file_path):
    precision, recall, _ = precision_recall_curve(ind_labels, ind_prob)
    try:
        ind_auc = auc(recall, precision)
    except ZeroDivisionError:
        ind_auc = 0.0
    plt.figure(0)
    plt.plot(recall, precision, lw=2, alpha=0.7, color='red',
             label='PRC curve (area = %0.2f)' % ind_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve', fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend(loc="lower left")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    figure_name = file_path + 'ind_prc.png'
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The Precision-Recall Curve of independent test can be found:')
        print(full_path)
        print('\n')
    return ind_auc
