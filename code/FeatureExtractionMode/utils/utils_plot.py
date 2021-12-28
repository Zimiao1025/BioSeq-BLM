import os
import random
from itertools import cycle
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d(data, labels, path, ind=False):
    if ind is True:
        figure_name = path + 'dimension_reduction_2d_ind.png'
    else:
        figure_name = path + 'dimension_reduction_2d.png'
    if os.path.isfile(figure_name):
        try:
            os.remove(figure_name)
        except OSError:
            print('File delete error!')
            pass
    color_sets = cycle(['crimson', 'navy', 'teal', 'darkorange', 'slategrey'])
    color_set = []
    label_set = list(set(labels))
    for i, j in zip(label_set, color_sets):
        color_set.append(j)
    my_dict = {}
    for i in range(len(label_set)):
        my_dict[label_set[i]] = color_set[i]
    plt.figure(0)
    if len(labels) == 0:
        plt.scatter(data[:, 0], data[:, 1], 20, c='r')
    else:
        df = pd.DataFrame({'X': data[:, 0], 'Y': data[:, 1], 'L': labels})
        labels_set = set(labels)
        for k in labels_set:
            new_data = df.loc[df.loc[:, "L"] == k, :]
            plt.scatter(np.array(new_data.X), np.array(new_data.Y), s=50, c=my_dict[k], alpha=0.7, label="Label_%s" % k)
            plt.legend(loc='best')
    plt.title('2D-figure of dimension reduction', fontsize=18)
    plt.xlabel('First principal component', fontsize=14)
    plt.ylabel('Second principal component', fontsize=14)
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The output 2D-figure for dimension reduction can be found:')
        print(full_path)
        print('\n')


def plot_3d(data, labels, path, ind=False):
    if ind is True:
        figure_name = path + 'dimension_reduction_3d_ind.png'
    else:
        figure_name = path + 'dimension_reduction_3d.png'
    if os.path.isfile(figure_name):
        try:
            os.remove(figure_name)
        except OSError:
            print('File delete error!')
            pass
    mark_sets = cycle(['o', 'o'])
    color_sets = cycle(['crimson', 'navy', 'teal', 'darkorange', 'slategrey'])
    label_set = list(set(labels))
    my_dict = {}
    m = 0
    for i in label_set:
        my_dict[i] = m
        m = m + 1
    mark_set = []
    color_set = []
    for i, j, k in zip(label_set, mark_sets, color_sets):
        mark_set.append(j)
        color_set.append(k)
    mc = np.zeros((len(labels), 2)).astype(str)
    for i in range(len(labels)):
        mc[i][0], mc[i][1] = mark_set[my_dict[labels[i]]], color_set[my_dict[labels[i]]]
    fig = plt.figure(0)
    axes3d = Axes3D(fig)

    for i in range(len(data)):
        axes3d.scatter(data[i][0], data[i][1], data[i][2], s=40, c=mc[i][1], alpha=0.7)

    plt.title('3D-figure of dimension reduction', fontsize=18)
    plt.xlabel('First PC', fontsize=12)
    plt.ylabel('Second PC', fontsize=12)
    axes3d.set_zlabel('Third PC', fontsize=12)

    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The output 3D-figure for dimension reduction can be found:')
        print(full_path)
        print('\n')


def plot_clustering_2d(data, my_cluster, path, mode, ind=False):
    if ind is True:
        figure_name = path + 'clustering_2d_ind.png'
    else:
        figure_name = path + 'clustering_2d.png'
    if os.path.isfile(figure_name):
        try:
            os.remove(figure_name)
        except OSError:
            print('File delete error!')
            pass
    if my_cluster is not None:
        if mode == 'sample':
            data = data
        else:
            data = data.T
        labels = np.array(my_cluster)[0:, 1:].reshape(-1, )
        clusters = list(map(float, labels))
        clustering = np.array(clusters)
        try:
            # print(data)
            y_2d = TSNE(n_components=2, init='pca', random_state=42).fit_transform(data)
            # y_2d = TSNE(n_components=2).fit_transform(data)
        except RuntimeWarning:
            y_2d = PCA(n_components=2).fit_transform(data)
        color_sets = cycle(['crimson', 'navy', 'teal', 'darkorange', 'slategrey', 'purple'])
        color_set = []
        label_set = list(set(labels))
        for i, j in zip(label_set, color_sets):
            color_set.append(j)
        my_dict = {}
        for i in range(len(label_set)):
            my_dict[label_set[i]] = color_set[i]
        plt.figure(0)
        labels_set = set(labels)
        if len(labels_set) > 6:
            plt.scatter(y_2d[:, 0], y_2d[:, 1], s=50, c=clustering, alpha=0.7)  # adjusted!
        else:
            df = pd.DataFrame({'X': y_2d[:, 0], 'Y': y_2d[:, 1], 'L': labels})
            for k in labels_set:
                new_data = df.loc[df.loc[:, "L"] == k, :]
                plt.scatter(np.array(new_data.X), np.array(new_data.Y), s=50, linewidths=0,
                            c=my_dict[k], alpha=0.7, label="cluster_%s" % k)
        plt.legend(loc='best')
        plt.title('2D-figure of clustering', fontsize=18)
        ax_width = 1
        ax = plt.gca()  # 获取边框
        ax.spines['bottom'].set_linewidth(ax_width)
        ax.spines['left'].set_linewidth(ax_width)
        ax.spines['top'].set_linewidth(ax_width)
        ax.spines['right'].set_linewidth(ax_width)
        plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
        plt.close(0)
        full_path = os.path.abspath(figure_name)
        if os.path.isfile(full_path):
            print('The output 2D-figure for clustering can be found:')
            print(full_path)
            print('\n')


def plot_fs(scores, path, ind=False):
    if ind is True:
        figure_name = path + 'Feature_importance_ind.png'
    else:
        figure_name = path + 'Feature_importance.png'
    if os.path.isfile(figure_name):
        try:
            os.remove(figure_name)
        except OSError:
            print('File delete error!')
            pass
    if scores is not None:
        index = []
        for i in range(len(scores)):
            index.append('F%d' % (i + 1))
        ranking = np.argsort(-scores)[:10]
        x_label = []
        height = []
        for i in range(len(ranking)):
            x_label.append(index[ranking[i]])
            height.append(scores[ranking[i]])
        plt.figure(0)
        plt.bar(range(len(height)), height, color='navy', alpha=0.7, tick_label=x_label)
        plt.xticks(size=10)
        plt.title('Feature importance ranking', fontsize=18)
        plt.xlabel('Feature index', fontsize=16)
        plt.ylabel('Feature importance', fontsize=16)
        ax_width = 1
        ax = plt.gca()  # 获取边框
        ax.spines['bottom'].set_linewidth(ax_width)
        ax.spines['left'].set_linewidth(ax_width)
        ax.spines['top'].set_linewidth(ax_width)
        ax.spines['right'].set_linewidth(ax_width)
        plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
        plt.close(0)
        full_path = os.path.abspath(figure_name)
        if os.path.isfile(full_path):
            print('The figure for feature importance can be found:')
            print(full_path)
            print('\n')


def plot_ap(vectors, cluster_centers_indices, labels_, path, ind=False):
    n_clusters_ = len(cluster_centers_indices)
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

    if ind is True:
        figure_name = path + 'AP_cluster_ind.png'
    else:
        figure_name = path + 'AP_cluster.png'
    if os.path.isfile(figure_name):
        try:
            os.remove(figure_name)
        except OSError:
            print('File delete error!')
            pass
    try:
        data = TSNE(n_components=2, init='pca', random_state=42).fit_transform(vectors)
    except RuntimeWarning:
        data = PCA(n_components=2).fit_transform(vectors)
    for k, col in zip(range(n_clusters_), colors):
        # labels == k 使用k与labels数组中的每个值进行比较
        # 如labels = [1,0],k=0,则‘labels==k’的结果为[False, True]
        class_members = labels_ == k
        cluster_center = data[cluster_centers_indices[k]]  # 聚类中心的坐标
        plt.plot(data[class_members, 0], data[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in data[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.title('Numbers of predicted centers：%d' % n_clusters_)
    plt.title('Affinity Propagation cluster diagram', fontsize=16)
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('Here is figure for %s method\n' % figure_name)
        print(full_path)


def plot_hc(vectors, index, path, ind=False):
    # 2020/7/26 加一个随机抽样，减少图片显示样本数目
    index_sp = []
    if len(vectors) > 22:
        range_list = list(range(len(vectors)))
        random.shuffle(range_list)
        sp_list = range_list[: 22]
        sp_list.sort()
        #
        vectors = vectors[sp_list]

        for k in sp_list:
            index_sp.append(index[k])
    else:
        index_sp = index

    if ind is True:
        image = path + 'H_cluster_ind.png'
    else:
        image = path + 'H_cluster.png'
    if os.path.isfile(image):
        try:
            os.remove(image)
        except OSError:
            print('File delete error!')
            pass
    plt.figure(0)
    dis_mat = sch.distance.pdist(vectors, 'euclidean')
    z = sch.linkage(dis_mat, method='ward')
    sch.dendrogram(z, labels=np.array(index_sp), leaf_rotation=270, leaf_font_size=8)
    plt.title('Hierarchical cluster diagram', fontsize=18)
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    plt.savefig(image, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(image)
    if os.path.isfile(full_path):
        print('Here is figure for %s method\n' % image)
        print(full_path)
