import os
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# from FeatureExtractionMode.utils.utils_plot import plot_2d, plot_3d, plot_clustering_2d, plot_ap, plot_fs, plot_hc
from FeatureExtractionMode.utils.utils_write import fa_vectors2files
from MachineLearningAlgorithm.utils.utils_read import files2vectors_info, seq_label_read


def fa_process(args, feature_vectors, labels, after_ps=False, ind=False):
    # normalization
    if args.sn != 'none':
        feature_vectors = normalization(feature_vectors, args.sn)

    # clustering
    if after_ps is True:
        if args.cl != 'none':
            assert args.nc is not None and args.nc <= feature_vectors.shape[0] and \
                   args.nc <= feature_vectors.shape[1]
            if args.nc is not None or args.cl == 'AP':
                cluster = clustering(feature_vectors, args.cm, args.nc, args.cl, args.results_dir, ind)
                save_cluster_result(cluster, args.results_dir, ind)
                # plot_clustering_2d(feature_vectors, cluster, args.results_dir, args.cm, ind)

    # feature select
    if args.fs != 'none':
        assert args.nf is not None and args.nf <= feature_vectors.shape[1]
        fs_vectors, scores = feature_select(feature_vectors, labels, args.nf, args.fs)
        if after_ps is True:
            save_fs_result(scores, args.fs, args.results_dir, ind)
            # # plot_fs(scores, args.nf, out_path, ind)
            # plot_fs(scores, args.results_dir, ind)  # 修改为仅仅绘制前20重要的特征
    else:
        fs_vectors = feature_vectors

    # dimension reduction
    if args.dr != 'none':
        assert args.np is not None and args.np <= feature_vectors.shape[1]
        dr_vectors = dimension_reduction(feature_vectors, args.np, args.dr)
        if after_ps is True:
            save_dr_result(dr_vectors, args.results_dir, ind)
            # plot_2d(dr_vectors, labels, args.results_dir, ind)
            # plot_3d(dr_vectors, labels, args.results_dir, ind)
    else:
        dr_vectors = feature_vectors

    if args.rdb == 'fs':
        assert args.fs != 'none', "Can't reduce dimension by feature select since feature select method is none"
        return fs_vectors
    elif args.rdb == 'dr':
        assert args.dr != 'none', "Can't reduce dimension by dimension reduce since dimension reduce method is none"
        return dr_vectors
    else:
        # 仅仅展示特征分析结果，而不对特征向量进行降维
        return feature_vectors


def feature_select(vectors, labels, n_features, scoring_func):
    res = np.zeros((vectors.shape[0], n_features))
    scores = np.zeros(vectors.shape[0])
    if scoring_func == 'chi2':
        selector = SelectKBest(chi2, k=n_features)
        selector.fit(vectors, labels)
        res = selector.transform(vectors)
        scores = selector.pvalues_
    elif scoring_func == 'F-value':
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(vectors, labels)
        res = selector.transform(vectors)
        scores = selector.pvalues_
    elif scoring_func == 'MIC':
        # print(n_features)
        selector = SelectKBest(mutual_info_classif, k=n_features)
        selector.fit(vectors, labels)
        res = selector.transform(vectors)
        scores = selector.pvalues_
        # print(scores)  # why none?
    elif scoring_func == 'RFE':
        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=n_features, step=1).fit(vectors, labels)
        res = rfe.transform(vectors)
        scores = rfe.ranking_
    elif scoring_func == 'Tree':
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(vectors, labels)
        scores = clf.feature_importances_
        model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=n_features)
        res = model.transform(vectors)
    return res, scores


def normalization(vectors, normal_method):
    if normal_method == 'min-max-scale':
        min_max_scale = preprocessing.MinMaxScaler()
        res = min_max_scale.fit_transform(vectors)
    elif normal_method == 'standard-scale':
        standard_scale = preprocessing.StandardScaler()
        res = standard_scale.fit_transform(vectors)
    elif normal_method == 'L1-normalize':
        res = preprocessing.normalize(vectors, norm='l1')
    else:
        res = preprocessing.normalize(vectors, norm='l2')
    return res


def clustering(vectors, mode, n_clusters, cluster_method, out_path, ind=False):
    index = []
    if mode == 'feature':
        vectors = vectors.T
        for i in range(len(vectors)):
            index.append('F%d' % (i + 1))
    else:
        for i in range(len(vectors)):
            index.append('S%d' % (i + 1))

    labels = np.zeros(len(vectors))
    if cluster_method == 'AP':
        ap = AffinityPropagation().fit(vectors)
        cluster_centers_indices = ap.cluster_centers_indices_
        labels = ap.labels_
        # plot_ap(vectors, cluster_centers_indices, labels, out_path, ind)
    elif cluster_method == 'DBSCAN':
        data = StandardScaler().fit_transform(vectors)
        db = DBSCAN().fit(data)
        labels = db.labels_
    elif cluster_method == 'GMM':
        gm = GaussianMixture(n_components=n_clusters).fit(vectors)
        labels = gm.predict(vectors)
    elif cluster_method == 'AGNES':
        # plot_hc(vectors, index, out_path, ind)
        connectivity = kneighbors_graph(vectors, n_neighbors=10, include_self=False)
        ward = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity,
                                       linkage='ward').fit(vectors)
        labels = ward.labels_
    elif cluster_method == 'Kmeans':
        labels = KMeans(n_clusters=n_clusters).fit_predict(vectors)

    res = []
    for i in range(len(vectors)):
        res.append([index[i], labels[i]])
    return res


def dimension_reduction(data, n_components, dr_method):

    new_data = np.zeros((data.shape[0], n_components))
    if dr_method == 'PCA':
        new_data = PCA(n_components=n_components, whiten=True).fit_transform(data)
    elif dr_method == 'KernelPCA':
        new_data = KernelPCA(n_components=n_components, kernel="rbf").fit_transform(data)
    elif dr_method == 'TSVD':
        new_data = TruncatedSVD(n_components).fit_transform(data)
    return new_data


def save_cluster_result(cluster, out_path, ind=False):
    if ind is True:
        filename = out_path + 'cluster_results_ind.txt'
    else:
        filename = out_path + 'cluster_results.txt'
    if cluster is None:
        return False
    else:
        my_cluster = np.array(cluster)
        df = pd.DataFrame({'name': my_cluster[:, 0], 'cluster': my_cluster[:, 1]})
        my_set = set(df.cluster.tolist())
        with open(filename, 'w') as f:
            f.write('# The sample/feature can be clustered into %d clusters:\n' % len(my_set))
            f.write('Feature\tcluster\n')
            for i in cluster:
                f.write(i[0] + '\t' + str(i[1]) + '\n')
    full_path = os.path.abspath(filename)
    if os.path.isfile(full_path):
        print('The output clustering file can be found:')
        print(full_path)
        print('\n')


def save_fs_result(scores, method, out_path, ind=False):
    if ind is True:
        filename = out_path + 'feature_selection_results_ind.txt'
    else:
        filename = out_path + 'feature_selection_results.txt'
    if scores is not None:
        index = []
        for i in range(len(scores)):
            index.append('F%d' % (i + 1))
        ranking = np.argsort(-scores)
        with open(filename, 'w') as f:
            f.write('# Feature selection method: %s\n' % method)
            for i in range(len(ranking)):
                f.write(index[ranking[i]] + '\t' + str(scores[ranking[i]]) + '\n')
    full_path = os.path.abspath(filename)
    if os.path.isfile(full_path):
        print('The output feature selection file can be found:')
        print(full_path)
        print('\n')


def save_dr_result(reduced_data, out_path, ind=False):
    if ind is True:
        filename = out_path + 'dimension_reduction_results_ind.txt'
    else:
        filename = out_path + 'dimension_reduction_results.txt'
    if reduced_data is not None:
        index = []
        for i in range(len(reduced_data)):
            index.append('F%d' % (i + 1))
    else:
        return False
    with open(filename, 'w') as f:
        f.write('Sample')
        for i in range(1, len(reduced_data[0]) + 1):
            f.write('\tPC' + str(i))
        f.write('\n')
        for i in range(len(reduced_data)):
            f.write('S%d' % i)
            for j in range(len(reduced_data[0])):
                f.write('\t' + str(reduced_data[i][j]))
            f.write('\n')
    full_path = os.path.abspath(filename)
    if os.path.isfile(full_path):
        print('The output dimension reduction file can be found:')
        print(full_path)
        print('\n')


def main(args):
    print("\nStep into analysis...\n")
    start_time = time.time()
    # 读取向量，样本数量和绝对路径
    args.results_dir = os.path.dirname(os.path.abspath(args.vec_file[0])) + '/'

    vectors, sample_num_list, in_files = files2vectors_info(args.vec_file, args.format)
    labels = seq_label_read(sample_num_list, args.label)
    fa_vectors = fa_process(args, vectors, labels, after_ps=True)
    fa_vectors2files(fa_vectors, sample_num_list, args.format, in_files)

    print("Done.")
    print(("Used time: %.2fs" % (time.time() - start_time)))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='BioSeq-NLP', description="Step into analysis, please select parameters ")

    # ----------------------- parameters for feature analysis---------------------- #
    # standardization or normalization
    parse.add_argument('-sn', choices=['min-max-scale', 'standard-scale', 'L1-normalize', 'L2-normalize', 'none'],
                       default='none', help=" Choose method of standardization or normalization for feature vectors.")
    # clustering
    parse.add_argument('-cl', choices=['AP', 'DBSCAN', 'GMM', 'AGNES', 'Kmeans', 'none'], default='none',
                       help="Choose method for clustering.")
    parse.add_argument('-cm', default='sample', choices=['feature', 'sample'], help="The mode for clustering.")
    parse.add_argument('-nc', type=int, help="The number of clusters.")
    # feature select
    parse.add_argument('-fs', choices=['chi2', 'F-value', 'MIC', 'RFE', 'Tree', 'none'], default='none',
                       help="Select feature select method.")
    parse.add_argument('-nf', type=int, help="The number of features after feature selection.")
    # dimension reduction
    parse.add_argument('-dr', choices=['PCA', 'KernelPCA', 'TSVD', 'none'], default='none',
                       help="Choose method for dimension reduction.")
    parse.add_argument('-np', type=int, help="The dimension of main component after dimension reduction.")
    # rdb
    parse.add_argument('-rdb', choices=['no', 'fs', 'dr'], default='no',
                       help="Reduce dimension by:\n"
                            " 'no'---none;\n"
                            " 'fs'---apply feature selection to parameter selection procedure;\n"
                            " 'dr'---apply dimension reduction to parameter selection procedure.\n")

    parse.add_argument('-vec_file', nargs='*', required=True, help="The input vector file or files")
    parse.add_argument('-label', type=int, nargs='*', required=True,
                       help="The corresponding label of input vector file or files")

    parse.add_argument('-format', default='csv', choices=['tab', 'svm', 'csv', 'tsv'],
                       help="The output format (default = csv).\n"
                            "tab -- Simple format, delimited by TAB.\n"
                            "svm -- The libSVM training data format.\n"
                            "csv, tsv -- The format that can be loaded into a spreadsheet program.")
    argv = parse.parse_args()
    main(argv)
