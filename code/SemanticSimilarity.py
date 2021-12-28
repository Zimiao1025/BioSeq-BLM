import os
from itertools import combinations_with_replacement, product
import threading

import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from FeatureExtractionMode.utils.utils_write import FormatWrite
from MachineLearningAlgorithm.utils.utils_read import files2vectors_seq

SEED = 42


# 欧氏距离
def euclidean_distance(vec1, vec2):
    # ord=2: 二范数
    score = np.linalg.norm(vec1-vec2, ord=2)
    return round(score, 4)


# 曼哈顿距离
def manhattan_distance(vec1, vec2):
    # ord=1: 一范数
    score = np.linalg.norm(vec1 - vec2, ord=1)
    return round(score, 4)


# 切比雪夫距离
def chebyshev_distance(vec1, vec2):
    # ord=np.inf: 无穷范数
    score = np.linalg.norm(vec1-vec2, ord=np.inf)
    return round(score, 4)


# 汉明距离
def hamming_distance(vec1, vec2):
    # 适用于二进制编码格式 !
    return len(np.nonzero(vec1-vec2)[0])  # 返回整数


# 杰卡德相似度
def jaccard_similarity_coefficient(vec1, vec2):
    # 适用于二进制编码格式 !
    score = distance.pdist(np.array([vec1, vec2]), "jaccard")[0]
    return round(score, 4)


# 余弦相似度
def cosine_similarity(vec1, vec2):
    score = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*(np.linalg.norm(vec2)))
    return round(score, 4)


# 皮尔森相关系数
def pearson_correlation_coefficient(vec1, vec2):
    score = pearsonr(vec1, vec2)[0]
    return round(score, 4)


# 相对熵又称交叉熵, Kullback-Leible散度（即KL散度）等,
# 这个指标不能用作距离衡量，因为该指标不具有对称性, 为了在并行计算时统一，采用对称KL散度
def kl_divergence(vec1, vec2):
    score = (entropy(vec1, vec2) + entropy(vec2, vec1)) / 2.0
    return round(score, 4)


def score_func_one(method, vectors1, vectors2, index1, index2, return_dict, sem):
    sem.acquire()
    if method == 'ED':
        score = euclidean_distance(vectors1[index1], vectors2[index2])
    elif method == 'MD':
        score = manhattan_distance(vectors1[index1], vectors2[index2])
    elif method == 'CD':
        score = chebyshev_distance(vectors1[index1], vectors2[index2])
    elif method == 'HD':
        score = hamming_distance(vectors1[index1], vectors2[index2])
    elif method == 'JSC':
        score = jaccard_similarity_coefficient(vectors1[index1], vectors2[index2])
    elif method == 'CS':
        score = cosine_similarity(vectors1[index1], vectors2[index2])
    elif method == 'PCC':
        score = pearson_correlation_coefficient(vectors1[index1], vectors2[index2])
    elif method == 'KLD':
        score = kl_divergence(vectors1[index1], vectors2[index2])
    else:
        print('Semantic Similarity method error!')
        return False
    return_dict[(index1, index2)] = score
    sem.release()


def score4train_vec(method, train_vectors, process_num):
    threads = []
    sem = threading.Semaphore(process_num)
    return_dict = {}
    row = train_vectors.shape[0]

    for i, j in combinations_with_replacement(list(range(len(train_vectors))), 2):
        threads.append(threading.Thread(target=score_func_one,
                                        args=(method, train_vectors, train_vectors, i, j, return_dict, sem)))
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    score_mat = np.zeros([row, row])
    for i in range(row):
        for j in range(row):
            if i <= j:
                score_mat[i][j] = return_dict[(i, j)]
            else:
                score_mat[i][j] = return_dict[(j, i)]
    return score_mat


def score4test_vec(method, train_vectors, test_vectors, process_num):
    threads = []
    sem = threading.Semaphore(process_num)

    return_dict = {}

    train_row = train_vectors.shape[0]
    test_row = test_vectors.shape[0]

    for i, j in product(list(range(test_row)), list(range(train_row))):
        threads.append(threading.Thread(target=score_func_one,
                                        args=(method, test_vectors, train_vectors, i, j, return_dict, sem)))
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    score_mat = np.zeros([test_row, train_row])
    for i in range(test_row):
        for j in range(train_row):
            score_mat[i][j] = return_dict[(i, j)]
    return score_mat


def partition_vectors(vectors, folds_num):
    fold = KFold(folds_num, shuffle=True, random_state=np.random.RandomState(SEED))
    folds_temp = list(fold.split(vectors))

    folds = []
    for i in range(folds_num):
        test_index = folds_temp[i][1]
        train_index = folds_temp[i][0]

        folds.append((train_index, test_index))
    return folds


def get_partition(vectors, labels, train_index, val_index):
    x_train = vectors[train_index]
    x_val = vectors[val_index]
    y_train = labels[train_index]
    y_val = labels[val_index]

    return x_train, y_train, x_val, y_val


def score_process(method, vec_files, labels, cv, out_format, process_num):
    vectors = files2vectors_seq(vec_files, out_format)
    dir_name, _ = os.path.splitext(vec_files[0])
    score_dir = dir_name + '/score/'

    if cv == 'j':
        folds_num = len(vectors)
    elif cv == '10':
        folds_num = 10
    else:
        folds_num = 5

    folds = partition_vectors(vectors, folds_num)
    count = 0
    for train_index, test_index in folds:
        x_train, y_train, x_test, y_test = get_partition(vectors, labels, train_index, test_index)
        train_mat = score4train_vec(method, x_train, process_num)
        test_mat = score4test_vec(method, x_train, x_test, process_num)
        count += 1
        temp_dir = score_dir + 'Fold%d/' % count
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        FormatWrite(train_mat, out_format, temp_dir + 'train_score.txt').write_to_file()
        FormatWrite(test_mat, out_format, temp_dir + 'test_score.txt').write_to_file()

        np.savetxt(temp_dir + 'train_label.txt', y_train)
        np.savetxt(temp_dir + 'test_label.txt', y_test)
        np.savetxt(temp_dir + 'test_index.txt', test_index)


def ind_score_process(method, vectors, ind_vec_file, labels, ind_labels, out_format, process_num):
    ind_vectors = files2vectors_seq(ind_vec_file, out_format)
    dir_name, _ = os.path.split(ind_vec_file[0])
    score_dir = dir_name + '/ind_score/'
    train_mat = score4train_vec(method, vectors, process_num)
    test_mat = score4test_vec(method, vectors, ind_vectors, process_num)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    FormatWrite(train_mat, out_format, score_dir + 'train_score.txt').write_to_file()
    FormatWrite(test_mat, out_format, score_dir + 'test_score.txt').write_to_file()

    np.savetxt(score_dir + 'train_label.txt', labels)
    np.savetxt(score_dir + 'test_label.txt', ind_labels)
