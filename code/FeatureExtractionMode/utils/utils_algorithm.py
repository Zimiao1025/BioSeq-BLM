import pickle
import math
from collections import Counter
from math import log
from random import shuffle

import networkx as nx
import numpy as np
from gensim.models import Word2Vec, FastText
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold


# TODO: TF-IDF
def tf_idf(sentence_list):
    corpus = []
    for sentence in sentence_list:
        document = ' '.join(sentence)
        corpus.append(document)
    count_vec = CountVectorizer()
    # 计算个词语出现的次数
    X = count_vec.fit_transform(corpus)
    # 获取词袋中所有文本关键词
    # word = count_vec.get_feature_names()
    # print(word)
    # 类调用
    transformer = TfidfTransformer()
    # print(transformer)
    # 将词频矩阵X统计成TF-IDF值
    tf_idf_vec = transformer.fit_transform(X)
    # 查看数据结构 tf-idf[i][j]表示i类文本中的tf-idf权重
    # print(tf_idf_vec.toarray())
    return tf_idf_vec.toarray()


# TODO: TextRank
def text_rank(sentence_list, alpha=0.85):
    corpus = []
    for sentence in sentence_list:
        document = ' '.join(sentence)
        corpus.append(document)
    count_vec = CountVectorizer()
    # 计算个词语出现的次数
    X = count_vec.fit_transform(corpus)
    # 类调用
    transformer = TfidfTransformer()
    # print(transformer)
    # 将词频矩阵X统计成TF-IDF值
    tf_idf_vec = transformer.fit_transform(X)
    similarity = nx.from_scipy_sparse_matrix(tf_idf_vec * tf_idf_vec.T)

    scores = nx.pagerank(similarity, alpha=alpha)

    vectors = []
    tf_idf_vec = tf_idf_vec.toarray()
    scores_val = list(scores.values())
    for i in range(len(scores_val)):
        vectors.append(tf_idf_vec[i] * scores_val[i])
    return np.array(vectors)


# TODO: TextRank
def text_rank1(sentence_list, alpha=0.85, window_size=3):
    # window_size用来统计词的共现关系以构建无向图
    max_iteration = 30  # max iteration for power iteration method
    d = alpha  # damping factor
    threshold = 0.001  # convergence threshold THRESHOLD
    vectors = []
    for ind in range(len(sentence_list)):
        document = sentence_list[ind]

        vocabulary = list(set(document))
        vector = np.zeros(len(document), dtype=np.float32)
        vocab_len = len(vocabulary)

        weighted_edge = np.zeros((vocab_len, vocab_len), dtype=np.float32)

        score = np.zeros(vocab_len, dtype=np.float32)
        covered_co_occurrence = []
        for i in range(0, vocab_len):
            score[i] = 1
            for j in range(0, vocab_len):
                if j == i:
                    weighted_edge[i][j] = 0
                else:
                    for window_start in range(0, (len(document) - window_size)):

                        window_end = window_start + window_size

                        window = document[window_start:window_end]

                        if (vocabulary[i] in window) and (vocabulary[j] in window):

                            index_of_i = window_start + window.index(vocabulary[i])
                            index_of_j = window_start + window.index(vocabulary[j])

                            # index_of_x is the absolute position of the xth term in the window
                            # (counting from 0)
                            # in the processed_text

                            if [index_of_i, index_of_j] not in covered_co_occurrence:
                                weighted_edge[i][j] += 1 / math.fabs(index_of_i - index_of_j)
                                covered_co_occurrence.append([index_of_i, index_of_j])

        inout = np.zeros(vocab_len, dtype=np.float32)

        for i in range(0, vocab_len):
            for j in range(0, vocab_len):
                inout[i] += weighted_edge[i][j]
        print('Power iteration method running...')
        print('Sequence index: %d' % ind)
        for it in range(0, max_iteration):
            prev_score = np.copy(score)

            for i in range(0, vocab_len):

                summation = 0
                for j in range(0, vocab_len):
                    if weighted_edge[i][j] != 0:
                        summation += (weighted_edge[i][j] / inout[j]) * score[j]

                score[i] = (1 - d) + d * summation

            if np.sum(np.fabs(prev_score - score)) <= threshold:  # convergence condition
                print("Converging at iteration " + str(it) + "....\n")
                break

        # for i in range(0, vocab_len):
        #     print("Score of " + vocabulary[i] + ": " + str(score[i]))

        for i in range(len(document)):
            index = vocabulary.index(document[i])
            rank_value = score[index]
            vector[i] = rank_value
        vectors.append(vector)
    return np.array(vectors)


# TODO: Word2vec
def data_partition(sample_size_list):
    num_sum = 0
    seed_ = 42
    folds_num = 5
    label_all = []
    for i in range(len(sample_size_list)):
        tmp_labels = [float(i)] * sample_size_list[i]
        label_all += tmp_labels
        num_sum += sample_size_list[i]
    label_all = np.array(label_all)
    pse_data = np.random.normal(loc=0.0, scale=1.0, size=[num_sum, num_sum])
    folds = StratifiedKFold(folds_num, shuffle=True, random_state=np.random.RandomState(seed_))
    folds_temp = list(folds.split(pse_data, label_all))
    folds = []
    for i in range(folds_num):
        train_index = folds_temp[i][0]
        test_index = folds_temp[i][1]
        folds.append((train_index, test_index))
    return folds


def word2vec(sentence_list, sample_size_list, fixed_len, word_size, win_size, vec_dim=10, skip_gram=0):
    n_row = (fixed_len - word_size + 1) * vec_dim  # the default win_size value is 100
    corpus_out = -np.ones((len(sentence_list), n_row))

    folds = data_partition(sample_size_list)
    print('word2vec processing ...')
    for i, (train_index, test_index) in enumerate(folds):
        print('Round [%s]' % (i + 1))
        train_sentences = []
        test_sentences = []
        for x in train_index:
            train_sentences.append(sentence_list[x])
        for y in test_index:
            test_sentences.append(sentence_list[y])
        # The core stone of Gene2vec  |  window: 一个句子中当前单词和被预测单词的最大距离。
        model = Word2Vec(train_sentences, size=vec_dim, window=win_size, sg=skip_gram)  # sg=1对应skip gram模型
        vectors = []
        for sentence in test_sentences:
            # print(sentence)
            vector = []
            for j in range(len(sentence)):
                try:
                    vec_temp = np.array(model[sentence[j]])

                except KeyError:
                    vec_temp = np.zeros(vec_dim)

                # print(len(vec_temp))
                if len(vector) == 0:
                    vector = vec_temp
                else:
                    vector = np.hstack((vector, vec_temp))
            vectors.append(vector)
        corpus_out[test_index] = np.array(vectors)
    return corpus_out


# TODO: Glove
def build_vocab(train_corpus):
    """
    Build a vocabulary with word frequencies for an entire corpus.
    Returns: {word : (ID, frequency)}
    """

    vocab = Counter()
    for line in train_corpus:
        for tokens in line:
            vocab.update([tokens])

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}


def build_co_occur(vocab, corpus, window_size=10, min_count=None):
    """
    Build a word co-occurrence list for the given corpus.
    return: (i_main, i_context, co_occurrence value)
    i_main -> the main word in the co_occurrence
    i_context -> is the ID of the context word
    co_occurrence` is the `X_{ij}` co_occurrence value as described in Pennington et al.(2014).
    If `min_count` is not `None`, co_occurrence pairs fewer than `min_count` times are ignored.
    """

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    co_occurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                       dtype=np.float64)

    for i, line in enumerate(corpus):

        token_ids = [vocab[word][0] for word in line]

        for center_i, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            # 将窗口左边的内容与窗口右边的内容区分开，
            context_ids = token_ids[max(0, center_i - window_size): center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)  # 原文用词对间的独立来除count

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                co_occurrences[center_id, left_id] += increment
                co_occurrences[left_id, center_id] += increment

    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)
    for i, (row, data) in enumerate(zip(co_occurrences.rows,
                                        co_occurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]


def run_iter(data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training.

    `data` is a pre-fetched data / weights list where each element is of
    the form

        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)

    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:
        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - log(cooccurrence))

        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost  # 注意这里乘了1/2

        # Compute gradients for word vector terms.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word, biases, etc.
        grad_main = weight * cost_inner * v_context  # 损失函数对vi求导，是不是缺了个因子2呢？ 不缺！
        grad_context = weight * cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))  # 梯度下降法，
        # 问题是为什么要除以np.sqrt(gradsq_W_main) -> 原文使用Adagrad算法， 利用梯度的对学习率约束
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
            gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)  # 向量
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2  # 标量
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def train_glove(vocab, co_occurrences, iter_callback=None, vector_size=100,
                iterations=25, **kwargs):
    """
    co_occurrences: (word_i_id, word_j_id, x_ij)

    If `iter_callback` is not `None`, the provided function will be
    called after each iteration with the learned `W` matrix so far.

    Keyword arguments are passed on to the iteration step function
    `run_iter`.

    Returns the computed word vector matrix .
    """

    vocab_size = len(vocab)

    # Word vector matrix. This matrix is (2V) * d, where V is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5].
    vector_matrix = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    #
    #  this matrix is same size with vector_matrix
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # NB: These are all views into the actual data matrices, so updates
    # to them will pass on to the real data structures
    data = [(vector_matrix[i_main], vector_matrix[i_context + vocab_size],
             biases[i_main: i_main + 1],
             biases[i_context + vocab_size: i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main: i_main + 1],
             gradient_squared_biases[i_context + vocab_size: i_context + vocab_size + 1],
             co_occurrence)
            for i_main, i_context, co_occurrence in co_occurrences]

    for i in range(iterations):

        cost = run_iter(data, **kwargs)
        print('global cost of glove model: %.4f' % cost)
        if iter_callback is not None:
            iter_callback(vector_matrix)

    return vector_matrix


def save_model(vector_matrix, out_path):
    with open(out_path, 'wb') as vector_f:
        pickle.dump(vector_matrix, vector_f, protocol=2)


def load_model(model_path):
    with open(model_path, 'wb') as vector_f:
        vector_matrix = pickle.load(vector_f)
    return vector_matrix


def merge_main_context(vector_matrix, merge_fun=lambda m, c: np.mean([m, c], axis=0),
                       normalize=True):
    """
    Merge the main-word and context-word vectors for a weight matrix
    using the provided merge function (which accepts a main-word and
    context-word vector and returns a merged version).

    By default, `merge_fun` returns the mean of the two vectors.
    """

    vocab_size = int(len(vector_matrix) / 2)
    for i, row in enumerate(vector_matrix[:vocab_size]):
        merged = merge_fun(row, vector_matrix[i + vocab_size])  # 按对应行进行求和
        if normalize:
            merged /= np.linalg.norm(merged)
        vector_matrix[i, :] = merged

    return vector_matrix[:vocab_size]


def glove(sentence_list, sample_size_list, fixed_len, word_size, win_size, vec_dim=10):
    n_row = (fixed_len - word_size + 1) * vec_dim
    corpus_out = -np.ones((len(sentence_list), n_row))

    folds = data_partition(sample_size_list)
    print('Glove processing ...')

    for i, (train_index, test_index) in enumerate(folds):
        print('Round [%s]' % (i + 1))
        train_sentences = []
        test_sentences = []
        for x in train_index:
            train_sentences.append(sentence_list[x])
        for y in test_index:
            test_sentences.append(sentence_list[y])
        # The core stone of Glove
        vocab = build_vocab(train_sentences)  # 词汇表
        # print(vocab): {'CTT': (0, 8), 'TTC': (1, 6), 'TCG': (2, 2), 'CGC': (3, 2), ...}
        # exit()
        co_occur = build_co_occur(vocab, train_sentences, window_size=win_size)  # “共现矩阵”

        vector_matrix = train_glove(vocab, co_occur, vector_size=vec_dim, iterations=50)  # 词向量矩阵(main + context)

        # Merge and normalize word vectors
        vector_matrix = merge_main_context(vector_matrix)  # 对词向量矩阵(main + context)按对应行平均并归一化
        vectors = []
        for sentence in test_sentences:
            vector = []
            for j in range(len(sentence)):
                try:
                    vec_temp = np.array(vector_matrix[vocab[sentence[j]][0]])
                    # vocab={'word': (id, frequency), ...}
                except KeyError:
                    vec_temp = np.zeros(vec_dim)
                if len(vector) == 0:
                    vector = vec_temp
                else:
                    vector = np.hstack((vector, vec_temp))
            vectors.append(vector)
        corpus_out[test_index] = np.array(vectors)
    print('....................')
    return corpus_out


# TODO: fastText
def fast_text(sentence_list, sample_size_list, fixed_len, word_size, win_size, vec_dim=10, skip_gram=0):
    n_row = (fixed_len - word_size + 1) * vec_dim  # the default win_size value is 100
    corpus_out = -np.ones((len(sentence_list), n_row))

    folds = data_partition(sample_size_list)
    print('fastText processing ...')
    for i, (train_index, test_index) in enumerate(folds):
        print('Round [%s]' % (i + 1))
        train_sentences = []
        test_sentences = []
        for x in train_index:
            train_sentences.append(sentence_list[x])
        for y in test_index:
            test_sentences.append(sentence_list[y])
        # The core stone of FastText
        model = FastText(sentence_list, size=vec_dim, window=win_size, sg=skip_gram)
        vectors = []
        for sentence in test_sentences:
            vector = []
            for j in range(len(sentence)):
                try:
                    vec_temp = np.array(model[sentence[j]])
                except KeyError:
                    vec_temp = np.zeros(vec_dim)
                if len(vector) == 0:
                    vector = vec_temp
                else:
                    vector = np.hstack((vector, vec_temp))
            vectors.append(vector)
        corpus_out[test_index] = np.array(vectors)
    print('.......................')
    return corpus_out
