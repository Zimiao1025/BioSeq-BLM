import os
from numpy import random
import shutil
import sys
from collections import Counter
from itertools import count, takewhile, product

from MachineLearningAlgorithm.utils.utils_math import construct_partition2two

# Alphabets of DNA, RNA, PROTEIN
DNA = "ACGT"
RNA = "ACGU"
PROTEIN = "ACDEFGHIKLMNPQRSTVWY"

# 每个residue特征应该包含method/words
Method_Res = ['One-hot', 'Binary-5bit', 'One-hot-6bit', 'Position-specific-2', 'Position-specific-3',
              'Position-specific-4', 'AESNN3', 'DBE', 'NCP', 'DPC', 'TPC', 'PP', 'PSSM', 'PSFM',
              'PAM250', 'BLOSUM62', 'BLAST-matrix', 'SS', 'SASA', 'RSS', 'CS']

# 每个sequence特征应该包含method/words
Method_One_Hot_Enc = ['One-hot', 'Binary-5bit', 'One-hot-6bit', 'Position-specific-2', 'Position-specific-3',
                      'Position-specific-4', 'AESNN3', 'DBE', 'NCP', 'DPC', 'TPC', 'PP', 'PSSM', 'PSFM',
                      'PAM250', 'BLOSUM62', 'BLAST-matrix', 'SS', 'SASA', 'RSS', 'CS']

All_Words = ['Kmer', 'RevKmer', 'Mismatch', 'Subsequence', 'Top-N-Gram', 'DR', 'DT']
DNA_Words = ['Kmer', 'RevKmer', 'Mismatch', 'Subsequence']
RNA_Words = ['Kmer', 'Mismatch', 'Subsequence']
Protein_Words = ['Kmer', 'Mismatch', 'Top-N-Gram', 'DR', 'DT']

Method_Topic_Model = ['LSA', 'PLSA', 'LDA', 'Labeled-LDA']

Method_Word_Embedding = ['word2vec', 'fastText', 'Glove']

Method_Syntax_Rules = ['DAC', 'DCC', 'DACC', 'TAC', 'TCC', 'TACC', 'MAC', 'GAC', 'NMBAC', 'AC', 'CC', 'ACC', 'PDT',
                       'PDT-Profile', 'AC-PSSM', 'CC-PSSM', 'ACC-PSSM', 'PSSM-DT', 'PSSM-RT', 'ZCPseKNC', 'ND',
                       'Motif-PSSM']

Method_Auto_features = ['MotifCNN', 'MotifDCNN', 'CNN-BiLSTM', 'DCNN-BiLSTM', 'Autoencoder']

Method_Semantic_Similarity = ['ED', 'MD', 'CD', 'HD', 'JSC', 'CS', 'PCC', 'KLD', 'none']

Score_dict = {'ED': 'Euclidean Distance', 'MD': 'Manhattan Distance', 'CD': 'Chebyshev Distance',
              'HD': 'hamming Distance', 'JSC': 'Jaccard Similarity Coefficient', 'CS': 'Cosine Similarity',
              'PCC': 'Pearson Correlation Coefficient', 'KLD': 'Kullback-Leible Divergence'}

# 特征提取的种类
Feature_Extract_Mode = ['OHE', 'BOW', 'TF-IDF', 'TR', 'WE', 'TM', 'SR', 'AF']  # 'semantic similarity' ?

Mode = {'OHE': 'one-hot encoding', 'BOW': 'bag of words', 'TF-IDF': 'term frequency–inverse document frequency',
        'TR': 'TextRank', 'WE': 'word embedding', 'TM': 'topic model', 'SR': 'syntax rules', 'AF': 'automatic features'}
Machine_Learning_Algorithm = ['SVM', 'RF', 'CRF', 'CNN', 'LSTM', 'GRU', 'Transformer', 'Weighted-Transformer',
                              'Reformer']
Ml = {'SVM': 'Support Vector Machine(SVM)', 'RF': 'Random Forest(RF)', 'CRF': 'Conditional Random Field(CRF)',
      'CNN': 'Convolutional Neural Networks(CNN)', 'LSTM': 'Long Short-Term Memory(LSTM)',
      'GRU': 'Gate Recurrent Unit(GRU)', 'Transformer': 'Transformer',
      'Weighted-Transformer': 'Weighted-Transformer', 'Reformer': 'Reformer'}
DeepLearning = ['CNN', 'LSTM', 'GRU', 'Transformer', 'Weighted-Transformer', 'Reformer']
Classification = ['SVM', 'RF', 'CNN', 'LSTM', 'GRU', 'Transformer', 'Weighted-Transformer', 'Reformer']
SequenceLabelling = ['CRF', 'CNN', 'LSTM', 'GRU', 'Transformer', 'Weighted-Transformer', 'Reformer']

# 路径
Final_Path = '/results/'
Batch_Path_Seq = '/results/batch/Seq/'
Batch_Path_Res = '/results/batch/Res/'

FE_PATH_Res = '/results/FE/Res/'
FE_BATCH_PATH_Res = '/results/batch/FE/Res/'
FE_PATH_Seq = '/results/FE/Seq/'
FE_BATCH_PATH_Seq = '/results/batch/FE/Seq/'

# Metric
Metric_Index = {'Acc': 0, 'MCC': 1, 'AUC': 2, 'BAcc': 3, 'F1': 8}
Metric_dict = {'Acc': 'Accuracy', 'MCC': 'Matthews Correlation Coefficient', 'AUC': 'Area Under Curve',
               'BAcc': 'Balanced-Accuracy', 'F1': 'F-Measure'}


def seq_sys_check(args, res=False):
    print('************************** PLEASE CHECK **************************')
    if args.mode != 'OHE':
        assert args.ml not in SequenceLabelling, 'The ' + Ml[args.ml] + ' machine learning algorithm can only ' \
                                                                        'construct predictor for one-hot feature!'
    print('Analysis category: %s sequence' % args.category)
    if res is False:
        print('Feature extraction mode: ', Mode[args.mode])
    else:
        print('Feature extraction method: ', args.method)
    print('Machine learning algorithm: ', Ml[args.ml])
    print('*******************************************************************')
    print('\n')


def check_contain_chinese(check_str):
    """Check if the path name and file name user input contain Chinese character.
    :param check_str: string to be checked.
    """
    current_path_uni = str(check_str.encode('gbk'), "gbk")
    for ch in current_path_uni:
        assert ch < '\u4e00' or ch > '\u9fff', 'Error: the path can not contain Chinese characters.'


def results_dir_check(results_dir):
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            print('results_dir:', results_dir)
        except OSError:
            pass
    # else:
    #     # 先删除再创建
    #
    #     try:
    #         shutil.rmtree(results_dir)
    #         print('results_dir:', results_dir)
    #         os.makedirs(results_dir)
    #     except OSError:
    #         pass


def make_params_dicts(params_list_dict):
    params_dict_list = []
    key_list = list(params_list_dict.keys())
    for value_comb in product(*list(params_list_dict.values())):
        temp_dict = {}
        for i in range(len(value_comb)):
            temp_dict[key_list[i]] = value_comb[i]
        params_dict_list.append(temp_dict)
    return params_dict_list


def residue_check(ml, cv, fragment, statistics_label):
    assert ml != 'CRF' or cv != 'i', "Error: the CRF can only use the k-fold cross-validation"
    assert ml != 'crf' or fragment == 1, "Sorry, If you use fragment method only svm and rf can be used!"
    # Judge the crf, only use for binary situation.
    assert ml != 'crf' or (len(statistics_label) != 2), "Error: the CRF only use for binary situation!"


def ohe_method_error(args):
    assert args.ml not in DeepLearning or args.method not in ['SASA', 'RSS', 'CS'], "One dimension vector isn't " \
                                                                                    "support Depp Learning"


def seq_feature_check(args):
    if args.mode == 'OHE':
        assert args.method in Method_One_Hot_Enc, "Please check method for 'OHE' mode!"

    if args.mode == 'BOW':
        if args.category == 'DNA':
            assert args.words in DNA_Words, "Please check words for 'BOW' mode of DNA sequence!"
        elif args.category == 'RNA':
            assert args.words in RNA_Words, "Please check words for 'BOW' mode of RNA sequence!"
        else:
            assert args.words in Protein_Words, "Please check words for 'BOW' mode of Protein sequence!"
    if args.mode == 'TM':
        assert args.method in Method_Topic_Model, "Please check method for 'TM' mode!"

    if args.mode == 'WE':
        assert args.method in Method_Word_Embedding, "Please check method for 'WE' mode!"

    if args.mode == 'SR':
        assert args.method in Method_Syntax_Rules, "Please check method for 'SR' mode!"

    if args.mode == 'AF':
        assert args.method in Method_Auto_features, "Please check method for 'AF' mode!"


def res_feature_check(args):
    if args.mode == 'OHE':
        assert args.method in Method_Res, "Please check method for 'OHE' mode!"


def f_range(start, stop, step):
    return takewhile(lambda x: x < stop, count(start, step))


def one_hot_check(args, **params_dict):
    # if method in ['DPC', 'TPC', 'PP']:
    #     params_dict['chosen_name'] = args.chosen_name
    # if method in ['PSSM', 'PSFM', 'SS', 'SASA', 'CS']:
    params_dict['cpu'] = [args.cpu]
    return params_dict


def bow_check(args, **params_dict):
    if args.words in ['Kmer', 'RevKmer', 'Mismatch', 'Subsequence']:
        if args.auto_opt == 1:
            params_dict['word_size'] = list(range(1, 5, 1))
        elif args.auto_opt == 2:
            params_dict['word_size'] = list(range(1, 7, 1))  # 这里之后估计得具体区分一下蛋白质和基因
        else:
            if args.word_size is not None:
                if len(args.word_size) == 1:
                    params_dict['word_size'] = list(range(args.word_size[0], args.word_size[0] + 1, 1))
                    # args.word_size.pop()
                elif len(args.word_size) == 2:
                    params_dict['word_size'] = list(range(args.word_size[0], args.word_size[1] + 1, 1))
                    # args.word_size.pop()
                    # args.word_size.pop()
                elif len(args.word_size) == 3:
                    params_dict['word_size'] = list(range(args.word_size[0], args.word_size[1] + 1, args.word_size[2]))
                    # args.word_size.pop()
                    # args.word_size.pop()
                    # args.word_size.pop()
                else:
                    error_info = 'The number of input value of parameter "word_size" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "word_size" missed!'
                sys.stderr.write(error_info)
                return False
    if args.words in ['Mismatch']:
        if args.auto_opt == 1:
            params_dict['mis_num'] = list(range(1, 5, 1))
        elif args.auto_opt == 2:
            params_dict['mis_num'] = list(range(1, 7, 1))
        else:
            if args.mis_num is not None:
                if len(args.mis_num) == 1:
                    params_dict['mis_num'] = list(range(args.mis_num[0], args.mis_num[0] + 1, 1))
                    # args.mis_num.pop()
                elif len(args.mis_num) == 2:
                    params_dict['mis_num'] = list(range(args.mis_num[0], args.mis_num[1] + 1, 1))
                    # args.mis_num.pop()
                    # args.mis_num.pop()
                elif len(args.mis_num) == 3:
                    params_dict['mis_num'] = list(range(args.mis_num[0], args.mis_num[1] + 1, args.mis_num[2]))
                    # args.mis_num.pop()
                    # args.mis_num.pop()
                    # args.mis_num.pop()
                else:
                    error_info = 'The number of input value of parameter "mis_num" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "mis_num" missed!'
                sys.stderr.write(error_info)
                return False
    if args.words in ['Subsequence']:
        if args.auto_opt == 1:
            params_dict['delta'] = list(f_range(0, 0.8, 0.2))
        elif args.auto_opt == 2:
            params_dict['delta'] = list(f_range(0, 1, 0.1))
        else:
            if args.delta is not None:
                if len(args.delta) == 1:
                    params_dict['delta'] = list(f_range(args.delta[0], args.delta[0] + 0.1, 0.1))
                    # args.delta.pop()
                elif len(args.delta) == 2:
                    params_dict['delta'] = list(f_range(args.delta[0], args.delta[1] + 0.1, 0.1))
                    # args.delta.pop()
                    # args.delta.pop()
                elif len(args.delta) == 3:
                    params_dict['delta'] = list(f_range(args.delta[0], args.delta[1] + 0.1, args.delta[2]))
                    # args.delta.pop()
                    # args.delta.pop()
                    # args.delta.pop()
                else:
                    error_info = 'The number of input value of parameter "delta" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "delta" missed!'
                sys.stderr.write(error_info)
                return False
    if args.words in ['Top-N-Gram']:
        if args.auto_opt == 1:
            params_dict['top_n'] = list(range(1, 3, 1))
        elif args.auto_opt == 2:
            params_dict['top_n'] = list(range(1, 4, 1))
        else:
            if args.top_n is not None:
                if len(args.top_n) == 1:
                    params_dict['top_n'] = list(range(args.top_n[0], args.top_n[0] + 1, 1))
                    # args.top_n.pop()
                elif len(args.top_n) == 2:
                    params_dict['top_n'] = list(range(args.top_n[0], args.top_n[1] + 1, 1))
                    # args.top_n.pop()
                    # args.top_n.pop()
                elif len(args.top_n) == 3:
                    params_dict['top_n'] = list(range(args.top_n[0], args.top_n[1] + 1, args.top_n[2]))
                    # args.top_n.pop()
                    # args.top_n.pop()
                    # args.top_n.pop()
                else:
                    error_info = 'The number of input value of parameter "top_n" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "top_n" missed!'
                sys.stderr.write(error_info)
                return False
    if args.words in ['DR', 'DT']:
        if args.auto_opt == 1:
            params_dict['max_dis'] = list(range(1, 5, 1))
        elif args.auto_opt == 2:
            params_dict['max_dis'] = list(range(1, 7, 1))
        else:
            if args.max_dis is not None:
                if len(args.max_dis) == 1:
                    params_dict['max_dis'] = list(range(args.max_dis[0], args.max_dis[0] + 1, 1))
                    # args.max_dis.pop()
                elif len(args.max_dis) == 2:
                    params_dict['max_dis'] = list(range(args.max_dis[0], args.max_dis[1] + 1, 1))
                    # args.max_dis.pop()
                    # args.max_dis.pop()
                elif len(args.max_dis) == 3:
                    params_dict['max_dis'] = list(range(args.max_dis[0], args.max_dis[1] + 1, args.max_dis[2]))
                    # args.max_dis.pop()
                    # args.max_dis.pop()
                    # args.max_dis.pop()
                else:
                    error_info = 'The number of input value of parameter "max_dis" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "max_dis" missed!'
                sys.stderr.write(error_info)
    if args.words in ['Top-N-Gram', 'DT']:
        params_dict['cpu'] = [args.cpu]

    # special for Mismatch BOW
    if 'mis_num' in list(params_dict.keys()):
        params_dict['mis_num'] = [1]
        if params_dict['word_size'][0] == 1:
            params_dict['word_size'] = params_dict['word_size'][1:]

    return params_dict


def words_check(args, **params_dict):
    if args.words in ['Kmer', 'RevKmer', 'Mismatch', 'Subsequence', 'Top-N-Gram', 'DR', 'DT']:
        if args.auto_opt == 1:
            params_dict['word_size'] = list(range(1, 5, 1))
        elif args.auto_opt == 2:
            params_dict['word_size'] = list(range(1, 6, 1))
        else:
            if args.word_size is not None:
                if len(args.word_size) == 1:
                    params_dict['word_size'] = list(range(args.word_size[0], args.word_size[0] + 1, 1))
                    # args.word_size.pop()
                elif len(args.word_size) == 2:
                    params_dict['word_size'] = list(range(args.word_size[0], args.word_size[1] + 1, 1))
                    # args.word_size.pop()
                    # args.word_size.pop()
                elif len(args.word_size) == 3:
                    params_dict['word_size'] = list(range(args.word_size[0], args.word_size[1] + 1, args.word_size[2]))
                    # args.word_size.pop()
                    # args.word_size.pop()
                    # args.word_size.pop()
                else:
                    error_info = 'The number of input value of parameter "word_size" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "word_size" missed!'
                sys.stderr.write(error_info)
                return False
    if args.words in ['Top-N-Gram']:
        if args.auto_opt == 1:
            params_dict['top_n'] = list(range(1, 3, 1))
        elif args.auto_opt == 2:
            params_dict['top_n'] = list(range(1, 4, 1))
        else:
            if args.top_n is not None:
                if len(args.top_n) == 1:
                    params_dict['top_n'] = list(range(args.top_n[0], args.top_n[0] + 1, 1))
                    # args.top_n.pop()
                elif len(args.top_n) == 2:
                    params_dict['top_n'] = list(range(args.top_n[0], args.top_n[1] + 1, 1))
                    # args.top_n.pop()
                    # args.top_n.pop()
                elif len(args.top_n) == 3:
                    params_dict['top_n'] = list(range(args.top_n[0], args.top_n[1] + 1, args.top_n[2]))
                    # args.top_n.pop()
                    # args.top_n.pop()
                    # args.top_n.pop()
                else:
                    error_info = 'The number of input value of parameter "top_n" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "top_n" missed!'
                sys.stderr.write(error_info)
                return False
    if args.words in ['DR', 'DT']:
        if args.auto_opt == 1:
            params_dict['max_dis'] = list(range(1, 5, 1))
        elif args.auto_opt == 2:
            params_dict['max_dis'] = list(range(1, 7, 1))
        else:
            if args.max_dis is not None:
                if len(args.max_dis) == 1:
                    params_dict['max_dis'] = list(range(args.max_dis[0], args.max_dis[0] + 1, 1))
                    # args.max_dis.pop()
                elif len(args.max_dis) == 2:
                    params_dict['max_dis'] = list(range(args.max_dis[0], args.max_dis[1] + 1, 1))
                    # args.max_dis.pop()
                    # args.max_dis.pop()
                elif len(args.max_dis) == 3:
                    params_dict['max_dis'] = list(range(args.max_dis[0], args.max_dis[1] + 1, args.max_dis[2]))
                    # args.max_dis.pop()
                    # args.max_dis.pop()
                    # args.max_dis.pop()
                else:
                    error_info = 'The number of input value of parameter "max_dis" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "max_dis" missed!'
                sys.stderr.write(error_info)
    if args.words in ['Top-N-Gram', 'DT']:
        params_dict['cpu'] = [args.cpu]
    return params_dict


def tr_check(args, **params_dict):
    params_dict = words_check(args, **params_dict)
    params_dict['alpha'] = [args.alpha]
    return params_dict


def we_check(args, **params_dict):
    params_dict = words_check(args, **params_dict)
    if args.method == 'Glove':
        params_dict['win_size'] = [args.win_size if args.win_size is not None else 10]
        params_dict['vec_dim'] = [args.vec_dim if args.vec_dim is not None else 100]
    else:
        params_dict['sg'] = [args.sg]
        params_dict['win_size'] = [args.win_size if args.win_size is not None else 5]
        params_dict['vec_dim'] = [args.vec_dim if args.vec_dim is not None else 10]
    return params_dict


def tm_check(args, **params_dict):
    if args.in_tm == 'BOW':
        params_dict = bow_check(args, **params_dict)
    else:
        params_dict = words_check(args, **params_dict)
        if args.in_tm == 'TextRank':
            params_dict = tr_check(args, **params_dict)
            params_dict['win_size'] = [args.win_size if args.win_size is not None else 3]
    params_dict['com_prop'] = [args.com_prop]
    return params_dict


METHODS_ACC_S = ['DAC', 'DCC', 'DACC', 'TAC', 'TCC', 'TACC', 'AC', 'CC', 'ACC']


def sr_check(args, **params_dict):
    if args.method in ['MAC', 'GAC', 'NMBAC', 'PDT', 'PDT-Profile', 'ZCPseKNC']:
        if args.auto_opt == 1:
            params_dict['lamada'] = list(range(1, 8, 1))
        elif args.auto_opt == 2:
            params_dict['lamada'] = list(range(1, 10, 1))
        else:
            if args.lamada is not None:
                if len(args.lamada) == 1:
                    params_dict['lamada'] = list(range(args.lamada[0], args.lamada[0] + 1, 1))
                    # args.lamada.pop()
                elif len(args.lamada) == 2:
                    params_dict['lamada'] = list(range(args.lamada[0], args.lamada[1] + 1, 1))
                    # args.lamada.pop()
                    # args.lamada.pop()
                elif len(args.lamada) == 3:
                    params_dict['lamada'] = list(range(args.lamada[0], args.lamada[1] + 1, args.lamada[2]))
                    # args.lamada.pop()
                    # args.lamada.pop()
                    # args.lamada.pop()
                else:
                    error_info = 'The number of input value of parameter "lamada" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "lamada" missed!'
                sys.stderr.write(error_info)
                return False
    if args.method in METHODS_ACC_S or args.method in ['ACC-PSSM', 'AC-PSSM', 'CC-PSSM']:
        if args.auto_opt == 1:
            params_dict['lag'] = list(range(1, 8, 1))
        elif args.auto_opt == 2:
            params_dict['lag'] = list(range(1, 10, 1))
        else:
            if args.lag is not None:
                if len(args.lag) == 1:
                    params_dict['lag'] = list(range(args.lag[0], args.lag[0] + 1, 1))
                    # args.lag.pop()
                elif len(args.lag) == 2:
                    params_dict['lag'] = list(range(args.lag[0], args.lag[1] + 1, 1))
                    # args.lag.pop()
                    # args.lag.pop()
                elif len(args.lag) == 3:
                    params_dict['lag'] = list(range(args.lag[0], args.lag[1] + 1, args.lag[2]))
                    # args.lag.pop()
                    # args.lag.pop()
                    # args.lag.pop()
                else:
                    error_info = 'The number of input value of parameter "lag" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "lag" missed!'
                sys.stderr.write(error_info)
                return False
    if args.method in ['ZCPseKNC']:
        if args.auto_opt == 1:
            params_dict['w'] = list(f_range(0, 0.8, 0.1))
        elif args.auto_opt == 2:
            params_dict['w'] = list(f_range(0, 1, 0.1))
        else:
            if args.w is not None:
                if len(args.w) == 1:
                    params_dict['w'] = list(f_range(args.w[0], args.w[0] + 0.1, 0.1))
                    # args.w.pop()
                elif len(args.w) == 2:
                    params_dict['w'] = list(f_range(args.w[0], args.w[1] + 0.1, 0.1))
                    # args.w.pop()
                    # args.w.pop()
                elif len(args.w) == 3:
                    params_dict['w'] = list(f_range(args.w[0], args.w[1] + 0.1, args.w[2]))
                    # args.w.pop()
                    # args.w.pop()
                    # args.w.pop()
                else:
                    error_info = 'The number of input value of parameter "w" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "w" missed!'
                sys.stderr.write(error_info)
                return False
        if args.auto_opt == 1:
            params_dict['k'] = list(range(1, 5, 1))
        elif args.auto_opt == 2:
            params_dict['k'] = list(range(1, 6, 1))
        else:
            if args.k is not None:
                if len(args.k) == 1:
                    params_dict['k'] = list(range(args.k[0], args.k[0] + 1, 1))
                    # args.k.pop()
                elif len(args.k) == 2:
                    params_dict['k'] = list(range(args.k[0], args.k[1] + 1, 1))
                    # args.k.pop()
                    # args.k.pop()
                elif len(args.k) == 3:
                    params_dict['k'] = list(range(args.k[0], args.k[1] + 1, args.k[2]))
                    # args.k.pop()
                    # args.k.pop()
                    # args.k.pop()
                else:
                    error_info = 'The number of input value of parameter "k" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "k" missed!'
                sys.stderr.write(error_info)
                return False

    if args.method in ['PDT-Profile']:
        if args.auto_opt == 1:
            params_dict['n'] = list(range(1, 3, 1))
        elif args.auto_opt == 2:
            params_dict['n'] = list(range(1, 4, 1))
        else:
            if args.n is not None:
                if len(args.n) == 1:
                    params_dict['n'] = list(range(args.n[0], args.n[0] + 1, 1))

                elif len(args.n) == 2:
                    params_dict['n'] = list(range(args.n[0], args.n[1] + 1, 1))

                elif len(args.n) == 3:
                    params_dict['n'] = list(range(args.n[0], args.n[1] + 1, args.n[2]))
                else:
                    error_info = 'The number of input value of parameter "n" should be no more than 3!'
                    sys.stderr.write(error_info)
                    return False
            else:
                error_info = 'Parameter "n" missed!'
                sys.stderr.write(error_info)
                return False
    if args.method == 'Motif-PSSM':
        params_dict['batch_size'] = [args.batch_size]

    if args.method in ['PDT-Profile', 'AC-PSSM', 'CC-PSSM', 'ACC-PSSM', 'PSSM-DT', 'PSSM-RT', 'Motif-PSSM']:
        params_dict['cpu'] = [args.cpu]

    return params_dict


def af_check(args, **params_dict):
    params_dict['prob'] = [args.dropout]
    params_dict['lr'] = [args.lr]
    params_dict['epoch'] = [args.epochs]
    params_dict['batch_size'] = [args.batch_size]
    params_dict['fea_dim'] = [args.fea_dim]
    return params_dict


def mode_params_check(args, all_params_list_dict, res=False):
    params_list_dict = {}

    if res is False:
        if args.mode == 'OHE':
            params_list_dict = one_hot_check(args, **params_list_dict)  # Example: {k: [1, 2, 3], w: [0.7, 0.8]}
            all_params_list_dict = one_hot_check(args, **all_params_list_dict)

        elif args.mode == 'BOW':
            params_list_dict = bow_check(args, **params_list_dict)
            all_params_list_dict = bow_check(args, **all_params_list_dict)

        elif args.mode == 'TF-IDF':
            params_list_dict = words_check(args, **params_list_dict)
            all_params_list_dict = words_check(args, **all_params_list_dict)

        elif args.mode == 'TR':
            params_list_dict = tr_check(args, **params_list_dict)
            all_params_list_dict = tr_check(args, **all_params_list_dict)

        elif args.mode == 'WE':
            params_list_dict = we_check(args, **params_list_dict)
            all_params_list_dict = we_check(args, **all_params_list_dict)

        elif args.mode == 'TM':
            params_list_dict = tm_check(args, **params_list_dict)
            all_params_list_dict = tm_check(args, **all_params_list_dict)

        elif args.mode == 'SR':
            params_list_dict = sr_check(args, **params_list_dict)
            all_params_list_dict = sr_check(args, **all_params_list_dict)

        else:
            params_list_dict = af_check(args, **params_list_dict)
            all_params_list_dict = af_check(args, **all_params_list_dict)

        return params_list_dict, all_params_list_dict
    else:
        params_list_dict = one_hot_check(args, **params_list_dict)  # Example: {k: [1, 2, 3], w: [0.7, 0.8]}
        return params_list_dict


def svm_params_check(cost, gamma, grid=0, param_list_dict=None):  # 1: meticulous; 0: 'rough'.
    if cost is not None:
        if len(cost) == 1:
            c_range = range(cost[0], cost[0] + 1, 1)
            cost.pop()
        elif len(cost) == 2:
            c_range = range(cost[0], cost[1] + 1, 1)
            cost.pop()
            cost.pop()
        elif len(cost) == 3:
            c_range = range(cost[0], cost[1] + 1, cost[2])
            cost.pop()
            cost.pop()
            cost.pop()
        else:
            error_info = 'The number of input value of parameter "cost" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    else:
        if grid == 0:
            c_range = range(-5, 11, 3)
        else:
            c_range = range(-5, 11, 1)
    if gamma is not None:
        if len(gamma) == 1:
            g_range = range(gamma[0], gamma[0] + 1, 1)
            gamma.pop()
        elif len(gamma) == 2:
            g_range = range(gamma[0], gamma[1] + 1, 1)
            gamma.pop()
            gamma.pop()
        elif len(gamma) == 3:
            g_range = range(gamma[0], gamma[1] + 1, gamma[2])
            gamma.pop()
            gamma.pop()
            gamma.pop()
        else:
            error_info = 'The number of input value of parameter "gamma" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    else:
        if grid == 0:
            g_range = range(-10, 6, 3)
        else:
            g_range = range(-10, 6, 1)
    param_list_dict['cost'] = list(c_range)
    param_list_dict['gamma'] = list(g_range)
    # test mode on/off
    # param_list_dict['cost'] = [10]
    # param_list_dict['gamma'] = [5]
    return param_list_dict


def rf_params_check(tree, grid='r', param_list_dict=None):  # 'm': meticulous; 'r': 'rough'.
    if tree is not None:
        if len(tree) == 1:
            t_range = range(tree[0], tree[0] + 10, 10)
            tree.pop()
        elif len(tree) == 2:
            t_range = range(tree[0], tree[1] + 10, 10)
            tree.pop()
            tree.pop()
        elif len(tree) == 3:
            t_range = range(tree[0], tree[1] + 10, tree[2])
            tree.pop()
            tree.pop()
            tree.pop()
        else:
            error_info = 'The number of input value of parameter "cost" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    else:
        if grid == 'r':
            t_range = range(100, 600, 200)
        else:
            t_range = range(100, 600, 100)
    param_list_dict['tree'] = list(t_range)
    return param_list_dict


def ml_params_check(args, all_params_list_dict):
    if args.ml == 'SVM':
        all_params_list_dict = svm_params_check(args.cost, args.gamma, args.grid, all_params_list_dict)
    else:
        all_params_list_dict = rf_params_check(args.tree, args.grid, all_params_list_dict)
    return all_params_list_dict


# 深度学习的参数检查
def dl_params_check(args, params_list_dict):
    params_list_dict['lr'] = [args.lr]
    params_list_dict['epochs'] = [args.epochs]
    params_list_dict['batch_size'] = [args.batch_size]
    params_list_dict['dropout'] = [args.dropout]

    if args.ml == 'LSTM' or args.ml == 'GRU':
        params_list_dict['hidden_dim'] = [args.hidden_dim]
        params_list_dict['n_layer'] = [args.n_layer]
    elif args.ml == 'CNN':
        params_list_dict['out_channels'] = [args.out_channels]
        params_list_dict['kernel_size'] = [args.kernel_size]
    elif args.ml == 'Transformer' or args.ml == 'Weighted-Transformer':
        params_list_dict['d_model'] = [args.d_model]
        params_list_dict['d_ff'] = [args.d_ff]
        params_list_dict['n_heads'] = [args.n_heads]
        params_list_dict['n_layer'] = [args.n_layer]
    else:
        params_list_dict['d_model'] = [args.d_model]
        params_list_dict['d_ff'] = [args.d_ff]
        params_list_dict['n_heads'] = [args.n_heads]
        params_list_dict['n_layer'] = [args.n_layer]
        params_list_dict['n_chunk'] = [args.n_chunk]
        params_list_dict['rounds'] = [args.rounds]
        params_list_dict['bucket_length'] = [args.bucket_length]
    return params_list_dict


def crf_params_check(args, params_list_dict):
    params_list_dict['lr'] = [args.lr]
    params_list_dict['epochs'] = [args.epochs]
    params_list_dict['batch_size'] = [args.batch_size]
    return params_list_dict

# def table_params(params_dict, opt=False):
#     tb = pt.PrettyTable()
#
#     if opt is False:
#         print('Parameter details'.center(21, '*'))
#         tb.field_names = ["parameter", "value"]
#     else:
#         print('\n')
#         print('\n')
#         print('+---------------------------+')
#         print('| Optimal parameter details |')
#         print('+---------------------------+')
#         tb.field_names = ["parameter", "optimal value"]
#     for item in list(params_dict.items()):
#         if item[0] not in ['out_files', 'ind_out_files']:
#             tb.add_row(item)
#     print(tb)
#     print('\n')


def prepare4train_seq(args, label_array, dl):
    info_dict = {}

    if args.cv == 'j':
        args.folds_num = sum(args.sample_num_list)
        info_dict['Validation method'] = 'Jackknife cross validation'
    elif args.cv == '10':
        args.folds_num = 10
        info_dict['Validation method'] = '10-fold cross validation'
    else:
        args.folds_num = 5
        info_dict['Validation method'] = '5-fold cross validation'
    args.folds = construct_partition2two(label_array, args.folds_num, True)  # 固定交叉验证的每一折index
    if dl is False:
        args.metric_index = Metric_Index[args.metric]
        info_dict['Metric for selection'] = Metric_dict[args.metric]

        if args.sp != 'none':
            if args.sp == 'over':
                info_dict['Technique for sampling'] = 'oversampling '
            elif args.sp == 'under':
                info_dict['Technique for sampling'] = 'undersampling '
            else:
                info_dict['Technique for sampling'] = 'combine oversampling  and undersampling '

    if len(Counter(label_array).keys()) > 2:  # 判断是二分类还是多分类
        args.multi = True
        info_dict['Type of Problem'] = 'Multi-class classification'
    else:
        args.multi = False
        info_dict['Type of Problem'] = 'Binary classification'

    print_base_dict(info_dict)
    return args


def print_base_dict(info_dict):
    print('\r')
    key_max_len = 0
    val_max_len = 0
    for key, val in info_dict.items():
        key_max_len = max(key_max_len, len(key))
        val_max_len = max(val_max_len, len(val))

    tag = '--'
    header = 'Basic settings for Parameter Selection'
    header_str1 = '+' + tag.center(key_max_len + val_max_len + 9, '-') + '+'
    header_str2 = '|' + header.center(key_max_len + val_max_len + 9, ' ') + '|'
    print(header_str1)
    print(header_str2)

    up_dn_str = '+' + tag.center(key_max_len + 4, '-') + '+' + tag.center(val_max_len + 4, '-') + '+'
    print(up_dn_str)
    for key, val in info_dict.items():
        temp_str = '|' + str(key).center(key_max_len + 4, ' ') + '|' + str(val).center(val_max_len + 4, ' ') + '|'
        print(temp_str)
        print(up_dn_str)

    print('\r')


def print_fe_dict(params_dict):
    print_dict = {}
    for key in list(params_dict.keys()):
        if key in ['word_size', 'mis_num', 'delta', 'top_n', 'alpha', 'win_size', 'vec_dim', 'lamada', 'lag',
                   'w', 'k', 'n']:
            print_dict[key] = params_dict[key]

    if print_dict != {}:
        print('\r')
        key_max_len = 20
        val_max_len = 20

        tag = '--'
        header = 'Feature Extraction Parameters'
        header_str1 = '+' + tag.center(key_max_len + val_max_len + 9, '-') + '+'
        header_str2 = '|' + header.center(key_max_len + val_max_len + 9, ' ') + '|'
        print(header_str1)
        print(header_str2)

        up_dn_str = '+' + tag.center(key_max_len + 4, '-') + '+' + tag.center(val_max_len + 4, '-') + '+'
        print(up_dn_str)
        for key, val in print_dict.items():
            temp_str = '|' + str(key).center(key_max_len + 4, ' ') + '|' + str(val).center(val_max_len + 4, ' ') + '|'
            print(temp_str)
            print(up_dn_str)

        print('\r')


def prepare4train_res(args, label_array, dl):
    info_dict = {}

    if args.cv == 'j':
        args.folds_num = sum(args.sample_num_list)
        info_dict['Validation method'] = 'Jackknife cross validation'
    elif args.cv == '10':
        args.folds_num = 10
        info_dict['Validation method'] = '10-fold cross validation'
    else:
        args.folds_num = 5
        info_dict['Validation method'] = '5-fold cross validation'

    if dl is False:
        args.folds = construct_partition2two(label_array, args.folds_num, True)  # 固定交叉验证的每一折index
        args.metric_index = Metric_Index[args.metric]
        info_dict['Metric for selection'] = Metric_dict[args.metric]
    else:
        label_array = random.normal(loc=0.0, scale=1, size=(len(label_array)))
        args.folds = construct_partition2two(label_array, args.folds_num, False)  # 固定交叉验证的每一折index

    # if len(Counter(label_array).keys()) > 2:  # 判断是二分类还是多分类
    #     args.multi = True
    #     info_dict['Type of Problem'] = 'Multi-class classification'
    # else:
    args.multi = False
    info_dict['Type of Problem'] = 'Binary classification'

    print_base_dict(info_dict)
    return args


def get_params(params_file_name):
    params_dict = {}
    with open(params_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if len(line) != 0:
                param = line.split('=')[0].split()[0]
                value = line.split('=')[1].split()[0]
                params_dict[param] = value
    print("params_dict: ", params_dict)
    return params_dict
