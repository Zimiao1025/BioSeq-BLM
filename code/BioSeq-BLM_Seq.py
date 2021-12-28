import multiprocessing
import os
import time

from CheckAll import Batch_Path_Seq, DeepLearning, Classification, Method_Semantic_Similarity, prepare4train_seq
from CheckAll import Method_One_Hot_Enc, Feature_Extract_Mode, check_contain_chinese, seq_sys_check, dl_params_check, \
    seq_feature_check, mode_params_check, results_dir_check, ml_params_check, make_params_dicts, Final_Path, All_Words
from FeatureAnalysis import fa_process
from FeatureExtractionMode.utils.utils_write import seq_file2one, gen_label_array, out_seq_file, out_ind_file, \
    opt_file_copy, out_dl_seq_file, create_all_seq_file, fixed_len_control
from FeatureExtractionSeq import one_seq_fe_process
from MachineLearningAlgorithm.Classification.dl_machine import dl_cv_process, dl_ind_process
from MachineLearningAlgorithm.utils.utils_read import files2vectors_seq, read_dl_vec4seq
from MachineLearningSeq import one_ml_process, params_select, ml_results, ind_ml_results
from FeatureExtractionMode.SR.pse import AAIndex


def create_results_dir(args, cur_dir):
    if args.bp == 1:
        results_dir = cur_dir + Batch_Path_Seq + str(args.category) + "/" + str(args.mode) + "/"

        if args.method is not None:
            results_dir += str(args.method) + "/"
        if args.in_tm is not None:
            results_dir += str(args.in_tm) + "/"
        if args.in_af is not None:
            results_dir += str(args.in_af) + "/"
        if args.words is not None:
            results_dir += str(args.words) + "/"
        if args.score != 'none':
            results_dir += str(args.score) + "/"
    else:
        results_dir = cur_dir + Final_Path
    results_dir_check(results_dir)
    return results_dir


def ml_fe_process(args):
    # 合并序列文件
    input_one_file = create_all_seq_file(args.seq_file, args.results_dir)
    # 统计样本数目和序列长度
    sp_num_list, seq_len_list = seq_file2one(args.category, args.seq_file, args.label, input_one_file)
    # 生成标签数组
    label_array = gen_label_array(sp_num_list, args.label)
    # 控制序列的固定长度(只需要操作一次）
    args.fixed_len = fixed_len_control(seq_len_list, args.fixed_len)
    # 多进程计算
    pool = multiprocessing.Pool(args.cpu)
    # 对每个mode的method进行检查
    seq_feature_check(args)
    # 对SVM或RF的参数进行检查并生成参数字典集合
    all_params_list_dict = {}
    all_params_list_dict = ml_params_check(args, all_params_list_dict)
    # 对每个mode的words和method的参数进行检查
    # params_list_dict 为只包括特征提取的参数的字典， all_params_list_dict为包含所有参数的字典
    params_list_dict, all_params_list_dict = mode_params_check(args, all_params_list_dict)
    # 列表字典 ---> 字典列表
    params_dict_list = make_params_dicts(all_params_list_dict)
    # print(params_dict_list)
    # exit()
    # 在参数便利前进行一系列准备工作: 1. 固定划分；2.设定指标；3.指定任务类型
    args = prepare4train_seq(args, label_array, dl=False)

    # 指定分析层面
    args.res = False

    params_dict_list_pro = []
    print('\n')
    print('Parameter Selection Processing...')
    print('\n')
    for i in range(len(params_dict_list)):
        params_dict = params_dict_list[i]
        # 生成特征向量文件名
        vec_files = out_seq_file(args.label, args.format, args.results_dir, params_dict, params_list_dict)
        params_dict['out_files'] = vec_files
        # 注意多进程计算的debug
        # one_ml_fe_process(args, input_one_file, label_array, vec_files, sp_num_list, args.folds, **params_dict)
        params_dict_list_pro.append(pool.apply_async(one_ml_fe_process, (args, input_one_file, label_array, vec_files,
                                                                         sp_num_list, args.folds, params_dict)))

    pool.close()
    pool.join()
    # exit()
    # 根据指标进行参数选择
    params_selected = params_select(params_dict_list_pro, args.results_dir)
    # 将最优的特征向量文件从"all_fea_files/"文件夹下复制到主文件下
    opt_files = opt_file_copy(params_selected['out_files'], args.results_dir)
    # 获取最优特征向量
    opt_vectors = files2vectors_seq(opt_files, args.format)
    print(' Shape of Optimal Feature vectors: [%d, %d] '.center(66, '*') % (opt_vectors.shape[0], opt_vectors.shape[1]))
    # 特征分析
    if args.score == 'none':
        opt_vectors = fa_process(args, opt_vectors, label_array, after_ps=True, ind=False)
        print(' Shape of Optimal Feature vectors after FA process: [%d, %d] '.center(66, '*') % (opt_vectors.shape[0],
                                                                                                 opt_vectors.shape[1]))
    # 构建分类器
    ml_results(args, opt_vectors, label_array, args.folds, params_selected['out_files'], params_selected)
    # -------- 独立测试-------- #
    # 即，将独立测试数据集在最优的model上进行测试
    if args.ind_seq_file is not None:
        ind_ml_fe_process(args, opt_vectors, label_array, params_selected)


def one_ml_fe_process(args, input_one_file, labels, vec_files, sp_num_list, folds, params_dict):
    # 特征提取
    # args.res = False
    one_seq_fe_process(args, input_one_file, labels, vec_files, sp_num_list, False, **params_dict)
    # 获取特征向量
    vectors = files2vectors_seq(vec_files, args.format)
    print(' Shape of Feature vectors: [%d, %d] '.center(66, '*') % (vectors.shape[0], vectors.shape[1]))
    if args.score == 'none':
        vectors = fa_process(args, vectors, labels, after_ps=False, ind=False)
        print(' Shape of Feature vectors after FA process: [%d, %d] '.center(66, '*') % (vectors.shape[0],
                                                                                         vectors.shape[1]))

    params_dict = one_ml_process(args, vectors, labels, folds, vec_files, params_dict)

    return params_dict


def ind_ml_fe_process(args, vectors, labels, params_selected):
    print('########################## Independent Test Begin ##########################\n')
    # 合并独立测试集序列文件
    ind_input_one_file = create_all_seq_file(args.ind_seq_file, args.results_dir)
    # 统计独立测试集样本数目和序列长度
    ind_sp_num_list, ind_seq_len_list = seq_file2one(args.category, args.ind_seq_file, args.label, ind_input_one_file)
    # 生成独立测试集标签数组
    ind_label_array = gen_label_array(ind_sp_num_list, args.label)

    # 生成独立测试集特征向量文件名
    ind_out_files = out_ind_file(args.label, args.format, args.results_dir)
    # 特征提取
    one_seq_fe_process(args, ind_input_one_file, ind_label_array, ind_out_files, ind_sp_num_list, True,
                       **params_selected)
    # 获取独立测试集特征向量
    ind_vectors = files2vectors_seq(ind_out_files, args.format)
    print(' Shape of Ind Feature vectors: [%d, %d] '.center(66, '*') % (ind_vectors.shape[0], ind_vectors.shape[1]))
    if args.score == 'none':
        ind_vectors = fa_process(args, ind_vectors, ind_label_array, after_ps=True, ind=True)
        print(' Shape of Ind Feature vectors after FA process: [%d, %d] '.center(66, '*') % (ind_vectors.shape[0],
                                                                                             ind_vectors.shape[1]))
    # 为独立测试集构建分类器
    args.ind_vec_file = ind_out_files
    ind_ml_results(args, vectors, labels, ind_vectors, ind_label_array, params_selected)

    print('########################## Independent Test Finish ##########################\n')


def dl_fe_process(args):
    # 合并序列文件
    input_one_file = create_all_seq_file(args.seq_file, args.results_dir)
    # 统计样本数目和序列长度
    sp_num_list, seq_len_list = seq_file2one(args.category, args.seq_file, args.label, input_one_file)
    # 生成标签数组
    label_array = gen_label_array(sp_num_list, args.label)

    # 控制序列的固定长度(仅仅需要在基准数据集上操作一次）
    args.fixed_len = fixed_len_control(seq_len_list, args.fixed_len)

    all_params_list_dict = {}
    all_params_list_dict = dl_params_check(args, all_params_list_dict)
    # 对每个mode的words和method的参数进行检查
    # params_list_dict 为只包括特征提取的参数的字典， all_params_list_dict为包含所有参数的字典
    params_list_dict, all_params_list_dict = mode_params_check(args, all_params_list_dict)
    # 列表字典 ---> 字典列表 --> 参数字典
    params_dict = make_params_dicts(all_params_list_dict)[0]
    # 特征向量文件命名
    out_files = out_dl_seq_file(args.label, args.results_dir, ind=False)
    # 深度特征向量提取
    one_seq_fe_process(args, input_one_file, label_array, out_files, sp_num_list, False, **params_dict)
    # 获取深度特征向量
    # fixed_seq_len_list: 最大序列长度为fixed_len的序列长度的列表
    vectors, fixed_seq_len_list = read_dl_vec4seq(args.fixed_len, out_files, return_sp=False)

    # 深度学习的独立测试和交叉验证分开
    if args.ind_seq_file is None:
        # 在参数便利前进行一系列准备工作: 1. 固定划分；2.设定指标；3.指定任务类型
        args = prepare4train_seq(args, label_array, dl=True)
        # 构建深度学习分类器
        dl_cv_process(args.ml, vectors, label_array, fixed_seq_len_list, args.fixed_len, args.folds, args.results_dir,
                      params_dict)
    else:
        # 独立验证开始
        ind_dl_fe_process(args, vectors, label_array, fixed_seq_len_list, params_dict)


def ind_dl_fe_process(args, vectors, labels, fixed_seq_len_list, params_dict):
    print('########################## Independent Test Begin ##########################\n')
    # 合并独立测试集序列文件
    ind_input_one_file = create_all_seq_file(args.ind_seq_file, args.results_dir)
    # 统计独立测试集样本数目和序列长度
    ind_sp_num_list, ind_seq_len_list = seq_file2one(args.category, args.ind_seq_file, args.label, ind_input_one_file)

    # 生成独立测试集标签数组
    ind_label_array = gen_label_array(ind_sp_num_list, args.label)

    # 生成独立测试集特征向量文件名
    ind_out_files = out_dl_seq_file(args.label, args.results_dir, ind=True)
    # 特征提取
    one_seq_fe_process(args, ind_input_one_file, ind_label_array, ind_out_files, ind_sp_num_list, True, **params_dict)
    # 获取独立测试集特征向量
    ind_vectors, ind_fixed_seq_len_list = read_dl_vec4seq(args.fixed_len, ind_out_files, return_sp=False)

    # 为独立测试构建深度学习分类器
    dl_ind_process(args.ml, vectors, labels, fixed_seq_len_list, ind_vectors, ind_label_array, ind_fixed_seq_len_list,
                   args.fixed_len, args.results_dir, params_dict)

    print('########################## Independent Test Finish ##########################\n')


def main(args):
    print("\nStep into analysis...\n")
    start_time = time.time()

    current_path = os.path.dirname(os.path.realpath(__file__))
    args.current_dir = os.path.dirname(os.getcwd())

    # 判断中文目录
    check_contain_chinese(current_path)

    # 判断mode和ml的组合是否合理
    seq_sys_check(args)

    # 生成结果文件夹
    args.results_dir = create_results_dir(args, args.current_dir)

    if args.ml in DeepLearning:
        args.dl = 1
        dl_fe_process(args)
    else:
        args.dl = 0
        ml_fe_process(args)

    print("Done.")
    print(("Used time: %.2fs" % (time.time() - start_time)))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='BioSeq-BLM', description="Step into analysis, please select parameters ")

    # ----------------------- parameters for FeatureExtraction ---------------------- #
    parse.add_argument('-category', type=str, choices=['DNA', 'RNA', 'Protein'], required=True,
                       help="The category of input sequences.")

    parse.add_argument('-mode', type=str, choices=Feature_Extract_Mode, required=True,
                       help="The feature extraction mode for input sequence which analogies with NLP, "
                            "for example: bag of words (BOW).")
    parse.add_argument('-score', type=str, choices=Method_Semantic_Similarity, default='none',
                       help="Choose whether calculate semantic similarity score for feature vectors.")
    parse.add_argument('-words', type=str, choices=All_Words,
                       help="If you select mode in ['BOW', 'TF-IDF', 'TR', 'WE', 'TM'], you should select word for "
                            "corresponding mode, for example Mismatch. Pay attention to that "
                            "different category has different words, please reference to manual.")
    parse.add_argument('-method', type=str,
                       help="If you select mode in ['OHE', 'WE', 'TM', 'SR', 'AF'], you should select method for "
                            "corresponding mode, for example select 'LDA' for 'TM' mode, select 'word2vec' for 'WE'"
                            " mode and so on. For different category, the methods belong to 'OHE' and 'SR' mode is "
                            "different, please reference to manual")
    parse.add_argument('-auto_opt', type=int, default=0, choices=[0, 1, 2],
                       help="Choose whether automatically traverse the argument list. "
                            "2 is automatically traversing the argument list set ahead, 1 is automatically traversing "
                            "the argument list in a smaller range, while 0 is not (default=0).")
    # parameters for one-hot encoding
    parse.add_argument('-cpu', type=int, default=1,
                       help="The maximum number of CPU cores used for multiprocessing in generating frequency profile"
                            " and the number of CPU cores used for multiprocessing during parameter selection process "
                            "(default=1).")
    parse.add_argument('-pp_file', type=str,
                       help="The physicochemical properties file user input.\n"
                            "if input nothing, the default physicochemical properties is:\n"
                            "DNA dinucleotide: Rise, Roll, Shift, Slide, Tilt, Twist.\n"
                            "DNA trinucleotide: Dnase I, Bendability (DNAse).\n"
                            "RNA: Rise, Roll, Shift, Slide, Tilt, Twist.\n"
                            "Protein: Hydrophobicity, Hydrophilicity, Mass.")
    parse.add_argument('-rss_file', type=str,
                       help="The second structure file for all input sequences.(The order of a specific sequence "
                            "should be corresponding to the order in 'all_seq_file.txt' file")
    # parameters for bag of words
    parse.add_argument('-word_size', type=int, nargs='*', default=[2],
                       help="The word size of sequences for specific words "
                            "(the range of word_size is between 1 and 6).")
    parse.add_argument('-mis_num', type=int, nargs='*', default=[1],
                       help="For Mismatch words. The max value inexact matching, mis_num should smaller than word_size "
                            "(the range of mis_num is between 1 and 6).")
    parse.add_argument('-delta', type=float, nargs='*', default=[0.5],
                       help="For Subsequence words. The value of penalized factor "
                            "(the range of delta is between 0 and 1).")
    parse.add_argument('-top_n', type=int, nargs='*', default=[1],
                       help="The maximum distance between structure statuses (the range of delta is between 1 and 4)."
                            "It works with Top-n-gram words.")
    parse.add_argument('-max_dis', type=int, nargs='*', default=[1],
                       help="The max distance value for DR words and DT words (default range is from 1 to 4).")
    # parameters for TextRank
    parse.add_argument('-alpha', type=float, default=0.85,
                       help="Damping parameter for PageRank which used in 'TR' mode, default=0.85.")
    # parameters for word embedding
    parse.add_argument('-win_size', type=int,
                       help="The maximum distance between the current and predicted word within a sentence for "
                            "'word2vec' in 'WE' mode, etc.")
    parse.add_argument('-vec_dim', type=int,
                       help="The output dimension of feature vectors for 'Glove' model and dimensionality of a word "
                            "vectors for 'word2vec' and 'fastText' method.")
    parse.add_argument('-sg', type=int, default=0,
                       help="Training algorithm for 'word2vec' and 'fastText' method. 1 for skip-gram, otherwise CBOW.")
    # parameters for topic model
    parse.add_argument('-in_tm', type=str, choices=['BOW', 'TF-IDF', 'TextRank'],
                       help="While topic model implement subject extraction from a text, the text need to be "
                            "preprocessed by one of mode in choices.")
    parse.add_argument('-com_prop', type=float, default=0.8,
                       help="If choose topic model mode, please set component proportion for output feature vectors.")
    # parameters for syntax rules
    parse.add_argument('-oli', type=int, choices=[0, 1], default=0,
                       help="Choose one kind of Oligonucleotide (default=0): 0 represents dinucleotid; "
                            "1 represents trinucleotide. For MAC, GAC, NMBAC methods of 'SR' mode.")
    parse.add_argument('-lag', type=int, nargs='*', default=[1],
                       help="The value of lag (default=1). For DACC, TACC, ACC, ACC-PSSM, AC-PSSM or CC-PSSM methods"
                            " and so on.")
    parse.add_argument('-lamada', type=int, nargs='*', default=[1],
                       help="The value of lamada (default=1). For MAC, PDT, PDT-Profile, GAC or NMBAC methods "
                            "and so on.")
    parse.add_argument('-w', type=float, nargs='*', default=[0.8],
                       help="The value of weight (default=0.1). For ZCPseKNC method.")
    parse.add_argument('-k', type=int, nargs='*', default=[3],
                       help="The value of Kmer, it works only with ZCPseKNC method.")
    parse.add_argument('-n', type=int, nargs='*', default=[1],
                       help="The maximum distance between structure statuses (default=1). "
                            "It works with PDT-Profile method.")
    parse.add_argument('-ui_file', help="The user-defined physicochemical property file.")
    parse.add_argument('-all_index', dest='a', action='store_true', help="Choose all physicochemical indices.")
    parse.add_argument('-no_all_index', dest='a', action='store_false',
                       help="Do not choose all physicochemical indices, default.")
    parse.set_defaults(a=False)
    # parameters for automatic features/deep learning algorithm
    parse.add_argument('-in_af', type=str, choices=Method_One_Hot_Enc,
                       help="Choose the input for 'AF' mode from 'OHE' mode.")
    parse.add_argument('-fea_dim', type=int, default=256,
                       help="The output dimension of feature vectors, it works with 'AF' mode.")
    parse.add_argument('-motif_database', type=str, choices=['ELM', 'Mega'],
                       help="The database where input motif file comes from.")
    parse.add_argument('-motif_file', type=str,
                       help="The short linear motifs from ELM database or structural motifs from the MegaMotifBase.")
    # ----------------------- parameters for feature analysis ---------------------- #
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
    # ----------------------- parameters for MachineLearning---------------------- #
    parse.add_argument('-ml', type=str, choices=Classification, required=True,
                       help="The machine learning algorithm, for example: Support Vector Machine(SVM).")
    parse.add_argument('-grid', type=int, nargs='*', choices=[0, 1], default=0,
                       help="grid = 0 for rough grid search, grid = 1 for meticulous grid search.")
    # parameters for svm
    parse.add_argument('-cost', type=int, nargs='*', help="Regularization parameter of 'SVM'.")
    parse.add_argument('-gamma', type=int, nargs='*', help="Kernel coefficient for 'rbf' of 'SVM'.")
    # parameters for rf
    parse.add_argument('-tree', type=int, nargs='*', help="The number of trees in the forest for 'RF'.")
    # ----------------------- parameters for DeepLearning---------------------- #
    parse.add_argument('-lr', type=float, default=0.99,
                       help="The value of learning rate, it works with 'AF' mode and deep learning algorithm.")
    parse.add_argument('-epochs', type=int,
                       help="The epoch number of train process for 'AF' mode and deep learning algorithm.")
    parse.add_argument('-batch_size', type=int, default=5,
                       help="The size of mini-batch, it works with 'AF' mode and deep learning algorithm.")
    parse.add_argument('-dropout', type=float, default=0.6,
                       help="The value of dropout prob, it works with 'AF' mode and deep learning algorithm.")
    # parameters for LSTM, GRU
    parse.add_argument('-hidden_dim', type=int, default=256,
                       help="The size of the intermediate (a.k.a., feed forward) layer, it works with 'AF' mode, "
                            "GRU and LSTM.")
    parse.add_argument('-n_layer', type=int, default=2,
                       help="The number of units for LSTM and GRU, it works with 'AF' mode, GRU and LSTM.")
    # parameters for CNN
    parse.add_argument('-out_channels', type=int, default=256, help="The number of output channels for 'CNN'.")
    parse.add_argument('-kernel_size', type=int, default=5, help="The size of stride for CNN.")
    # parameters for Transformer and Weighted-Transformer
    parse.add_argument('-d_model', type=int, default=256,
                       help="The dimension of multi-head attention layer for Transformer or Weighted-Transformer.")
    parse.add_argument('-d_ff', type=int, default=1024,
                       help="The dimension of fully connected layer of Transformer or Weighted-Transformer.")
    parse.add_argument('-n_heads', type=int, default=4,
                       help="The number of heads for Transformer or Weighted-Transformer.")
    # parameters for Reformer
    parse.add_argument('-n_chunk', type=int, default=8,
                       help="The number of chunks for processing lsh attention.")
    parse.add_argument('-rounds', type=int, default=1024,
                       help="The number of rounds for multiple rounds of hashing to reduce probability that similar "
                            "items fall in different buckets.")
    parse.add_argument('-bucket_length', type=int, default=64,
                       help="Average size of qk per bucket, 64 was recommended in paper")
    # parameters for ML parameter selection and cross validation
    parse.add_argument('-metric', type=str, choices=['Acc', 'MCC', 'AUC', 'BAcc', 'F1'], default='Acc',
                       help="The metric for parameter selection")
    parse.add_argument('-cv', choices=['5', '10', 'j'], default='5',
                       help="The cross validation mode.\n"
                            "5 or 10: 5-fold or 10-fold cross validation.\n"
                            "j: (character 'j') jackknife cross validation.")
    parse.add_argument('-sp', type=str, choices=['none', 'over', 'under', 'combine'], default='none',
                       help="Select technique for oversampling.")
    # ----------------------- parameters for input and output ---------------------- #
    # parameters for input
    parse.add_argument('-seq_file', nargs='*', required=True, help="The input files in FASTA format.")
    parse.add_argument('-label', type=int, nargs='*', required=True,
                       help="The corresponding label of input sequence files. For deep learning method, the label can "
                            "only set as positive integer")
    parse.add_argument('-ind_seq_file', nargs='*', help="The input independent test files in FASTA format.")

    parse.add_argument('-fixed_len', type=int,
                       help="The length of sequence will be fixed via cutting or padding. If you don't set "
                            "value for 'fixed_len', it will be the maximum length of all input sequences. ")
    # parameters for output
    parse.add_argument('-format', default='csv', choices=['tab', 'svm', 'csv', 'tsv'],
                       help="The output format (default = csv).\n"
                            "tab -- Simple format, delimited by TAB.\n"
                            "svm -- The libSVM training data format.\n"
                            "csv, tsv -- The format that can be loaded into a spreadsheet program.")
    parse.add_argument('-bp', type=int, choices=[0, 1], default=0,
                       help="Select use batch mode or not, the parameter will change the directory for generating file "
                            "based on the method you choose.")
    argv = parse.parse_args()
    main(argv)
