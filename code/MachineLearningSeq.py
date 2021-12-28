import multiprocessing
import os
import time

from CheckAll import ml_params_check, dl_params_check, make_params_dicts, Classification, DeepLearning, \
    Method_Semantic_Similarity, prepare4train_seq
from FeatureExtractionMode.utils.utils_write import opt_params2file, gen_label_array, fixed_len_control
from MachineLearningAlgorithm.Classification.dl_machine import dl_cv_process, dl_ind_process
from MachineLearningAlgorithm.Classification.ml_machine import ml_cv_process, ml_cv_results, ml_ind_results, \
    ml_score_cv_process, ml_score_cv_results, ml_score_ind_results
from MachineLearningAlgorithm.utils.utils_read import files2vectors_info, seq_label_read, read_dl_vec4seq
from SemanticSimilarity import ind_score_process


def ml_process(args):
    # 从输入的向量文件获取特征向量，标签数组和样本数目
    vectors, sp_num_list, vec_files = files2vectors_info(args.vec_file, args.format)
    # 生成标签数组
    label_array = gen_label_array(sp_num_list, args.label)

    # 对SVM或RF的参数进行检查并生成参数字典集合
    all_params_list_dict = {}
    all_params_list_dict = ml_params_check(args, all_params_list_dict)
    # 列表字典 ---> 字典列表
    params_dict_list = make_params_dicts(all_params_list_dict)
    # 并行参数筛选
    pool = multiprocessing.Pool(args.cpu)
    # 在参数便利前进行一系列准备工作: 1. 固定划分；2.设定指标；3.指定任务类型
    args = prepare4train_seq(args, label_array, dl=False)

    params_dict_list_pro = []
    for i in range(len(params_dict_list)):
        params_dict = params_dict_list[i]
        params_dict['out_files'] = vec_files
        params_dict_list_pro.append(pool.apply_async(one_ml_process, (args, vectors, label_array, args.folds, vec_files,
                                                                      params_dict)))

    pool.close()
    pool.join()

    params_selected = params_select(params_dict_list_pro, args.results_dir)
    ml_results(args, vectors, label_array, args.folds, params_selected['out_files'], params_selected)

    if args.ind_vec_file is not None:
        # 从输入的独立测试向量文件获取特征向量，标签数组和样本数目
        ind_vectors, ind_sp_num_list, ind_vec_files = files2vectors_info(args.ind_vec_file, args.format)
        # 生成标签数组
        ind_label_array = gen_label_array(ind_sp_num_list, args.label)
        ind_ml_results(args, vectors, label_array, ind_vectors, ind_label_array, params_selected)


def one_ml_process(args, vectors, labels, folds, vec_files, params_dict):
    if args.score == 'none':
        params_dict = ml_cv_process(args.ml, vectors, labels, folds, args.metric_index, args.sp, args.multi, args.res,
                                    params_dict)
    else:
        params_dict = ml_score_cv_process(args.ml, vec_files, args.folds_num, args.metric_index,
                                          args.sp, args.multi, args.format, params_dict)
    return params_dict


def ml_results(args, vectors, labels, folds, vec_files, params_selected):
    if args.score == 'none':
        ml_cv_results(args.ml, vectors, labels, folds, args.sp, args.multi, args.res, args.results_dir, params_selected)
    else:
        ml_score_cv_results(args.ml, vec_files, labels, args.folds_num, args.sp, args.multi,
                            args.format, args.results_dir, params_selected)
    return params_selected


def ind_ml_results(args, vectors, labels, ind_vectors, ind_labels, params_selected):
    if args.score == 'none':
        ml_ind_results(args.ml, ind_vectors, ind_labels, args.multi, args.res, args.results_dir, params_selected)
    else:
        ind_score_process(args.score, vectors, args.ind_vec_file, labels, ind_labels, args.format, args.cpu)
        ml_score_ind_results(args.ml, args.ind_vec_file[0], args.sp, args.multi, args.format,
                             args.results_dir, params_selected)


def params_select(params_list, out_dir):
    evaluation = params_list[0].get()['metric']
    params_list_selected = params_list[0].get()
    for i in range(len(params_list)):
        if params_list[i].get()['metric'] > evaluation:
            evaluation = params_list[i].get()['metric']
            params_list_selected = params_list[i].get()
    del params_list_selected['metric']
    # params_list_selected['PSI'] = metric
    # table_params(params_list_selected, True)  # 打印选择后的参数
    opt_params2file(params_list_selected, out_dir)  # 将最优参数写入文件

    return params_list_selected


def dl_process(args):
    # 从输入的向量文件获取特征向量，标签数组和样本数目
    # fixed_seq_len_list: 最大序列长度为fixed_len的序列长度的列表
    vectors, sp_num_list, fixed_seq_len_list = read_dl_vec4seq(args.fixed_len, args.vec_file, return_sp=True)
    # 生成标签数组
    label_array = gen_label_array(sp_num_list, args.label)
    # 控制序列的固定长度
    args.fixed_len = fixed_len_control(fixed_seq_len_list, args.fixed_len)

    # 对Deep Learning的参数进行检查并生成参数字典集合
    all_params_list_dict = {}
    all_params_list_dict = dl_params_check(args, all_params_list_dict)
    # 列表字典 ---> 字典列表
    params_dict = make_params_dicts(all_params_list_dict)[0]

    # split data set according to cross validation approach
    if args.ind_vec_file is None:
        # 在参数便利前进行一系列准备工作: 1. 固定划分；2.设定指标；3.指定任务类型
        args = prepare4train_seq(args, label_array, dl=True)

        dl_cv_process(args.ml, vectors, label_array, fixed_seq_len_list, args.fixed_len, args.folds, args.results_dir,
                      params_dict)
    else:
        # 从输入的向量文件获取特征向量，标签数组和样本数目
        ind_vectors, ind_sp_num_list, ind_fixed_seq_len_list = read_dl_vec4seq(args, args.vec_file, return_sp=True)
        ind_label_array = seq_label_read(ind_sp_num_list, args.label)

        dl_ind_process(args.ml, vectors, label_array, fixed_seq_len_list, args.ind_vectors, ind_label_array,
                       ind_fixed_seq_len_list, args.fixed_len, args.results_dir, params_dict)


def main(args):
    print("\nStep into analysis...\n")
    start_time = time.time()

    args.results_dir = os.path.dirname(os.path.abspath(args.vec_file[0])) + '/'
    args.res = False

    if args.ml in DeepLearning:
        dl_process(args)
    else:
        ml_process(args)

    print("Done.")
    print(("Used time: %.2fs" % (time.time() - start_time)))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='BioSeq-NLP', description="Step into analysis, please select parameters ")

    parse.add_argument('-ml', type=str, choices=Classification, required=True,
                       help="The machine learning algorithm, for example: Support Vector Machine(SVM).")

    # parameters for scoring
    parse.add_argument('-score', type=str, choices=Method_Semantic_Similarity, default='none',
                       help="Choose whether calculate semantic similarity score and what method for calculation.")
    # ----------------------- parameters for MachineLearning---------------------- #
    parse.add_argument('-cpu', type=int, default=1,
                       help="The number of CPU cores used for multiprocessing during parameter selection process."
                            "(default=1).")
    parse.add_argument('-grid', type=int, nargs='*', choices=[0, 1], default=0,
                       help="grid = 0 for rough grid search, grid = 1 for meticulous grid search.")
    # parameters for svm
    parse.add_argument('-cost', type=int, nargs='*', help="Regularization parameter of 'SVM'.")
    parse.add_argument('-gamma', type=int, nargs='*', help="Kernel coefficient for 'rbf' of 'SVM'.")
    # parameters for rf
    parse.add_argument('-tree', type=int, nargs='*', help="The number of trees in the forest for 'RF'.")

    # ----------------------- parameters for DeepLearning---------------------- #
    parse.add_argument('-lr', type=float, default=0.99, help="The value of learning rate for deep learning.")
    parse.add_argument('-epochs', type=int, help="The epoch number for train deep model.")
    parse.add_argument('-batch_size', type=int, default=50, help="The size of mini-batch for deep learning.")
    parse.add_argument('-dropout', type=float, default=0.6, help="The value of dropout prob for deep learning.")
    # parameters for LSTM, GRU
    parse.add_argument('-hidden_dim', type=int, default=256,
                       help="The size of the intermediate (a.k.a., feed forward) layer.")
    parse.add_argument('-n_layer', type=int, default=2, help="The number of units for 'LSTM' and 'GRU'.")
    # parameters for CNN
    parse.add_argument('-out_channels', type=int, default=256, help="The number of output channels for 'CNN'.")
    parse.add_argument('-kernel_size', type=int, default=5, help="The size of stride for 'CNN'.")
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
    parse.add_argument('-vec_file', nargs='*', required=True, help="The input feature vector files.")
    parse.add_argument('-label', type=int, nargs='*', required=True,
                       help="The corresponding label of input sequence files. For deep learning method, the label can "
                            "only set as positive integer")
    parse.add_argument('-ind_vec_file', nargs='*', help="The input feature vector files of independent test dataset.")
    parse.add_argument('-fixed_len', type=int,
                       help="The length of sequence will be fixed via cutting or padding. If you don't set "
                            "value for 'fixed_len', it will be the maximum length of all input sequences. ")
    parse.add_argument('-format', default='csv', choices=['tab', 'svm', 'csv', 'tsv'],
                       help="The output format (default = csv).\n"
                            "tab -- Simple format, delimited by TAB.\n"
                            "svm -- The libSVM training data format.\n"
                            "csv, tsv -- The format that can be loaded into a spreadsheet program.")

    argv = parse.parse_args()
    main(argv)
