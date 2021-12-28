import multiprocessing
import os
import time

from CheckAll import Machine_Learning_Algorithm, DeepLearning, prepare4train_res, dl_params_check, make_params_dicts
from FeatureExtractionMode.utils.utils_write import opt_params2file, read_res_label_file
from MachineLearningAlgorithm.Classification.ml_machine import ml_cv_process
from MachineLearningAlgorithm.Classification.ml_machine import ml_cv_results, ml_ind_results
from MachineLearningAlgorithm.SequenceLabelling.dl_machine import dl_cv_process as res_dcp
from MachineLearningAlgorithm.SequenceLabelling.dl_machine import dl_ind_process as res_dip
from MachineLearningAlgorithm.utils.utils_read import files2vectors_res, read_base_mat4res, res_label_read


def res_cl_process(args):
    # 读取特征向量文件
    vectors, sp_num_list = files2vectors_res(args.vec_file, args.format)
    # 根据不同标签样本数目生成标签数组
    label_array = res_label_read(sp_num_list, args.label)
    # ** 残基层面特征提取和标签数组生成完毕 ** #

    # 在参数便利前进行一系列准备工作: 1. 固定划分；2.设定指标；3.指定任务类型
    args = prepare4train_res(args, label_array, dl=False)

    # ** 通过遍历SVM/RF参数字典列表来筛选参数 ** #
    # SVM/RF参数字典
    params_dict_list = args.params_dict_list
    # 多进程控制
    pool = multiprocessing.Pool(args.cpu)
    params_dict_list_pro = []
    for i in range(len(params_dict_list)):
        params_dict = params_dict_list[i]
        params_dict_list_pro.append(pool.apply_async(one_cl_process, (args, vectors, label_array, args.folds,
                                                                      params_dict)))

    pool.close()
    pool.join()
    # ** 筛选结束 ** #

    # 根据指标进行参数选择
    params_selected = params_select(params_dict_list_pro, args.results_dir)

    # 构建分类器
    ml_cv_results(args.ml, vectors, label_array, args.folds, args.sp, args.multi, args.res, args.results_dir,
                  params_selected)

    # -------- 独立测试-------- #
    # 即，将独立测试数据集在最优的model上进行测试
    if args.ind_vec_file is not None:
        # 读取特征向量文件
        ind_vectors, ind_sp_num_list = files2vectors_res(args.ind_vec_file, args.format)
        # 根据不同标签样本数目生成标签数组
        ind_label_array = res_label_read(ind_sp_num_list, args.label)
        # ** 残基层面特征提取和标签数组生成完毕 ** #
        ml_ind_results(args.ml, ind_vectors, ind_label_array, args.multi, args.res, args.results_dir, params_selected)
    # -------- 独立测试-------- #


def res_dl_process(args):

    # 读取标签列表和标签长度列表  res_labels_list --> list[list1, list2,..]
    args.res_labels_list, label_len_list = read_res_label_file(args.label_file)

    all_params_list_dict = {}  # 包含了机器学习和特征提取的参数
    if args.ml in DeepLearning:
        all_params_list_dict = dl_params_check(args, all_params_list_dict)
        # 列表字典 ---> 字典列表
        args.params_dict_list = make_params_dicts(all_params_list_dict)
    # 深度学习参数字典
    params_dict = args.params_dict_list[0]
    # 读取base文件向量,对向量矩阵和序列长度数组进行处理
    vec_mat, fixed_seq_len_list = read_base_mat4res(args.vec_file[0], args.fixed_len)

    # 不同于SVM/RF, 深度学习
    if args.ind_vec_file is None:
        # 在参数便利前进行一系列准备工作: 1. 固定划分；2.设定指标；3.指定任务类型
        args = prepare4train_res(args, args.res_labels_list, dl=True)

        res_dcp(args.ml, vec_mat, args.res_labels_list, fixed_seq_len_list, args.fixed_len, args.folds,
                args.results_dir, params_dict)
    else:
        ind_res_dl_fe_process(args, vec_mat, args.res_labels_list, fixed_seq_len_list, params_dict)


def ind_res_dl_fe_process(args, vec_mat, res_labels_list, fixed_seq_len_list, params_dict):
    print('########################## Independent Test Begin ##########################\n')

    ind_vec_mat, ind_fixed_seq_len_list = read_base_mat4res(args.ind_fea_file, args.fixed_len)

    res_dip(args.ml, vec_mat, res_labels_list, fixed_seq_len_list, ind_vec_mat, args.ind_res_labels_list,
            ind_fixed_seq_len_list, args.fixed_len, args.results_dir, params_dict)

    print('########################## Independent Test Finish ##########################\n')


def one_cl_process(args, vectors, labels, folds, params_dict):
    params_dict = ml_cv_process(args.ml, vectors, labels, folds, args.metric_index, args.sp, args.multi, args.res,
                                params_dict)
    return params_dict


def params_select(params_list, out_dir):
    evaluation = params_list[0].get()['metric']
    params_list_selected = params_list[0].get()
    for i in range(len(params_list)):
        if params_list[i].get()['metric'] > evaluation:
            evaluation = params_list[i].get()['metric']
            params_list_selected = params_list[i].get()
    del params_list_selected['metric']
    opt_params2file(params_list_selected, out_dir)  # 将最优参数写入文件

    return params_list_selected


def main(args):
    print("\nStep into analysis...\n")
    start_time = time.time()

    args.results_dir = os.path.dirname(os.path.abspath(args.vec_file[0])) + '/'
    args.res = True

    if args.ml in DeepLearning:
        res_dl_process(args)
    else:
        res_cl_process(args)

    print("Done.")
    print(("Used time: %.2fs" % (time.time() - start_time)))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='BioSeq-NLP', description="Step into analysis, please select parameters ")

    parse.add_argument('-ml', type=str, choices=Machine_Learning_Algorithm, required=True,
                       help="The machine-learning algorithm for constructing predictor, "
                            "for example: Support Vector Machine (SVM).")

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
                       help="The dimension of multi-head attention layer for Transformer or Weighted-Transformer "
                            "or Reformer.")
    parse.add_argument('-d_ff', type=int, default=1024,
                       help="The dimension of feed forward layer of Transformer or Weighted-Transformer "
                            "or Reformer.")
    parse.add_argument('-n_heads', type=int, default=4,
                       help="The number of heads for multi-head attention.")
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
    parse.add_argument('-vec_file', nargs='*', required=True, help="The input feature vector file(s).")
    parse.add_argument('-label_file', required=True, help="The corresponding label file is required.")
    parse.add_argument('-ind_vec_file', nargs='*', help="The feature vector files of independent test dataset.")
    parse.add_argument('-ind_label_file', help="The corresponding label file of independent test dataset.")
    parse.add_argument('-fixed_len', type=int,
                       help="The length of sequence will be fixed via cutting or padding. If you don't set "
                            "value for 'fixed_len', it will be the maximum length of all input sequences. ")
    # parameters for output
    parse.add_argument('-format', default='csv', choices=['tab', 'svm', 'csv', 'tsv'],
                       help="The output format (default = csv).\n"
                            "tab -- Simple format, delimited by TAB.\n"
                            "svm -- The libSVM training data format.\n"
                            "csv, tsv -- The format that can be loaded into a spreadsheet program.")
    argv = parse.parse_args()
    main(argv)
