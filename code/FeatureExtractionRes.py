import os
import time

from CheckAll import FE_PATH_Res, FE_BATCH_PATH_Res
from CheckAll import results_dir_check, check_contain_chinese, seq_sys_check, res_feature_check, Method_Res
from FeatureExtractionMode.OHE.OHE4vec import ohe2res_base, sliding_win2files, mat_list2frag_array
from FeatureExtractionMode.utils.utils_write import read_res_seq_file, read_res_label_file, fixed_len_control, \
    res_file_check, out_res_file, out_dl_frag_file, res_base2frag_vec
from MachineLearningAlgorithm.utils.utils_read import read_base_vec_list4res


def create_results_dir(args, cur_dir):
    if args.bp == 1:
        results_dir = cur_dir + FE_BATCH_PATH_Res + str(args.category) + "/" + str(args.method) + "/"
    else:
        results_dir = cur_dir + FE_PATH_Res
    results_dir_check(results_dir)
    return results_dir


def res_fe_process(args, fragment):
    # ** 残基层面特征提取和标签数组生成开始 ** #
    # 为存储SVM和RF输入特征的文件命名
    out_files = out_res_file(args.label, args.results_dir, args.format, args.fragment, ind=False)
    # 读取base特征文件, 待写入
    vectors_list = read_base_vec_list4res(args.fea_files)
    # fragment判断,生成对应的特征向量
    if fragment == 0:
        assert args.window is not None, "If -fragment is 0, lease set window size!"
        # 在fragment=0时,通过滑窗技巧为每个残基生成特征
        sliding_win2files(vectors_list, args.res_labels_list, args.window, args.format, out_files)
    else:
        # 在fragment=1时, 将每个残基片段的base特征进行flatten
        mat_list2frag_array(vectors_list, args.res_labels_list, args.fixed_len, args.format, out_files)


def frag_fe_process(args):
    # ** 当fragment为1,且选则深度学习特征提取方法时进行下列操作 ** #

    # 生成特征向量文件名
    out_files = out_dl_frag_file(args.label, args.results_dir, ind=False)
    # 生成深度特征向量文件
    res_base2frag_vec(args.fea_file, args.res_labels_list, args.fixed_len, out_files)


def main(args):
    print("\nStep into analysis...\n")
    start_time = time.time()
    current_path = os.path.dirname(os.path.realpath(__file__))
    args.current_dir = os.path.dirname(os.getcwd())

    # 判断中文目录
    check_contain_chinese(current_path)

    # 判断mode和ml的组合是否合理
    args.mode = 'OHE'
    args.score = 'none'
    seq_sys_check(args, True)

    # 生成结果文件夹
    args.results_dir = create_results_dir(args, args.current_dir)

    # 读取序列文件里每条序列的长度
    seq_len_list = read_res_seq_file(args.seq_file, args.category)
    # 读取标签列表和标签长度列表  res_labels_list --> list[list1, list2,..]
    args.res_labels_list, label_len_list = read_res_label_file(args.label_file)
    # fragment=0: 判断标签是否有缺失且最短序列长度是否大于5; fragment=1: 判断标签是否唯一
    res_file_check(seq_len_list, label_len_list, args.fragment)
    # 这里直接针对残基问题设置标签
    args.label = [1, 0]
    # 控制序列的固定长度(只需要在benchmark dataset上操作一次）
    args.fixed_len = fixed_len_control(seq_len_list, args.fixed_len)

    # 对每个残基层面的method进行检查
    res_feature_check(args)

    # 所有res特征在基准数据集上的基础输出文件
    args.fea_file = args.results_dir + 'res_features.txt'
    # 提取残基层面特征,生成向量文件
    ohe2res_base(args.seq_file, args.category, args.method, args.current_dir, args.pp_file, args.rss_file,
                 args.fea_file, args.cpu)

    if args.fragment == 1:
        if args.dl == 1:
            frag_fe_process(args)
        else:
            res_fe_process(args, 1)
    else:
        if args.dl == 1:
            pass
        else:
            res_fe_process(args, 0)

    print("Done.")
    print(("Used time: %.2fs" % (time.time() - start_time)))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='BioSeq-NLP', description="Step into analysis, please select parameters ")

    # parameters for whole framework
    parse.add_argument('-category', type=str, choices=['DNA', 'RNA', 'Protein'], required=True,
                       help="The category of input sequences.")

    parse.add_argument('-method', type=str, required=True, choices=Method_Res,
                       help="Please select feature extraction method for residue level analysis")

    # parameters for residue
    parse.add_argument('-dl', type=int, default=0, choices=[0, 1],
                       help="Select whether use sliding window technique to transform sequence-labelling question "
                            "to classification question")
    parse.add_argument('-window', type=int,
                       help="The window size when construct sliding window technique for allocating every "
                            "label a short sequence")
    parse.add_argument('-fragment', type=int, default=0, choices=[0, 1],
                       help="Please choose whether use the fragment method, 1 is yes while 0 is no.")

    # parameters for one-hot encoding
    parse.add_argument('-cpu', type=int, default=1,
                       help="The maximum number of CPU cores used for multiprocessing in generating frequency profile")
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
    # parameters for input
    parse.add_argument('-seq_file', required=True, help="The input file in FASTA format.")
    parse.add_argument('-label_file', required=True, help="The corresponding label file.")
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
