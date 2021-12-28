import multiprocessing
import os
import time

from CheckAll import Method_One_Hot_Enc, Feature_Extract_Mode, All_Words, FE_PATH_Seq, FE_BATCH_PATH_Seq, \
    Method_Semantic_Similarity
from CheckAll import check_contain_chinese, seq_feature_check, mode_params_check, results_dir_check, \
    make_params_dicts, print_fe_dict
from FeatureExtractionMode.utils.utils_write import seq_file2one, gen_label_array, out_seq_file, out_dl_seq_file, \
    create_all_seq_file, fixed_len_control
from SemanticSimilarity import score_process


def create_results_dir(args, cur_dir):
    if args.bp == 1:
        results_dir = cur_dir + FE_BATCH_PATH_Seq + str(args.category) + "/" + str(args.mode) + "/"

        if args.method is not None:
            results_dir += str(args.method) + "/"
        if args.in_tm is not None:
            results_dir += str(args.in_tm) + "/"
        if args.in_af is not None:
            results_dir += str(args.in_af) + "/"
        if args.words is not None:
            results_dir += str(args.words) + "/"
    else:
        results_dir = cur_dir + FE_PATH_Seq
    results_dir_check(results_dir)
    return results_dir


def seq_fe_process(args):
    current_path = os.path.dirname(os.path.realpath(__file__))
    args.current_dir = os.path.dirname(os.getcwd())
    # 判断是否包含中文路径
    check_contain_chinese(current_path)

    # 生成结果文件夹
    args.results_dir = create_results_dir(args, args.current_dir)

    # 合并序列文件
    input_one_file = create_all_seq_file(args.seq_file, args.result_dir)
    # 统计样本数目和序列长度
    sp_num_list, seq_len_list = seq_file2one(args.category, args.seq_file, args.label, input_one_file)
    # 生成标签数组
    label_array = gen_label_array(sp_num_list, args.label)
    # 控制序列的固定长度
    args.fixed_len = fixed_len_control(seq_len_list, args.fixed_len)

    # 通过遍历参数字典列表来获得特征向量，同时将特征向量写入文件（打分特征除外）
    # 对每个mode的method进行检查
    seq_feature_check(args)
    # 对每个mode的words和method的参数进行检查
    all_params_list_dict = {}  # 适配框架
    # params_list_dict 为只包括特征提取的参数的字典
    params_list_dict, all_params_list_dict = mode_params_check(args, all_params_list_dict)
    params_dict_list = make_params_dicts(all_params_list_dict)
    # 这里的策略是遍历所有数值参数来并行计算

    if args.dl == 0:
        # 多进程计算
        pool = multiprocessing.Pool(args.cpu)
        for i in range(len(params_dict_list)):
            params_dict = params_dict_list[i]
            vec_files = out_seq_file(args.label, args.format, args.results_dir, params_dict, params_list_dict)
            params_dict['out_files'] = vec_files
            # 注意参数报错pool并不会显示，所以需要测试模式，而非直接并行
            # 测试模式
            # one_seq_fe_process(args, input_one_file, labels, vec_files, sample_num_list, False, **params_dict)
            pool.apply_async(one_seq_fe_process, (args, input_one_file, label_array, vec_files, sp_num_list, False,
                                                  params_dict))

        pool.close()
        pool.join()
    else:
        params_dict = params_dict_list[0]
        vec_files = out_dl_seq_file(args.label, args.results_dir, ind=False)
        params_dict['out_files'] = vec_files
        one_seq_fe_process(args, input_one_file, label_array, vec_files, sp_num_list, False, **params_dict)


def one_seq_fe_process(args, input_one_file, labels, vec_files, sample_num_list, ind, **params_dict):

    print_fe_dict(params_dict)  # 输出特征提取参数详细信息

    if args.mode == 'OHE':
        from FeatureExtractionMode.OHE.OHE4vec import ohe2seq_vec, ohe2seq_mat
        for out_file in vec_files:
            if not os.path.exists(out_file):
                if args.dl == 0:
                    ohe2seq_vec(input_one_file, args.category, args.method, args.current_dir, args.pp_file,
                                args.rss_file, sample_num_list, args.fixed_len, args.format, vec_files, args.cpu)
                    if args.score != 'none' and ind is False:
                        score_process(args.score, vec_files, labels, args.cv, args.format, args.cpu)
                else:
                    ohe2seq_mat(input_one_file, args.category, args.method, args.current_dir, args.pp_file,
                                args.rss_file, sample_num_list, vec_files, args.cpu)

    elif args.mode == 'BOW':
        from FeatureExtractionMode.BOW.BOW4vec import bow
        for out_file in vec_files:
            if not os.path.exists(out_file):
                bow(input_one_file, args.category, args.words, sample_num_list, args.format,
                    vec_files, args.current_dir, False, **params_dict)
                if args.score != 'none' and ind is False:
                    score_process(args.score, vec_files, labels, args.cv, args.format, args.cpu)

    elif args.mode == 'TF-IDF':
        from FeatureExtractionMode.TF_IDF.TF_IDF4vec import tf_idf
        for out_file in vec_files:
            if not os.path.exists(out_file):
                tf_idf(input_one_file, args.category, args.words, args.fixed_len, sample_num_list,
                       args.format, vec_files, args.current_dir, False, **params_dict)
                if args.score != 'none' and ind is False:
                    score_process(args.score, vec_files, labels, args.cv, args.format, args.cpu)

    elif args.mode == 'TR':
        from FeatureExtractionMode.TR.TR4vec import text_rank
        for out_file in vec_files:
            if not os.path.exists(out_file):
                text_rank(input_one_file, args.category, args.words, args.fixed_len, sample_num_list,
                          args.format, vec_files, args.current_dir, False, **params_dict)
                if args.score != 'none' and ind is False:
                    score_process(args.score, vec_files, labels, args.cv, args.format, args.cpu)
    elif args.mode == 'WE':
        from FeatureExtractionMode.WE.WE4vec import word_emb
        for out_file in vec_files:
            if not os.path.exists(out_file):
                word_emb(args.method, input_one_file, args.category, args.words, args.fixed_len,
                         sample_num_list, args.format, vec_files, args.current_dir, **params_dict)
                if args.score != 'none' and ind is False:
                    score_process(args.score, vec_files, labels, args.cv, args.format, args.cpu)

    elif args.mode == 'TM':
        from FeatureExtractionMode.TM.TM4vec import topic_model
        for out_file in vec_files:
            if not os.path.exists(out_file):
                topic_model(args.in_tm, args.method, input_one_file, labels, args.category, args.words, args.fixed_len,
                            sample_num_list, args.format, vec_files, args.current_dir, **params_dict)
                if args.score != 'none' and ind is False:
                    score_process(args.score, vec_files, labels, args.cv, args.format, args.cpu)
    elif args.mode == 'SR':
        from FeatureExtractionMode.SR.SR4vec import syntax_rules
        from FeatureExtractionMode.SR.pse import AAIndex
        for out_file in vec_files:
            if not os.path.exists(out_file):
                syntax_rules(args.method, input_one_file, args.category, sample_num_list,
                             args.format, vec_files, args.current_dir, args, **params_dict)
                if args.score != 'none' and ind is False:
                    score_process(args.score, vec_files, labels, args.cv, args.format, args.cpu)
    else:
        from FeatureExtractionMode.AF.AF4vec import auto_feature
        for out_file in vec_files:
            if not os.path.exists(out_file):
                # method, in_fa, input_file, labels, sample_num_list, out_format, out_file_list, alphabet, cur_dir,
                # chosen_file, cpu, fixed_len, ** params_dict
                auto_feature(args.method, input_one_file, labels, sample_num_list, vec_files, args, **params_dict)
                if args.score != 'none' and ind is False:
                    score_process(args.score, vec_files, labels, args.cv, args.format, args.cpu)


def main(args):
    print("\nStep into analysis...\n")
    start_time = time.time()

    seq_fe_process(args)

    print("Done.")
    print(("Used time: %.2fs" % (time.time() - start_time)))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='BioSeq-NLP', description="Step into analysis, please select parameters ")

    # parameters for whole framework
    parse.add_argument('-dl', type=int, default=0, choices=[0, 1],
                       help="Select whether generate features for deep learning algorithm.")

    parse.add_argument('-category', type=str, choices=['DNA', 'RNA', 'Protein'], required=True,
                       help="The category of input sequences.")

    parse.add_argument('-mode', type=str, choices=Feature_Extract_Mode, required=True,
                       help="The feature extraction mode for input sequence which analogies with NLP, "
                            "for example: bag of words (BOW).")

    # parameters for mode
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
                       help="Damping parameter for PageRank used in 'TR' mode, default=0.85.")

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
    parse.add_argument('-lr', type=float, default=0.99,
                       help="The value of learning rate, it works only with 'AF' mode.")
    parse.add_argument('-epochs', type=int,
                       help="The epoch number of train process for 'AF' mode.")
    parse.add_argument('-batch_size', type=int, default=5,
                       help="The size of mini-batch, it works only with 'AF' mode.")
    parse.add_argument('-dropout', type=float, default=0.6,
                       help="The value of dropout prob, it works only with 'AF' mode.")
    parse.add_argument('-fea_dim', type=int, default=256,
                       help="The output dimension of feature vectors, it works only with 'AF' mode.")
    parse.add_argument('-hidden_dim', type=int, default=256,
                       help="Only for automatic features mode."
                            "The size of the intermediate (a.k.a., feed forward) layer, it works only with 'AF' mode.")
    parse.add_argument('-n_layer', type=int, default=2,
                       help="The number of units for LSTM and GRU, it works only with 'AF' mode.")
    parse.add_argument('-motif_database', type=str, choices=['ELM', 'Mega'],
                       help="The database where input motif file comes from.")
    parse.add_argument('-motif_file', type=str,
                       help="The short linear motifs from ELM database or structural motifs from the MegaMotifBase.")
    # parameters for scoring
    parse.add_argument('-score', type=str, choices=Method_Semantic_Similarity, default='none',
                       help="Choose whether calculate semantic similarity score and what method for calculation.")
    parse.add_argument('-cv', choices=['5', '10', 'j'], default='5',
                       help="The cross validation mode.\n"
                            "5 or 10: 5-fold or 10-fold cross validation,\n"
                            "j: (character 'j') jackknife cross validation.")
    # parameters for input
    parse.add_argument('-seq_file', nargs='*', required=True, help="The input files in FASTA format.")
    parse.add_argument('-label', type=int, nargs='*', required=True,
                       help="The corresponding label of input sequence files")

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
