import os
import shutil
import sys

import numpy as np

from ..utils.utils_const import DNA, RNA, PROTEIN
from ..utils.utils_fasta import get_seqs


class FormatWrite(object):
    def __init__(self, vectors, out_format, out_name):
        self.vectors = vectors
        self.out_format = out_format
        self.out_name = out_name

    def write_svm(self):
        """Write the vectors into disk in livSVM format."""
        len_vector_list = len(self.vectors)
        len_label_list = len(self.vectors)

        if len_vector_list == 0:
            sys.exit("The vector is none.")
        if len_label_list == 0:
            sys.exit("The label is none.")
        if len_vector_list != len_label_list:
            sys.exit("The length of vector and label is different.")

        with open(self.out_name, 'w') as f:
            for ind1, vec in enumerate(self.vectors):
                temp_write = str(self.vectors[ind1])
                for ind2, val in enumerate(vec):
                    temp_write += ' ' + str(ind2 + 1) + ':' + str(vec[ind2])
                f.write(temp_write)
                f.write('\n')

    def write_tab(self):
        """Write the vectors into disk in tab format."""
        with open(self.out_name, 'w') as f:
            for vec in self.vectors:
                f.write(str(vec[0]))
                for val in vec[1:]:
                    f.write('\t' + str(val))
                f.write('\n')

    def write_csv(self):
        """Write the vectors into disk in csv format."""
        import csv
        with open(self.out_name, 'w', newline='') as csv_file:
            spam_writer = csv.writer(csv_file, delimiter=',',
                                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for vec in self.vectors:
                spam_writer.writerow(vec)

    def write_tsv(self):
        """Write the vectors into disk in csv format."""
        import csv
        with open(self.out_name, 'w', newline='') as tsv_file:
            spam_writer = csv.writer(tsv_file, delimiter='\t',
                                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for vec in self.vectors:
                spam_writer.writerow(vec)

    def write_to_file(self):
        if self.out_format == 'svm':
            self.write_svm()
        elif self.out_format == 'tab':
            self.write_tab()
        elif self.out_format == 'csv':
            self.write_csv()
        elif self.out_format == 'tsv':
            self.write_tsv()
        else:
            print('Output file format error! Please check.')
            return False


def vectors2files(vectors, sample_num_list, out_format, out_file_list):
    # print('执行了vectors2files函数吗这里')
    # print(vectors)
    st = 0
    for i in range(len(sample_num_list)):
        ed = st + sample_num_list[i]
        FormatWrite(vectors[st:ed, :], out_format, out_file_list[i]).write_to_file()
        st = ed
    for index, output_file in enumerate(out_file_list):
        out_with_full_path = os.path.abspath(output_file)
        if os.path.isfile(out_with_full_path):
            if index == 0:
                print('The output files of feature vectors for (ind) dataset can be found here:')
            print(out_with_full_path)
    print('\n')


def res_vectors2file(vec, out_format, out_file):
    FormatWrite(vec, out_format, out_file).write_to_file()
    out_with_full_path = os.path.abspath(out_file)
    if os.path.isfile(out_with_full_path):
        print(out_with_full_path)


def dl_vec2file(res_mats, sample_num_list, out_files):
    count = 0
    # print(sample_num_list)
    # print(len(res_mats))
    # print(len(res_mats[0]))
    for out_file in out_files:
        with open(out_file, 'w') as f:
            for i in range(sample_num_list[count]):
                f.write('>vec of sequence: %d\n' % (i+1))
                for j in range(len(res_mats[i])):
                    # print('res_mats[i][j]', res_mats[i][j])
                    for val in list(res_mats[i][j]):
                        f.write('\t' + str(val))
                    f.write('\n')
                f.write('> end')
        count += 1


def write_res_base_vec(res_mats, out_file):
    with open(out_file, 'w') as f:
        for i in range(len(res_mats)):
            f.write('>vec of sequence: %d\n' % (i+1))
            for j in range(len(res_mats[i])):
                for val in list(res_mats[i][j]):
                    f.write('\t' + str(val))
                f.write('\n')
        f.write('> end')


def res_base2frag_vec(in_file, res_labels, fixed_len, out_files):
    vectors_list = []
    f = open(in_file, 'r')
    lines = f.readlines()
    vectors = []
    flag = 0
    for line in lines:
        line = line.strip()
        if len(line) != 0:
            if line[0] != '>':
                vector = line.split('\t')
                print(vector)
                vector = list(map(float, vector))
                vectors.append(vector)
                flag = 1
            else:
                if flag == 1:
                    vectors_list.append(np.array(vectors))
                    vectors = []
                    flag = 0
    f.close()
    label_array = []
    for res_label in res_labels:
        label_array += res_label

    pos_vec_mat = []
    neg_vec_mat = []
    for i in range(len(vectors_list)):
        temp_arr = np.zeros((fixed_len, len(vectors_list[i][0])))
        seq_len = len(vectors_list[i])

        temp_len = min(seq_len, fixed_len)
        temp_arr[:temp_len, :] = vectors_list[i][:temp_len, :]
        if label_array[i] == 1:
            pos_vec_mat.append(temp_arr)
        else:
            neg_vec_mat.append(temp_arr)
    vec_mat_list = pos_vec_mat + neg_vec_mat
    sp_num_list = [len(pos_vec_mat), len(neg_vec_mat)]
    dl_vec2file(vec_mat_list, sp_num_list, out_files)


def fa_vectors2files(vectors, sample_num_list, out_format, file_list):
    out_file_list = []
    for in_file_name in file_list:
        file_dir, file_name = os.path.split(in_file_name)
        fa_file_name = 'fa_' + file_name
        out_file_name = os.path.join(file_dir, fa_file_name)
        out_file_list.append(out_file_name)

    vectors2files(vectors, sample_num_list, out_format, out_file_list)


# def table_sample(level, ml, sample_num_list, label_list, fixed_len, ind):
#     tb = pt.PrettyTable()
#     print('+---------------------------------------------------+')
#     if ind is True:
#         print('|   The information of independent test dataset     |')
#     else:
#         print('|       The information of benchmark dataset        |')
#     print('+---------------------------------------------------+')
#     tb.field_names = ["label of sample", "number of sample"]
#     if ml in ['CNN', 'LSTM', 'GRU', 'Transformer', 'Weighted-Transformer', 'Reformer'] and level == 'residue':
#         tb.add_row([label_list, sample_num_list[0]])
#     else:
#         for label, sample_num in zip(label_list, sample_num_list):
#             tb.add_row([label, sample_num, fixed_len])
#     print(tb)
#     print('\n')
#
#
# def table_params(params_dict, opt=False):
#     tb = pt.PrettyTable()
#
#     if opt is False:
#         print('Parameter details'.center(21, '*'))
#         tb.field_names = ["parameter", "value"]
#     else:
#         print('\n')
#         print('\n')
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


def create_all_seq_file(seq_files, tgt_dir, ind=False):
    suffix = os.path.splitext(seq_files[0])[-1]
    if ind is False:
        return tgt_dir + '/' + 'all_seq_file' + suffix
    else:
        return tgt_dir + '/' + 'ind_all_seq_file' + suffix


def seq_file2one(category, seq_files, label_list, out_file):
    if category == 'DNA':
        alphabet = DNA
    elif category == 'RNA':
        alphabet = RNA
    else:
        alphabet = PROTEIN

    sp_num_list = []  # 每一种标签序列的数目(list[])
    seq_all = []       # 每一种标签序列的列表(list[list[]])
    seq_len_list = []   # 每一种标签序列的长度列表(list[])
    for i in range(len(seq_files)):
        with open(seq_files[i], 'r') as in_f:
            seq_list = get_seqs(in_f, alphabet)
            for seq in seq_list:
                seq_len_list.append(len(seq))
            seq_num = len(seq_list)
            sp_num_list.append(seq_num)
            seq_all.append(seq_list)

    # 写入所有序列
    with open(out_file, 'w') as out_f:
        for i in range(len(label_list)):
            for j in range(len(seq_all[i])):
                out_f.write('>Sequence[' + str(j + 1) + '] | ' + 'Label[' + str(label_list[i]) + ']')
                out_f.write('\n')
                out_f.write(seq_all[i][j])
                out_f.write('\n')

    return sp_num_list, seq_len_list


def gen_label_array(sp_num_list, label_list):
    labels = []
    for i in range(len(sp_num_list)):
        labels += [int(label_list[i])] * sp_num_list[i]
    return np.array(labels)


def fixed_len_control(seq_len_list, fixed_len):
    fixed_len = max(seq_len_list) if fixed_len is None else fixed_len
    return fixed_len


def opt_params2file(selected_params, result_path):
    temp_re = 'Optimal value of all parameters:\n'
    for key, value in list(selected_params.items()):
        if key not in ['out_files', 'ind_out_files']:
            temp_re += str(key) + ' = ' + str(value) + '\n'
    filename = result_path + 'Opt_params.txt'
    with open(filename, 'w') as f:
        f.write(temp_re)

    full_path = os.path.abspath(filename)
    if os.path.isfile(full_path):
        print('The output file for final results can be found:')
        print(full_path)
        print('\n')


def out_seq_file(label_list, out_format, results_dir, params_dict, params_list_dict):
    # 这里需要注意的是比如params_list_dict = {k: [1, 2, 3], w: [0.7, 0.8], n: [3]}, 则最终的输出文件名只包含k和w
    output_file_list = []
    multi_fea = False
    # print(params_list_dict)
    params_val_list = list(params_list_dict.values())
    for params_val in params_val_list:
        if len(params_val) > 1:
            multi_fea = True

    for i in range(len(label_list)):
        if multi_fea is False:
            fea_path = results_dir
        else:
            fea_path = results_dir + 'all_fea_files/'

        if multi_fea is False:
            fea_path += 'cv_features[' + str(label_list[i]) + ']_' + str(out_format) + '.txt'  # 给文件名加上标签
        else:
            for key in params_list_dict.keys():
                if len(params_list_dict[key]) >= 2:
                    fea_path += str(key) + '_' + str(params_dict[key]) + '_'  # For example: _k_2_lag_5
            fea_path += '/'
            if not os.path.exists(fea_path):
                try:
                    os.makedirs(fea_path)
                except OSError:
                    pass
            fea_path += 'cv_features[' + str(label_list[i]) + ']_' + str(out_format) + '.txt'  # 给文件名加上标签
        output_file_list.append(fea_path)

    return output_file_list


def out_ind_file(label, out_format, results_dir):
    output_file_list = []
    for i in range(len(label)):
        fea_path = results_dir
        fea_path += 'ind_features[' + str(label[i]) + ']_' + str(out_format) + '.txt'  # 给文件名加上标签
        output_file_list.append(fea_path)

    return output_file_list


def out_dl_seq_file(label, results_dir, ind=False):
    output_files = []
    for i in range(len(label)):
        if ind is True:
            fea_path = results_dir + 'ind_dl_features[' + str(label[i]) + ']_.txt'
        else:
            fea_path = results_dir + 'cv_dl_features[' + str(label[i]) + ']_.txt'

        output_files.append(fea_path)

    return output_files


def out_res_file(label, results_dir, out_format, fragment, ind):
    output_files = []
    for i in range(len(label)):
        if ind is True:
            if fragment == 0:
                fea_path = results_dir + 'ind_res_features[' + str(label[i]) + ']_' + str(out_format) + '.txt'
            else:
                fea_path = results_dir + 'ind_res_frag_features[' + str(label[i]) + ']_' + str(out_format) + '.txt'
        else:
            if fragment == 0:
                fea_path = results_dir + 'cv_res_features[' + str(label[i]) + ']_' + str(out_format) + '.txt'
            else:
                fea_path = results_dir + 'cv_res_frag_features[' + str(label[i]) + ']_' + str(out_format) + '.txt'

        output_files.append(fea_path)

    return output_files


def out_dl_frag_file(label, results_dir, ind=False):
    output_files = []
    for i in range(len(label)):
        if ind is True:
            fea_path = results_dir + 'ind_dl_frag_features[' + str(label[i]) + ']_.txt'
        else:
            fea_path = results_dir + 'cv_dl_frag_features[' + str(label[i]) + ']_.txt'

        output_files.append(fea_path)

    return output_files


def opt_file_copy(source_files, results_dir):
    # adding exception handling
    target_files = []
    for source_file in source_files:
        dir_name, file_name = os.path.split(source_file)
        target_file = results_dir + 'opt_' + '_'.join(file_name.split('_')[-3:])

        target_files.append(target_file)
        try:
            shutil.copyfile(source_file, target_file)
        except IOError as e:
            print("Unable to copy file. %s\n" % e)
            return False

    for index, output_file in enumerate(target_files):
        out_with_full_path = os.path.abspath(output_file)
        if os.path.isfile(out_with_full_path):
            if index == 0:
                print('+----------------------------------------------------------------+')
                print('| The output files of optimal feature vectors can be found here: |')
                print('+----------------------------------------------------------------+')
            print(out_with_full_path)
    print('\n')

    return target_files


def read_res_seq_file(seq_file, category):
    if category == 'DNA':
        alphabet = DNA
    elif category == 'RNA':
        alphabet = RNA
    else:
        alphabet = PROTEIN

    seq_len_list = []  # 每一种标签序列的长度列表(list[])
    with open(seq_file, 'r') as in_f:
        seq_list = get_seqs(in_f, alphabet)
        for seq in seq_list:
            seq_len_list.append(len(seq))

    return seq_len_list


def read_res_label_file(label_file):

    res_labels_list = []
    label_len_list = []

    f = open(label_file, 'r')
    lines = f.readlines()
    for line in lines:
        if line[0] != '>':
            labels = line.strip().split()
            labels = list(map(int, labels))
            label_len_list.append(len(labels))
            res_labels_list.append(labels)
    f.close()

    return res_labels_list, label_len_list


def res_file_check(seq_len_list, label_len_list, fragment):
    count = 0
    # print(seq_len_list)
    # print(label_len_list)
    assert len(seq_len_list) == len(label_len_list), "The number of sequence should be equal to it's label!"

    for seq_len, label_len in zip(seq_len_list, label_len_list):
        if fragment == 0:
            assert seq_len == label_len, 'The length of sequence[' + str(count+1) + '] is not equal to corresponding ' \
                                                                                    'labels'
            assert label_len >= 5, 'The number of labels for sequence[' + str(count+1) + '] should not less than 5'
        else:
            assert label_len == 1, 'If -fragment is 1, each sequence should have only one label!'
