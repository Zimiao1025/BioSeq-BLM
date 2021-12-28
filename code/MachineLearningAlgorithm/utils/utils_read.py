import os
import numpy as np


class FormatRead(object):
    def __init__(self, filename, in_format):
        self.in_file = filename  # 单个文件的话需要先将字符串转换为列表
        self.in_format = in_format

    def read_csv(self):
        """transform csv format file to vector arrays"""
        vectors = []
        f = open(self.in_file, 'r')
        lines = f.readlines()
        for line in lines:
            vector = line.strip().split(',')
            vector = list(map(float, vector))
            vectors.append(vector)
        f.close()

        return np.array(vectors, dtype=np.float32)

    def read_tsv(self):
        """transform tsv format file to vector arrays"""
        vectors = []
        f = open(self.in_file, 'r')
        lines = f.readlines()
        for line in lines:
            vector = line.strip().split('\t')
            vector = list(map(float, vector))
            vectors.append(vector)
        f.close()

        return np.array(vectors, dtype=np.float32)

    def read_svm(self):
        """transform svm format file to vector arrays"""
        vectors = []
        f = open(self.in_file, 'r')
        lines = f.readlines()
        for line in lines:
            vector = []
            for i in range(1, len(line.split())):
                temp = line.split()[i]
                temp_vec = temp.split(':')[1]
                vector.append(float(temp_vec))
            vectors += vector
        f.close()

        return np.array(vectors, dtype=np.float32)

    def read_tab(self):
        """transform tab format file to vector arrays"""
        lens = 'flag'
        vectors = []
        f = open(self.in_file, 'r')
        lines = f.readlines()
        for line in lines:
            if ':' in line or ',' in line:
                print('The format of the input file should be tab format.')
                return False
            else:
                lst = line.strip().split()
                tmp = len(lst)
                if lens == 'flag':
                    lens = tmp
                elif tmp != lens:
                    print('The lengths of the feature vectors are not same. Please check.')
                    return False
                lst = list(map(float, lst))
                vectors.append(lst)
        f.close()

        return np.array(vectors, dtype=np.float32)

    def write_to_file(self):
        if self.in_format == 'svm':
            return self.read_svm()
        elif self.in_format == 'tab':
            return self.read_tab()
        elif self.in_format == 'csv':
            return self.read_csv()
        elif self.in_format == 'tsv':
            return self.read_tsv()
        else:
            print('Output file format error! Please check.')
            return False


def files2vectors_seq(file_list, in_format):
    in_files = []
    for in_file_name in file_list:
        in_file_path = os.path.abspath(in_file_name)
        assert os.path.isfile(in_file_path), 'The feature vector file: ' + in_file_path + ' is not exist!'
        in_files.append(in_file_path)

    vectors = None
    for i, in_file in enumerate(in_files):
        temp_vec = FormatRead(in_file, in_format).write_to_file()
        if vectors is None:
            vectors = temp_vec
        else:
            vectors = np.vstack((vectors, temp_vec))

    return vectors


def files2vectors_info(file_list, in_format):
    in_files = []
    for in_file_name in file_list:
        in_file_path = os.path.abspath(in_file_name)
        assert os.path.isfile(in_file_path), 'The feature vector file: ' + in_file_path + ' is not exist!'
        in_files.append(in_file_path)

    vec_num_list = []
    vectors = None
    for i, in_file in enumerate(in_files):
        temp_vec = FormatRead(in_file, in_format).write_to_file()
        if vectors is None:
            vectors = temp_vec
        else:
            vectors = np.vstack((vectors, temp_vec))
        vec_num = len(temp_vec)
        vec_num_list.append(vec_num)

    return vectors, vec_num_list, in_files


def files2vectors_res(file_list, in_format):
    in_files = []
    for in_file_name in file_list:
        in_file_path = os.path.abspath(in_file_name)
        assert os.path.isfile(in_file_path), 'The feature vector file: ' + in_file_path + ' is not exist!'
        in_files.append(in_file_path)

    vec_num_list = []
    vectors = None
    for i, in_file in enumerate(in_files):
        temp_vec = FormatRead(in_file, in_format).write_to_file()
        if vectors is None:
            vectors = temp_vec
        else:
            vectors = np.vstack((vectors, temp_vec))
        vec_num = len(temp_vec)
        vec_num_list.append(vec_num)

    return vectors, vec_num_list


def read_dl_vec4seq(fixed_len, in_files, return_sp):
    vectors_list = []
    seq_len_list = []
    sp_num_list = []
    # print(in_files)
    for in_file in in_files:
        count = 0
        f = open(in_file, 'r')
        lines = f.readlines()
        vectors = []
        flag = 0

        for line in lines:
            if len(line.strip()) != 0:
                if line[0] != '>':
                    vector = line.strip().split('\t')
                    vector = list(map(float, vector))
                    vectors.append(vector)
                    flag = 1
                else:
                    if flag == 1:
                        seq_len_list.append(len(vectors))
                        vectors_list.append(np.array(vectors))
                        vectors = []
                        count += 1
                        flag = 0

        f.close()
        sp_num_list.append(count)

    # print(len(vectors_list))
    vec_mat, fixed_seq_len_list = fixed_opt(fixed_len, vectors_list, seq_len_list)
    if return_sp is True:
        return vec_mat, sp_num_list, fixed_seq_len_list
    else:
        return vec_mat, fixed_seq_len_list


def read_base_mat4res(in_file, fixed_len):
    vectors_list = []
    seq_len_list = []
    # print(in_file)
    # exit()
    f = open(in_file, 'r')
    lines = f.readlines()
    vectors = []
    flag = 0
    for line in lines:
        if len(line.strip()) != 0:
            if line[0] != '>':
                vector = line.strip().split('\t')
                vector = list(map(float, vector))
                vectors.append(vector)
                flag = 1
            else:
                if flag == 1:
                    seq_len_list.append(len(vectors))
                    vectors_list.append(np.array(vectors))
                    vectors = []
                    flag = 0
    f.close()

    # print(vectors_list[0][0])
    # exit()
    vec_mat, fixed_seq_len_list = fixed_opt(fixed_len, vectors_list, seq_len_list)

    return vec_mat, fixed_seq_len_list


def read_base_vec_list4res(in_file):
    vectors_list = []
    f = open(in_file, 'r')
    lines = f.readlines()
    vectors = []
    flag = 0
    for line in lines:
        if len(line.strip()) != 0:
            if line[0] != '>':
                vector = line.strip().split('\t')
                vector = list(map(float, vector))
                vectors.append(vector)
                flag = 1
            else:
                if flag == 1:
                    vectors_list.append(np.array(vectors))
                    vectors = []
                    flag = 0
    f.close()

    return vectors_list


def fixed_opt(fixed_len, vectors_list, seq_len_list):
    vec_mat = []

    for i in range(len(vectors_list)):
        # print(i)
        temp_arr = np.zeros((fixed_len, len(vectors_list[i][0])))
        seq_len = len(vectors_list[i])
        if seq_len > fixed_len:
            seq_len_list[i] = fixed_len
        temp_len = min(seq_len, fixed_len)
        temp_arr[:temp_len, :] = vectors_list[i][:temp_len, :]
        vec_mat.append(temp_arr)

    print(np.array(vec_mat).shape)
    return np.array(vec_mat), seq_len_list


def seq_label_read(vec_num_list, label_list):
    labels = []
    for i in range(len(label_list)):
        labels += [label_list[i]] * vec_num_list[i]
    return np.array(labels)


def res_label_read(vec_num_list, label_list):
    labels = []
    for i in range(len(label_list)):
        labels += [label_list[i]] * vec_num_list[i]
    return np.array(labels)


def res_dl_label_read(res_label_list, fixed_len):
    res_label_mat = []
    for res_label in res_label_list:
        temp_arr = np.zeros(fixed_len)
        seq_len = len(res_label)
        temp_len = min(seq_len, fixed_len)
        temp_arr[:temp_len] = res_label[:temp_len]
        res_label_mat.append(temp_arr)

    return np.array(res_label_mat)
