import numpy as np
from ..utils.utils_words import make_km_list
from ..utils.utils_const import aaList, aaList_DNA, aaList_RNA, aaList_sixbits, aaList_five, aaList_AESNN3, aaList_ncp


class ResidueComposition2Vectors(object):
    # TODO: 在模型的初始化函数中定义模型要用到的变量
    def __init__(self, alphabet):
        """
        Initialize the object.
        :param alphabet: DNA, RNA or Protein
        """
        if alphabet == 'DNA':
            self.size = 4
            self.alphabet_list = aaList_DNA
        elif alphabet == 'RNA':
            self.size = 4
            self.alphabet_list = aaList_RNA
        else:
            self.size = 20
            self.alphabet_list = aaList

        self.aaList_Index = []
        self.vec_mat_list = []

    def one_hot(self, input_file):
        with open(input_file) as r:
            for line in r:
                if line[0] == '>':
                    continue
                else:
                    index_list = []
                    line = line.strip().upper()
                    for k in range(len(line)):
                        index_list.append(str(self.alphabet_list.index(line[k])))
                    self.aaList_Index.append(index_list)

        for i in range(len(self.aaList_Index)):

            temp_length = len(self.aaList_Index[i])  # 小于或等于fixed_len的长度值
            vec_mat = np.zeros((temp_length, self.size))
            for j in range(temp_length):
                vector = [0] * self.size
                vector[int(self.aaList_Index[i][j])] = 1
                vector = list(map(float, vector))
                vec_mat[j] = vector
            self.vec_mat_list.append(vec_mat)
        return self.vec_mat_list

    def position_specific(self, k, input_file):
        kms = make_km_list(k, self.alphabet_list)
        with open(input_file) as f:
            for line in f:
                if line[0] == '>':
                    continue
                else:
                    line = line.upper().strip()
                    vec_mat = []
                    length = len(line)
                    for s in range(length-k+1):
                        seq = line[s:(s + k)]
                        index = kms.index(seq)
                        fe = [0] * len(kms)
                        fe[index] = 1
                        vec_mat.append(list(map(float, fe)))

                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def one_hot_six_bits(self, input_file):
        index_list = []
        six_bits_alphabet_list = aaList_sixbits
        with open(input_file) as r:
            for line in r:
                if line[0] == '>':
                    continue
                else:
                    index = []
                    line = line.strip()
                    length = len(line)
                    for k in range(length):
                        for i in range(len(six_bits_alphabet_list)):
                            if line[k] in six_bits_alphabet_list[i]:
                                index.append(str(i))
                    index_list.append(index)
        for i in range(len(index_list)):
            vec_mat = []
            temp_len = len(index_list[i])
            for j in range(temp_len):
                vector = [0] * 6
                vector[int(index_list[i][j])] = 1
                vector = list(map(float, vector))

                vec_mat.append(vector)
            self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def one_hot_five(self, input_file):
        index_list = []
        alphabet_list = aaList
        five_alphabet_list = aaList_five
        with open(input_file) as r:
            for line in r:
                if line[0] == '>':
                    continue
                else:
                    index = []
                    line = line.strip()
                    length = len(line)
                    for k in range(length):
                        index.append(str(alphabet_list.index(line[k])))
                    index_list.append(index)
        for i in range(len(index_list)):
            vec_mat = []
            temp_len = len(index_list[i])
            for j in range(temp_len):
                vector = five_alphabet_list[int(index_list[i][j])]
                vector = list(map(float, vector))
                vec_mat.append(vector)
            self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def aesnn3(self, input_file):
        # Just for Protein
        encoding_schemes = aaList_AESNN3
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line[0] != '>':
                    vec_mat = []
                    temp_len = len(line)
                    for i in range(temp_len):
                        vector = encoding_schemes[line[i]]
                        vector = list(map(float, vector))
                        vec_mat.append(vector)

                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def dbe(self, input_file):
        kms = make_km_list(2, self.alphabet_list)
        with open(input_file) as f:
            for line in f:
                if line[0] == '>':
                    continue
                else:
                    line = line.upper().strip()
                    vec_mat = []
                    temp_len = len(line)

                    for i in range(temp_len-1):
                        seq = line[i:(i + 2)]
                        index = kms.index(seq)
                        fe = list(map(float, bin(index)[2:].zfill(4)))  # for index=15, fe=[1, 1, 1, 1]
                        vec_mat.append(list(map(float, fe)))
                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def ncp(self, input_file):
        # Just for RNA
        encoding_schemes = aaList_ncp
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line[0] != '>':
                    vec_mat = []
                    temp_len = len(line)

                    for i in range(temp_len):
                        vector = encoding_schemes[line[i]]
                        vector = list(map(float, vector))
                        vec_mat.append(vector)
                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list
