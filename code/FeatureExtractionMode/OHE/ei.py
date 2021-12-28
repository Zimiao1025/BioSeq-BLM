import os

import numpy as np

from ..SR.profile import simplify_pssm, read_pssm, get_blosum62
from ..utils.utils_const import aaList_DNA, aaList, DBM_List
from ..utils.utils_pssm import produce_all_frequency, sep_file, generate_pssm
from ..utils.utils_psfm import km2index, sep_file_psfm, run_group_search, profile_worker


class EvolutionaryInformation2Vectors(object):
    # TODO: 在模型的初始化函数中定义模型要用到的变量
    def __init__(self, alphabet, cur_dir=None):
        """
        Initialize the object.
        :param alphabet: DNA, RNA or Protein
        """

        if alphabet == 'DNA':
            self.size = 4
            self.alphabet_list = aaList_DNA
        elif alphabet == 'RNA':
            print('Evolutionary information class method is not adapt for RNA!')
            exit()
        else:
            self.size = 20
            self.alphabet_list = aaList

        self.cur_dir = cur_dir
        full_path = os.path.realpath(__file__)
        self.ei_dir = os.path.dirname(full_path) + '/data/'
        self.sw_dir = cur_dir + '/software/'
        self.vec_mat_list = []

    def blast_matrix(self, input_file):
        dbm_dict = DBM_List
        with open(input_file) as f:
            for line in f:
                line = line.strip().upper()
                if line[0] == '>':
                    continue
                else:
                    vec_mat = []
                    temp_len = len(line)
                    for i in range(temp_len):
                        vector = dbm_dict[line[i]]
                        vec_mat.append(vector)
                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def pam250(self, file_path):
        pam250 = {}
        pam250_path = self.ei_dir + 'PAM250.txt'
        pam250_reader = open(pam250_path)
        count = 0
        # read the matrix of pam250
        for line in pam250_reader:
            count += 1
            if count <= 1:
                continue
            line = line.strip('\r').split()
            # print(line)
            if line[0] != '*':
                pam250[line[0]] = [float(x) for x in line[1:21]]
        # print(pam250)
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line[0] == '>':
                    continue
                else:
                    vec_mat = []
                    temp_len = len(line)
                    for i in range(temp_len):
                        vec_mat.append(pam250[line[i]])
                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def blosum62(self, file_path):
        blosum62 = {}
        blosum62_path = self.ei_dir + 'blosum62'
        blosum_reader = open(blosum62_path)
        count = 0
        # read the matrix of blosum62
        for line in blosum_reader:
            count += 1
            if count <= 7:
                continue
            line = line.strip('\r').split()
            if line[0] != '*':
                blosum62[line[0]] = [float(x) for x in line[1:21]]
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line[0] == '>':
                    continue
                else:
                    vec_mat = []
                    temp_len = len(line)
                    for i in range(temp_len):
                        vec_mat.append(blosum62[line[i]])
                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def pssm(self, input_file, process_num):
        pssm_path, seq_name = sep_file(input_file)
        # pssm_path: # D:\Leon\bionlp\BioSeq-NLP/data/results/Protein/sequence/OHE/SVM/PSSM/all_seq

        pssm_dir = produce_all_frequency(pssm_path, self.sw_dir, process_num)
        # 调试模式 on/off
        # pssm_dir = self.cur_dir + "/results/all_seq_cv/pssm"

        # print('pssm_dir: ', pssm_dir)
        # pssm_dir: D:\Leon\bionlp\BioSeq-NLP\data\results\Protein\sequence\OHE\SVM\PSSM\all_seq/pssm

        dir_name = os.path.split(pssm_dir)[0]

        xml_dir = dir_name + '/xml'
        # print('xml_dir: ', xml_dir)
        # xml_dir: D:\Leon\bionlp\BioSeq-NLP\data\results\Protein\sequence\OHE\SVM\PSSM\all_seq/xml

        final_result = ''.join([dir_name, '/final_result'])
        # print('final_result: ', final_result)
        # final_result:  D:\Leon\bionlp\BioSeq-NLP\data\results\Protein\sequence\OHE\SVM\PSSM\all_seq/final_result
        if not os.path.isdir(final_result):
            os.mkdir(final_result)

        dir_list = os.listdir(xml_dir)
        # print('dir_list: ', dir_list)
        # dir_list:  ['1.xml', '10.xml', '11.xml', '12.xml', '13.xml', '14.xml', '15.xml', '16.xml', '17.xml',
        # '18.xml', '19.xml', '2.xml', '20.xml', '3.xml', '4.xml', '5.xml', '6.xml', '7.xml', '8.xml', '9.xml']

        index_list = []
        for elem in dir_list:
            xml_full_path = ''.join([xml_dir, '/', elem])
            # print("xml_full_path: ", xml_full_path)
            # xml_full_path:  D:\Leon\bionlp\BioSeq-NLP\data\results\Protein\sequence\OHE\SVM\PSSM\all_seq/xml/1.xml
            name, suffix = os.path.splitext(elem)
            # print("suffix: ", suffix)
            # suffix:  .xml
            if os.path.isfile(xml_full_path) and suffix == '.xml':
                index_list.append(int(name))

        index_list.sort()
        # print('index_list:', index_list)
        # index_list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        pssm_pro_files = []
        seq_names = []

        for index in index_list:
            pssm_file = pssm_dir + '/' + str(index) + '.pssm'
            pssm_file_list = list(os.path.splitext(pssm_file))
            pssm_process_file = pssm_file_list[0] + '_pro' + pssm_file_list[1]
            seq_name = pssm_file_list[0].split('/')[-1]
            seq_names.append(seq_name)
            pssm_pro_files.append(pssm_process_file)
            simplify_pssm(pssm_file, pssm_process_file)
            # pssm_process_file为除去前三行后八行且每行只包含字母后的前20个数字

            pssm = read_pssm(pssm_process_file)

            if pssm is False:
                p1 = os.path.split(pssm_process_file)
                seq_path = os.path.split(p1[0])[0] + '/' + seq_name + '.txt'
                with open(seq_path) as f:
                    lines = f.readlines()
                    protein_seq = lines[1].strip().upper()
                    pssm = get_blosum62(protein_seq)
                    pssm = np.array(pssm)
                    protein_seq = [np.array([x]) for x in list(protein_seq)]
                    protein_seq = np.array(protein_seq)
                    pssm = np.hstack((protein_seq, pssm))

            temp_vec = generate_pssm(pssm)
            self.vec_mat_list.append(temp_vec)

        return self.vec_mat_list

    def psfm(self, file_path, process_num):
        k = 1
        km_index = km2index(self.alphabet_list, k)

        headers = sorted(iter(km_index.items()), key=lambda d: d[1])
        # print("km_index: ", km_index)
        # print("headers: ", headers)
        # km_index:  {'A': 0, 'C': 1, ..., 'W': 18, 'Y': 19}
        # headers:  [('A', 0), ('C', 1), ..., ('W', 18), ('Y', 19)]
        profile_home = os.path.split(file_path)[0] + '/' + str(os.path.split(file_path)[1].split('.')[0])
        # print('profile_home', profile_home)
        # profile_home D:\Leon\bionlp\BioSeq-NLP/data/results/Protein/sequence/OHE/SVM/PSFM/all_seq

        if not os.path.exists(profile_home):
            try:
                os.makedirs(profile_home)
            except OSError:
                pass
        seq_dir, seq_name = sep_file_psfm(profile_home, file_path)
        # print('seq_dir', seq_dir)
        # print('seq_name', seq_name)
        # seq_dir D:\Leon\bionlp\BioSeq-NLP\data\results\Protein\sequence\OHE\SVM\PSFM\all_seq\all_seq
        # seq_name ['1AKHA\t|1~1', '1AOII\t|1~2', '1B6WA\t|1~3', ...]

        profile_home = seq_dir

        seq_dir = os.listdir(seq_dir)
        seq_dir.sort()
        pssm_dir = profile_home + '/pssm'
        if not os.path.isdir(pssm_dir):
            try:
                os.makedirs(pssm_dir)
            except OSError:
                pass
        xml_dir = profile_home + '/xml'
        if not os.path.isdir(xml_dir):
            try:
                os.makedirs(xml_dir)
            except OSError:
                pass
        msa_dir = profile_home + '/msa'
        if not os.path.isdir(msa_dir):
            try:
                os.makedirs(msa_dir)
            except OSError:
                pass
        psfm_dir = profile_home + '/psfm'
        if not os.path.isdir(psfm_dir):
            try:
                os.makedirs(psfm_dir)
            except OSError:
                pass

        index_list = []
        for elem in seq_dir:
            name, suffix = os.path.splitext(elem)
            index_list.append(int(name))
        index_list.sort()
        # print('index_list:', index_list)
        # index_list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # exit()

        run_group_search(index_list, profile_home, self.sw_dir, process_num)  # 并行计算msa
        for i in range(0, len(index_list)):
            seq = str(index_list[i]) + '.txt'
            out_file = profile_worker(seq, self.alphabet_list, k, profile_home, headers)
            psfm_mat = read_pssm(out_file)

            temp_vec = generate_pssm(psfm_mat)
            self.vec_mat_list.append(temp_vec)

        return self.vec_mat_list
