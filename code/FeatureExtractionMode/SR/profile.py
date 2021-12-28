import math
import subprocess
import threading
import os
import sys
import time
import pickle

import numpy as np
from itertools import product

from .acc import pdt
from ..utils.utils_pssm import sep_file, produce_all_frequency
from ..utils.utils_words import convert_tng_to_fasta


def pdt_profile(inputfile, n, lamada, sw_dir, process_num):
    """Generate PDT-Profile features.
    :param inputfile: input sequence file in FASTA format.
    :param n: the n most frequent amino acids in the amino acid frequency profiles.
    :param lamada: the distance between two amino acids.
     :param sw_dir: the main dir of software.
    :param process_num: the number of processes used for multiprocessing.
    """
    # tng_list, seq_name = top_n_gram(inputfile, n, process_num)
    dirname, seq_name = sep_file(inputfile)
    pssm_dir = produce_all_frequency(dirname, sw_dir, process_num)
    tng_fasta = convert_tng_to_fasta(pssm_dir, seq_name, inputfile, n, sw_dir)
    # convert_tng_to_fasta(pssm_dir, seq_name, input_file, n, sw_dir)
    return pdt(tng_fasta, lamada, sw_dir)


# -------------------------------------------------------------------------------------
# PDT-Profile end
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# ACC-PSSM, AC-PSSM, CC-PSSM start
# -------------------------------------------------------------------------------------

def blosum_pssm(seq_file, new_blosum_dict, dirname):
    """Generate pssm file using blosum62 matrix.
    :param seq_file: the sequence file containing one sequence.
    :param new_blosum_dict: the blosum62 dict after processing.
    :param dirname: the directory name for storing the generated files.
    """
    pssm_list = []
    with open(seq_file, 'r') as f:
        for line in f:
            if line.strip().startswith('>'):
                continue
            else:
                for index, i in enumerate(line.strip()):
                    blosum_list = list(map(str, new_blosum_dict[i]))
                    blosum_str = ' '.join(blosum_list)
                    line_str = ' '.join([str(index + 1), i, blosum_str])
                    pssm_list.append(line_str)

    blosum_dir = ''.join([dirname, '/blosum_pssm'])
    if not os.path.isdir(blosum_dir):
        os.mkdir(blosum_dir)
    seq_file_name = os.path.splitext(seq_file)[0]
    seq_file_name = os.path.split(seq_file_name)[1]
    blosum_file = ''.join([blosum_dir, '/', seq_file_name, '.pssm'])
    with open(blosum_file, 'w') as f:
        for i in range(3):
            f.write('\n')
        for line in pssm_list:
            f.write(line)
            f.write('\n')
    return os.path.abspath(blosum_file)


def read_blosum():
    """Read blosum dict and delete some keys and values."""
    full_path = os.path.realpath(__file__)
    file_path = os.path.dirname(full_path) + '/data/blosum62.pkl'
    with open(file_path, 'rb') as f:
        blosum_dict = pickle.load(f)

    blosum_dict.pop('*')
    blosum_dict.pop('B')
    blosum_dict.pop('Z')
    blosum_dict.pop('X')
    blosum_dict.pop('alphas')

    for key in blosum_dict:
        for i in range(4):
            blosum_dict[key].pop()
    return blosum_dict


def acc_pssm_cmd(pssm_file, lag, acc_out_file, sw_dir, sem):
    """ACC-PSSM command.
    :param pssm_file: the .pssm file.
    :param lag: the distance between two amino acids.
    :param acc_out_file: the output file of the acc program.
    :param sw_dir: the main dir of software.
    :param sem: 是用于控制进入数量的锁，控制同时进行的线程，内部是基于Condition来进行实现的
    """
    sem.acquire()

    if sys.platform.startswith('win'):
        acc_cmd = sw_dir + 'acc_pssm/acc.exe'
    else:
        acc_cmd = sw_dir + 'acc_pssm/acc'
        os.chmod(acc_cmd, 0o777)

    cmd = ' '.join([acc_cmd, ' ', str(lag), ' ', pssm_file, ' ', acc_out_file])
    subprocess.call(cmd, shell=True)
    time.sleep(2)
    sem.release()


def sep_acc_vector(acc_out_file):
    """Seperate acc_out_file and output the acc, ac, cc vectors.
    :param acc_out_file: the output file of the acc program.
    """
    acc_vec_list = []
    ac_vec_list = []
    cc_vec_list = []
    with open(acc_out_file, 'r') as f:
        for line in f:
            line = round(float(line.strip()), 3)
            # line = float(line.strip())
            acc_vec_list.append(line)

    for i in range(0, len(acc_vec_list), 400):
        ac_vec_list.extend(acc_vec_list[i:i + 20])
        cc_vec_list.extend(acc_vec_list[i + 20:i + 400])

    return acc_vec_list, ac_vec_list, cc_vec_list


def make_acc_pssm_vector(inputfile, lag, vec_type, sw_dir, process_num):
    """Generate ACC, AC, CC feature vectors.
    :param inputfile: input sequence file in FASTA format.
    :param lag: the distance between two amino acids.
    :param vec_type: the type of the vectors generated, ACC-PSSM, AC-PSSM
    or CC-PSSM.
    :param sw_dir: the main dir of software.
    :param process_num: the number of processes used for multiprocessing.
    """
    dirname, seq_name = sep_file(inputfile)
    pssm_dir = produce_all_frequency(dirname, sw_dir, process_num)
    # 调试模式 on/off
    # pssm_dir = "D:\\Leon\\bionlp\\BioSeq-NLP\\data\\cv_results\\Protein\\sequence\\SR\\SVM\\ACC-PSSM/all_seq_cv/pssm"

    dir_list = os.listdir(pssm_dir)
    index_list = []
    for elem in dir_list:
        pssm_full_path = ''.join([pssm_dir, '/', elem])
        name, suffix = os.path.splitext(elem)
        if os.path.isfile(pssm_full_path) and suffix == '.pssm':
            index_list.append(int(name))

    index_list.sort()
    new_blosum_dict = {}
    if len(index_list) != len(seq_name):
        new_blosum_dict = read_blosum()

    acc_out_fold = dirname + '/acc_out'
    acc_vectors = []
    ac_vectors = []
    cc_vectors = []

    if not os.path.isdir(acc_out_fold):
        os.mkdir(acc_out_fold)

    out_file_list = []
    threads = []
    sem = threading.Semaphore(process_num)

    for i in range(1, len(seq_name) + 1):
        if i in index_list:
            pssm_full_path = ''.join([pssm_dir, '/', str(i), '.pssm'])
        else:
            seq_file = ''.join([dirname, '/', str(i), '.txt'])
            pssm_full_path = blosum_pssm(seq_file, new_blosum_dict, dirname)
        acc_out_file = ''.join([acc_out_fold, '/', str(i), '.out'])
        out_file_list.append(acc_out_file)
        # acc_pssm_cmd(pssm_full_path, lag, acc_out_file, sw_dir)
        threads.append(threading.Thread(target=acc_pssm_cmd,
                                        args=(pssm_full_path, lag, acc_out_file, sw_dir, sem)))
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    for out_file in out_file_list:
        acc_vec_list, ac_vec_list, cc_vec_list = sep_acc_vector(out_file)
        acc_vectors.append(acc_vec_list)
        ac_vectors.append(ac_vec_list)
        cc_vectors.append(cc_vec_list)
    if vec_type == 'acc':
        return np.array(acc_vectors)
    elif vec_type == 'ac':
        return np.array(ac_vectors)
    elif vec_type == 'cc':
        return np.array(cc_vectors)
    else:
        return False


# -------------------------------------------------------------------------------------
# ACC-PSSM, AC-PSSM, CC-PSSM end
# -------------------------------------------------------------------------------------

def initialization():
    blosum62 = {}

    full_path = os.path.realpath(__file__)
    blosum62_path = os.path.dirname(full_path) + '/data/blosum62'

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
    return blosum62


# PSSM RT starts
def format_each_line(each_line):
    col = each_line[5:8].strip() + '\t'
    col += ('\t'.join(each_line[9:].strip().split()[:20]) + '\n')
    return col


def simplify_pssm(pssm_file, new_file):
    if os.path.exists(pssm_file):
        with open(pssm_file) as input_pssm, open(new_file, 'w') as outfile:
            count = 0
            for each_line in input_pssm:
                count += 1
                if count <= 2:
                    continue
                if not len(each_line.strip()):
                    break
                one_line = format_each_line(each_line)
                if count == 3:
                    one_line = ' ' + one_line
                outfile.write(one_line)


def pssm_ksb(input_file, sw_dir, process_num, is_pssm_dt=False):
    dirname, seq_name = sep_file(input_file)
    pssm_dir = produce_all_frequency(dirname, sw_dir, process_num)

    dir_name = os.path.split(pssm_dir)[0]

    xml_dir = dir_name + '/xml'

    final_result = ''.join([dir_name, '/final_result'])
    if not os.path.isdir(final_result):
        os.mkdir(final_result)

    dir_list = os.listdir(xml_dir)
    index_list = []
    for elem in dir_list:
        xml_full_path = ''.join([xml_dir, '/', elem])
        name, suffix = os.path.splitext(elem)
        if os.path.isfile(xml_full_path) and suffix == '.xml':
            index_list.append(int(name))

    index_list.sort()

    pssm_pro_files = []
    seq_names = []
    vectors = []

    for index in index_list:
        pssm_file = pssm_dir + '/' + str(index) + '.pssm'
        pssm_file_list = list(os.path.splitext(pssm_file))
        pssm_process_file = pssm_file_list[0] + '_pro' + pssm_file_list[1]
        seq_name = pssm_file_list[0].split('/')[-1]
        seq_names.append(seq_name)
        pssm_pro_files.append(pssm_process_file)
        simplify_pssm(pssm_file, pssm_process_file)

        if is_pssm_dt:
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

            vector = generate_ksb_pssm(pssm, 4)
            vectors.append(list(vector))

    if is_pssm_dt:
        return np.array(vectors)
    return pssm_pro_files, seq_names


def get_blosum62(protein):
    blosum62 = initialization()
    pssm_score = []
    for aa in protein:
        aa = aa.upper()
        pssm_score.append(blosum62[aa])
    return pssm_score


def pssm_rt_method(input_file, process_num, sw_dir, fixed_len):
    pssm_files, seq_names = pssm_ksb(input_file, sw_dir, process_num)
    #  pssm_ksb(input_file, sw_dir, process_num, is_pssm_dt=False):
    vectors = []
    for pssm_file, seq_name in zip(pssm_files, seq_names):
        pssm_score = []
        if os.path.exists(pssm_file):
            with open(pssm_file) as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = [int(x) for x in line.strip().split('\t')[1:]]
                    pssm_score.append(line)
        else:
            p1 = os.path.split(pssm_file)
            seq_path = os.path.split(p1[0])[0] + '/' + seq_name + '.txt'
            with open(seq_path) as f:
                lines = f.readlines()
                pssm_score = get_blosum62(lines[1].strip())
        vectors.append(get_rt_vector(pssm_score, fixed_len))
    return np.array(vectors)


def get_rt_vector(pssm_score, fixed_len):
    pssm_score = normalized(pssm_score)
    temp_len = len(pssm_score)
    fixed_pssm_score = []
    for i in range(fixed_len):
        if i < temp_len:
            fixed_pssm_score.append(pssm_score[i])
        else:
            fixed_pssm_score.append([0.0]*20)
    residue = cal_residue_conservation(fixed_pssm_score)
    pair = cal_pair_relationships(fixed_pssm_score)
    multi = cal_multi_relationships(fixed_pssm_score)
    vector = residue + list(pair) + multi

    # print('len of pssm_score ', len(pssm_score))
    # print('len of pssm_score[0] ', len(pssm_score[0]))
    # print('len of fixed_pssm_score ', len(fixed_pssm_score))
    # print('len of residue ', len(residue))
    # print('len of pair ', len(list(pair)))
    # print('len of multi ', len(multi))
    # print('len of vector ', len(vector))
    return vector


def normalized(pssm_score):
    nor_pssm_score = []
    for i in pssm_score:
        nor_pssm_score.append([1 / (1 + math.e ** x) for x in i])  # 按行正则化
    return nor_pssm_score


def cal_residue_conservation(pssm_score):
    residue_conservation = []
    for i in pssm_score:
        residue_conservation += i
    return residue_conservation


def cal_pair_relationships(pssm_score):
    target_position = int(len(pssm_score) / 2)
    pair_scores = []
    if len(pssm_score) % 2 == 1:
        target_position += 1
    for i in range(len(pssm_score)):
        if i != target_position:
            pair = product(pssm_score[i], pssm_score[target_position])
            pair_scores.append([x[0] * x[1] for x in pair])
    pair_scores = [np.array(x) for x in pair_scores]
    pair_relationships = pair_scores[0]
    for i in pair_scores:
        pair_relationships += i
    return pair_relationships.tolist()


def cal_multi_relationships(pssm_score):
    multi_relationship_left = []
    multi_relationship_right = []
    target_position = int(len(pssm_score) / 2)
    if len(pssm_score) % 2 == 1:
        target_position += 1
    pssm_score = [np.array(x) for x in pssm_score]
    pssm_score = np.array(pssm_score)
    for i in range(5):
        left = pssm_score[:target_position - 1, i].tolist()
        multi_relationship_left.append(sum(left))
        right = pssm_score[target_position:, i].tolist()
        multi_relationship_right.append(sum(right))
    multi_relationship = multi_relationship_left
    multi_relationship.extend(multi_relationship_right)
    return multi_relationship


# PSSM RT ends

# PSSM DT starts
def read_pssm(pssm_file):
    if os.path.exists(pssm_file):
        with open(pssm_file, 'r') as f:
            lines = f.readlines()
            pssm_arr = []
            count = 0
            for line in lines:
                if count == 0:
                    count += 1
                    pass
                else:
                    line = line.strip().split()
                    pssm_arr.append(line)
            pssm = np.array(pssm_arr)   # 一个文件对应一个n*20的矩阵
            return pssm
    else:
        return False


def create_matrix(row_size, column_size):
    Matrix = np.zeros((row_size, column_size))
    return Matrix


def aver(matrix_sum, seq_len):
    matrix_array = np.array(matrix_sum)
    matrix_array = np.divide(matrix_array, seq_len)
    matrix_array_shp = np.shape(matrix_array)
    matrix_average = [(np.reshape(matrix_array, (matrix_array_shp[0] * matrix_array_shp[1],)))]
    return matrix_average


def generate_ksb_pssm(pssm, lag):
    seq_cn = float(np.shape(pssm)[0])
    vector = []
    for i in range(1, lag + 1):
        matrix_final = pre_handle_columns(pssm, i)
        ksb_vector = aver(matrix_final, float(seq_cn - i))
        vector += list(ksb_vector[0])
    return vector


def pre_handle_columns(pssm, step):
    pssm = pssm[:, 1:21]
    pssm = pssm.astype(float)
    matrix_final = [[0.0] * 20] * 20
    matrix_final = np.array(matrix_final)
    seq_cn = np.shape(pssm)[0]
    for i in range(20):
        for j in range(20):
            for k in range(seq_cn - step):
                matrix_final[i][j] += (pssm[k][i] * pssm[k + step][j])
    return matrix_final


def pssm_dt_method(input_file, process_num, sw_dir):
    return pssm_ksb(input_file, sw_dir, process_num, True)
