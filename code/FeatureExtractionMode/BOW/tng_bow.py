import os
import numpy as np
from ..utils.utils_words import make_km_list
from ..utils.utils_const import PROTEIN
from ..utils.utils_pssm import sep_file, produce_all_frequency
from ..utils.utils_words import produce_top_n_gram


def tng_bow(input_file, n, cur_dir, process_num):
    """Generate top-n-gram list.
    :param input_file: input sequence file.
    :param n: the n most frequent amino acids in the amino acid frequency profiles.
    :param cur_dir: the main dir of code.
    :param process_num: the number of processes used for multiprocessing.
    """
    pssm_path, seq_name = sep_file(input_file)
    sw_dir = cur_dir + '/software/'
    pssm_dir = produce_all_frequency(pssm_path, sw_dir, process_num)
    print('pssm_dir: ', pssm_dir)
    # 调试模式 on/off
    # pssm_dir = cur_dir + "/data/results/Protein/sequence/OHE/SVM/PSSM/all_seq/pssm"

    dir_name = os.path.split(pssm_dir)[0]
    fasta_name = os.path.split(dir_name)[1]

    final_result = ''.join([dir_name, '/final_result'])
    print('final_result: ', final_result)

    if not os.path.isdir(final_result):
        os.mkdir(final_result)

    tng_file_name = ''.join([final_result, '/', fasta_name, '_new.txt'])
    with open(tng_file_name, 'w') as f:
        for index, tng in enumerate(produce_top_n_gram(pssm_dir, seq_name, n, sw_dir)):
            f.write('>')
            f.write(seq_name[index])
            f.write('\n')
            for elem in tng:
                f.write(elem)
                f.write(' ')
            f.write('\n')

    gram_list = make_km_list(n, PROTEIN)
    vector_list = []
    for tng in produce_top_n_gram(pssm_dir, seq_name, n, sw_dir):
        vec_len = len(tng)
        # print vec_len
        vector = []
        for elem in gram_list:
            gram_count = tng.count(elem)
            occur_freq = round((gram_count * 1.0) / vec_len, 4)
            vector.append(occur_freq)
        vector_list.append(vector)

    return np.array(vector_list)
