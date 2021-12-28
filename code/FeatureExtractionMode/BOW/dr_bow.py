import numpy as np
from ..utils.utils_words import make_km_list
from ..utils.utils_fasta import get_seqs
from ..utils.utils_const import PROTEIN


def dr_bow(input_file, max_dis):
    """
    The Distance Residue method.
    :param input_file: the input sequence file.
    :param max_dis: the value of the maximum distance.
    """
    assert int(max_dis) > 0
    aa_pairs = make_km_list(2, PROTEIN)
    aa_list = list(PROTEIN)
    vector_list = []
    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, alphabet=PROTEIN)

    for line in seq_list:
        vector = []
        len_line = len(line)
        for i in range(max_dis + 1):
            if i == 0:
                temp = [line.count(j) for j in aa_list]
                vector.extend(temp)
            else:
                new_line = []
                for index, elem in enumerate(line):
                    if (index + i) < len_line:
                        new_line.append(line[index] + line[index + i])
                temp = [new_line.count(j) for j in aa_pairs]
                vector.extend(temp)
        vector_list.append(vector)
        
    return np.array(vector_list)
