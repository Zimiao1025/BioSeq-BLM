import os
import re
import subprocess
import sys
import numpy as np

from .pse import get_phyche_list, get_extra_index, get_phyche_value, get_aaindex, extend_aaindex
from ..utils.utils_const import DNA, RNA, PROTEIN
from ..utils.utils_fasta import get_seqs
from ..utils.utils_words import seq_length_fixed


def get_values(prop, sup_info):
    values = ""
    name = re.search(prop, sup_info)
    if name:
        str_prop = prop + "\s*\,(.+)"
        b = re.search(str_prop, sup_info)
        if b:
            values = b.group(1)
    return values


def sep_sequence(seq, k):
    i = k - 1
    sub_seqs = []
    while i < len(seq):
        j = 0
        nuc = ''
        while j < k:
            nuc = seq[i - j] + nuc
            j = j + 1
        sub_seqs.append(nuc)
        i += 1
    return sub_seqs


def get_specific_value(olinuc, olinucs, prop, sup_info):
    olinucs = olinucs.strip().split(",")
    values = get_values(prop, sup_info).rstrip()
    values = values.strip().split(",")

    count = olinucs.index(olinuc)
    value = values[count]
    return float(value)


def ave_p(seq, olinucs, length, k, prop, sup_info):
    limit = length - k + 1
    i = 1
    s = 0
    while i < limit or i == limit:
        value = get_specific_value(seq[i - 1], olinucs, prop, sup_info)
        s = s + value
        i = i + 1
    s = s / limit
    return s


#  geary
# --------------------------------------
# inputs: seq = string, length = int, k = int, lamada = int, prop = string,
#         SupFileName = string
# output: final = int
def geary(seq, olinucs, length, k, lamada, prop, sup_info):
    lim = length - k + 1
    limit = length - k - lamada + 1
    b = 1
    sqr = 0
    while b < limit or b == limit:
        cur_value = get_specific_value(seq[b - 1], olinucs, prop, sup_info)

        next_value = get_specific_value(seq[b + lamada - 1], olinucs, prop, sup_info)

        sqr = sqr + ((cur_value - next_value) * (cur_value - next_value))
        b = b + 1
    top = sqr * lim
    limit2 = (length - k - lamada + 1)
    c = 1
    sqr2 = 0
    while c < limit2 or c == limit2:
        current = get_specific_value(seq[c - 1], olinucs, prop, sup_info)
        avg = ave_p(seq, olinucs, length, k, prop, sup_info)
        sqr2 = sqr2 + (current - avg) * (current - avg)
        c = c + 1
    bottom = sqr2 * limit * 2
    final = float((top / bottom) * 1000) / 1000.0
    return final


#  Moreau
# -------------------------------------
# inputs: seq = string, length = int, k = int, lamada = int, prop = string,
#         supFileName = string
# output: final = int

def moreau(seq, olinucs, length, k, lamada, prop, sup_info):
    limit = length - k - lamada + 1
    d = 1
    prod = 0
    while d < limit or d == limit:
        cur_value = get_specific_value(seq[d - 1], olinucs, prop, sup_info)

        next_value = get_specific_value(seq[d + lamada - 1], olinucs, prop, sup_info)

        prod = prod + (cur_value * next_value)
        d = d + 1
    final = prod / limit
    return final


#  moran
# --------------------------------------
# inputs: seq = string, length = int, k = int, lamada = int, prop = string,
#         SupFileName = string
# output: final = int

def moran(seq, olinucs, length, k, lamada, prop, sup_info):
    limit = length - k - lamada + 1
    j = 1
    top = 0
    avg = ave_p(seq, olinucs, length, k, prop, sup_info)
    while j < limit or j == limit:
        cur_value = get_specific_value(seq[j - 1], olinucs, prop, sup_info)

        partOne = cur_value - avg
        next_value = get_specific_value(seq[j + lamada - 1], olinucs, prop, sup_info)

        partTwo = next_value - avg
        top = top + (partOne * partTwo)
        j = j + 1
    top = top / limit
    limit2 = length - k + 1
    bottom = 0
    b = 1
    while b < limit2 or b == limit2:
        current = get_specific_value(seq[b - 1], olinucs, prop, sup_info)

        bottom = bottom + ((current - avg) * (current - avg))
        b = b + 1
    bottom = bottom / limit2
    final = top / bottom
    return final


def auto_correlation(auto_method, input_file, props, k, lamada, alphabet):
    if not props:
        error_info = 'Error, The phyche_list, extra_index_file and all_prop can\'t be all False.'
        raise ValueError(error_info)

    input_data = open(input_file, 'r')
    sequences = get_seqs(input_data, alphabet)
    # Getting supporting info from files
    full_path = os.path.realpath(__file__)

    if k == 2 and alphabet == RNA:
        sup_file_name = '%s/data/Supporting_Information_S1_RNA.txt' % os.path.dirname(full_path)
    elif k == 2 and alphabet == DNA:
        sup_file_name = '%s/data/Supporting_Information_S1_DNA.txt' % os.path.dirname(full_path)
    elif k == 3 and alphabet == DNA:
        sup_file_name = '%s/data/Supporting_Information_S3_DNA.txt' % os.path.dirname(full_path)
    else:
        print('Supporting Information error!')
        return False
    sup_file = open(sup_file_name, 'r')
    sup_info = sup_file.read()
    # o = re.search('Physicochemical properties\,(.+)\n', sup_info)
    o = re.search('Physicochemical properties,(.+)\n', sup_info)
    olinucs = ''
    if o:
        olinucs = o.group(1).rstrip()
    sup_file.close()
    # Writing to output file
    m = 0
    vectors = []
    for sequence in sequences:
        length = len(sequence)
        seq = sep_sequence(sequence, k)
        values = []
        for prop in props:
            if auto_method.upper() == 'MAC':
                value = float("%.3f" % moran(seq, olinucs, length, k, lamada, prop, sup_info))
                values.append(value)
            elif auto_method.upper() == 'GAC':
                value = float("%.3f" % geary(seq, olinucs, length, k, lamada, prop, sup_info))
                values.append(value)
            elif auto_method.upper() == 'NMBAC':
                value = float("%.3f" % moreau(seq, olinucs, length, k, lamada, prop, sup_info))
                values.append(value)
        vectors.append(values)
        m += 1
    return np.array(vectors)


# ====================================================================================================

def acc(input_data, k, lag, phyche_list, alphabet, extra_index_file=None, all_prop=False, theta_type=1):
    """This is a complete acc in PseKNC.

    :param alphabet:
    :param lag:
    :param input_data:
    :param k: int, the value of k-tuple.
    :param phyche_list: list, the input physicochemical properties list.
    :param extra_index_file: a file path includes the user-defined phyche_index.
    :param all_prop: bool, choose all physicochemical properties or not.
    :param theta_type: the value 1, 2 and 3 for ac, cc or acc.
    """
    phyche_list = get_phyche_list(k, phyche_list,
                                  extra_index_file=extra_index_file, alphabet=alphabet, all_prop=all_prop)

    phyche_vals = None
    if alphabet == DNA or alphabet == RNA:
        if extra_index_file is not None:
            extra_phyche_index = get_extra_index(extra_index_file)
            from .util_sr import normalize_index
            phyche_vals = get_phyche_value(k, phyche_list, alphabet,
                                           normalize_index(extra_phyche_index, alphabet, is_convert_dict=True))
        else:
            phyche_vals = get_phyche_value(k, phyche_list, alphabet)
    elif alphabet == PROTEIN:
        phyche_vals = get_aaindex(phyche_list)

        if extra_index_file is not None:
            phyche_vals.extend(extend_aaindex(extra_index_file))

    seqs = get_seqs(input_data, alphabet)
    if alphabet == PROTEIN:
        # Transform the data format to dict {acid: [phyche_vals]}.

        phyche_keys = list(phyche_vals[0].index_dict.keys())
        phyche_vals = [list(e.index_dict.values()) for e in phyche_vals]
        new_phyche_vals = list(zip(*[e for e in phyche_vals]))
        phyche_vals = {key: list(val) for key, val in zip(phyche_keys, new_phyche_vals)}

    if theta_type == 1:
        return make_ac_vec(seqs, lag, phyche_vals, k)
    elif theta_type == 2:
        return make_cc_vec(seqs, lag, phyche_vals, k)
    elif theta_type == 3:
        return make_acc_vec(seqs, lag, phyche_vals, k)


def make_ac_vec(sequence_list, lag, phyche_value, k):
    # Get the length of phyche_vals.
    phyche_values = list(phyche_value.values())
    len_phyche_value = len(phyche_values[0])

    vec_ac = []
    for sequence in sequence_list:
        len_seq = len(sequence)
        each_vec = []

        for temp_lag in range(1, lag + 1):
            for j in range(len_phyche_value):

                # Calculate average phyche_value for a nucleotide.
                ave_phyche_value = 0.0
                for i in range(len_seq - k):
                    nucleotide = sequence[i: i + k]
                    ave_phyche_value += float(phyche_value[nucleotide][j])
                ave_phyche_value /= (len_seq - k)

                # Calculate the vector.
                temp_sum = 0.0
                for i in range(len_seq - temp_lag - k + 1):
                    nucleotide1 = sequence[i: i + k]
                    nucleotide2 = sequence[i + temp_lag: i + temp_lag + k]
                    temp_sum += (float(phyche_value[nucleotide1][j]) - ave_phyche_value) * (
                        float(phyche_value[nucleotide2][j]))

                each_vec.append(round(temp_sum / (len_seq - temp_lag - k + 1), 8))
        vec_ac.append(each_vec)

    return np.array(vec_ac)


def make_cc_vec(sequence_list, lag, phyche_value, k):
    phyche_values = list(phyche_value.values())
    len_phyche_value = len(phyche_values[0])

    vec_cc = []
    for sequence in sequence_list:
        len_seq = len(sequence)
        each_vec = []

        for temp_lag in range(1, lag + 1):
            for i1 in range(len_phyche_value):
                for i2 in range(len_phyche_value):
                    if i1 != i2:
                        # Calculate average phyche_value for a nucleotide.
                        ave_phyche_value1 = 0.0
                        ave_phyche_value2 = 0.0
                        for j in range(len_seq - k):
                            nucleotide = sequence[j: j + k]
                            ave_phyche_value1 += float(phyche_value[nucleotide][i1])
                            ave_phyche_value2 += float(phyche_value[nucleotide][i2])
                        ave_phyche_value1 /= (len_seq - k)
                        ave_phyche_value2 /= (len_seq - k)

                        # Calculate the vector.
                        temp_sum = 0.0
                        for j in range(len_seq - temp_lag - k + 1):
                            nucleotide1 = sequence[j: j + k]
                            nucleotide2 = sequence[j + temp_lag: j + temp_lag + k]
                            temp_sum += (float(phyche_value[nucleotide1][i1]) - ave_phyche_value1) * \
                                        (float(phyche_value[nucleotide2][i2]) - ave_phyche_value2)
                        each_vec.append(round(temp_sum / (len_seq - temp_lag - k + 1), 8))

        vec_cc.append(each_vec)

    return np.array(vec_cc)


def make_acc_vec(seqs, lag, phyche_values, k):
    # from functools import reduce
    # zipped = list(zip(make_ac_vec(seqs, lag, phyche_values, k), make_cc_vec(seqs, lag, phyche_values, k)))
    # return np.array(reduce(lambda x, y: x + y, e) for e in zipped)
    ac_vec = make_ac_vec(seqs, lag, phyche_values, k)
    cc_vec = make_cc_vec(seqs, lag, phyche_values, k)
    acc_vec = np.hstack((ac_vec, cc_vec))
    return acc_vec


# --------------------------------------------------------------------------
# PDT method
# --------------------------------------------------------------------------

def pdt_cmd_(input_file, lamada, sw_dir):
    """Concatenation of pdt command.
    :param input_file: the input sequence file in FASTA format.
    :param lamada: the value of parameter lamada.
    :param sw_dir: the main dir of software.
    """
    if sys.platform.startswith('win'):
        pdt_cmd = sw_dir + 'pdt/pdt.exe'
    else:
        pdt_cmd = sw_dir + 'pdt/pdt'
        os.chmod(pdt_cmd, 0o777)

    aaindex_file = sw_dir + 'pdt/aaindex_norm.txt'
    file_path, suffix = os.path.splitext(input_file)
    output_file = ''.join([file_path, '_pdt', suffix])

    cmd = ''.join([pdt_cmd, ' ', input_file, ' ', aaindex_file, ' ', str(lamada), ' ', output_file])

    subprocess.call(cmd, shell=True)
    return os.path.abspath(output_file)


def pdt(input_file, lamada, sw_dir):
    """Execute pdt command and generate feature vectors.
    :param input_file: the input sequence file in FASTA format.
    :param lamada: the value of parameter lamada.
    :param sw_dir: the main dir of software.
    """

    output_file = pdt_cmd_(input_file, lamada, sw_dir)

    vector_list = []
    with open(output_file, 'r') as f:
        for line in f:
            temp_list = line.strip().split('\t')
            vector = [round(float(elem), 3) for elem in temp_list]
            vector_list.append(vector)
    # print vector_list
    return np.array(vector_list)


def nd(input_file, alphabet, fixed_len):
    # for Gene
    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, alphabet)
    seq_list = seq_length_fixed(seq_list, fixed_len)
    nd_list = np.zeros((len(seq_list), fixed_len))
    for seq in seq_list:
        for j in range(fixed_len):
            if j < len(seq):
                if seq[j].upper() == 'A':
                    nd_list[j] = round(seq[0:j + 1].count('A') / (j + 1), 3)
                elif seq[j].upper() == 'U':
                    nd_list[j] = round(seq[0:j + 1].count('U') / (j + 1), 3)
                elif seq[j].upper() == 'C':
                    nd_list[j] = round(seq[0:j + 1].count('C') / (j + 1), 3)
                elif seq[j].upper() == 'G':
                    nd_list[j] = round(seq[0:j + 1].count('G') / (j + 1), 3)
                elif seq[j].upper() == 'T':
                    nd_list[j] = round(seq[0:j + 1].count('T') / (j + 1), 3)
            else:
                nd_list[j] = 0.0
    return nd_list
