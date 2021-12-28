import os
import pickle
import sys
from math import pow
from numpy import array

from ..utils.utils_const import DNA, RNA
from ..utils.utils_fasta import get_seqs
from ..utils.utils_words import make_km_list
from .index_list import DNA, RNA, PROTEIN, didna_list, tridna_list, dirna_list, pro_list


class AAIndex:
    def __init__(self, head, index_dict):
        self.head = head
        self.index_dict = index_dict

    def __str__(self):
        return "%s\n%s" % (self.head, self.index_dict)


def frequency_p(tol_str, tar_str):
    """Generate the frequency of tar_str in tol_str.

    :param tol_str: mother string.
    :param tar_str: substring.
    """
    i, j, tar_count, tar1_count, tar2_count, tar3_count = 0, 0, 0, 0, 0, 0
    len_tol_str = len(tol_str)
    len_tar_str = len(tar_str)
    while i < len_tol_str and j < len_tar_str:
        if tol_str[i] == tar_str[j]:
            i += 1
            j += 1
            if j >= len_tar_str:
                tar_count += 1
                i = i - j + 1
                j = 0
                if (i + 1) % 3 == 1:
                    # judge the position of last base of kmer in corresponding codon. pay attention to "i + 1"
                    tar1_count += 1
                elif (i + 1) % 3 == 2:
                    tar2_count += 1
                else:
                    tar3_count += 1
        else:
            i = i - j + 1
            j = 0
    tar_list = (tar_count, tar1_count, tar2_count, tar3_count)
    return tar_list


def z_curve(sequence, k, alphabet):
    km = make_km_list(k, alphabet)
    len_km = len(km)
    i = 0
    f_z_curve = []
    fx_list = []
    fy_list = []
    fz_list = []
    while i < len_km:
        j = 1
        fre1_list = []
        fre2_list = []
        fre3_list = []
        while j <= 4:
            fre1 = frequency_p(sequence, str(km[i]))[1]
            fre2 = frequency_p(sequence, str(km[i]))[2]
            fre3 = frequency_p(sequence, str(km[i]))[3]
            fre1_list.append(fre1)
            fre2_list.append(fre2)
            fre3_list.append(fre3)
            j += 1
            i += 1
        fx1 = (fre1_list[0] + fre1_list[2]) - (fre1_list[1] + fre1_list[3])
        fx2 = (fre2_list[0] + fre2_list[2]) - (fre2_list[1] + fre2_list[3])
        fx3 = (fre3_list[0] + fre3_list[2]) - (fre3_list[1] + fre3_list[3])
        fx_list.append(fx1)
        fx_list.append(fx2)
        fx_list.append(fx3)
        fy1 = (fre1_list[0] + fre1_list[1]) - (fre1_list[2] + fre1_list[3])
        fy2 = (fre2_list[0] + fre2_list[1]) - (fre2_list[2] + fre2_list[3])
        fy3 = (fre3_list[0] + fre3_list[1]) - (fre3_list[2] + fre3_list[3])
        fy_list.append(fy1)
        fy_list.append(fy2)
        fy_list.append(fy3)
        fz1 = (fre1_list[0] + fre1_list[3]) - (fre1_list[1] + fre1_list[2])
        fz2 = (fre2_list[0] + fre2_list[3]) - (fre2_list[1] + fre2_list[2])
        fz3 = (fre3_list[0] + fre3_list[3]) - (fre3_list[1] + fre3_list[2])
        fz_list.append(fz1)
        fz_list.append(fz2)
        fz_list.append(fz3)
    for i in range(0, len(fx_list)):
        f_z_curve.append(fx_list[i])
    for i in range(0, len(fy_list)):
        f_z_curve.append(fy_list[i])
    for i in range(0, len(fz_list)):
        f_z_curve.append(fz_list[i])

    return f_z_curve


def zcpseknc(input_data, k, w, lamada, alphabet):
    """This is a complete process in ZCPseKNC."""
    with open(input_data, 'r') as f:
        seq_list = get_seqs(f, alphabet)

    return make_zcpseknc_vector(seq_list, k, w, lamada, alphabet)


def get_phyche_list(k, phyche_list, extra_index_file, alphabet, all_prop=False):
    # """Get phyche_list and check it.
    #
    # :param k: int, the value of k-tuple.
    # :param phyche_list: list, the input physicochemical properties list.
    # :param all_prop: bool, choose all physicochemical properties or not.
    # """
    if phyche_list is None or len(phyche_list) == 0:
        if extra_index_file is None and all_prop is False:
            error_info = 'Error, The phyche_list, extra_index_file and all_prop can\'t be all False.'
            raise ValueError(error_info)

    try:
        if alphabet == DNA:
            if k == 2:
                all_prop_list = didna_list
            elif k == 3:
                all_prop_list = tridna_list
            else:
                error_info = 'Error, the k value must be 2 or 3.'
                raise ValueError(error_info)
        elif alphabet == RNA:
            if k == 2:
                all_prop_list = dirna_list
            else:
                error_info = 'Error, the k or alphabet error.'
                raise ValueError(error_info)
        elif alphabet == PROTEIN:
            all_prop_list = pro_list
        else:
            error_info = "Error, the alphabet must be dna, rna or protein."
            raise ValueError(error_info)
    except:
        raise

    # Set and check physicochemical properties.
    try:
        # Set all properties.
        if all_prop is True:
            phyche_list = all_prop_list
        # Check phyche properties.
        else:
            for e in phyche_list:
                if e not in all_prop_list:
                    error_info = 'Sorry, the physicochemical properties ' + e + ' is not exit.'
                    raise NameError(error_info)
    except:
        raise

    return phyche_list


def get_extra_index(filename):
    """Get the extend indices from index file, only work for DNA and RNA."""
    extra_index_vals = []
    with open(filename) as f:
        lines = f.readlines()
        for ind, line in enumerate(lines):
            if line[0] == '>':
                vals = lines[ind + 2].rstrip().strip().split('\t')
                vals = [float(val) for val in vals]
                extra_index_vals.append(vals)

    return extra_index_vals


def get_aaindex(index_list):
    """Get the aaindex from data/aaindex.data.

    :param index_list: the index we want to get.
    :return: a list of AAIndex obj.
    """
    new_aaindex = []
    full_path = os.path.realpath(__file__)
    file_path = "%s/data/aaindex.data" % os.path.dirname(full_path)
    with open(file_path, 'rb') as f:
        aaindex = pickle.load(f)
        for index_vals in aaindex:
            if index_vals.head in index_list:
                new_aaindex.append(index_vals)

    return new_aaindex


def extend_aaindex(filename):
    """Extend the user-defined AAIndex from user's file.
    :return: a list of AAIndex obj.
    """
    from .extract_aaindex import norm_index_vals

    aaindex = get_ext_ind_pro(filename)
    for ind, (head, index_dict) in enumerate(aaindex):
        aaindex[ind] = AAIndex(head, norm_index_vals(index_dict))
    return aaindex


def get_ext_ind_pro(filename):
    """Get the extend indices from index file, only work for protein."""
    inds = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aaindex = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[0] == '>':
                temp_name = line[1:].rstrip()
                vals = lines[i + 2].rstrip().split('\t')
                ind_val = {ind: float(val) for ind, val in zip(inds, vals)}
                aaindex.append((temp_name, ind_val))

    return aaindex


def get_phyche_value(k, phyche_list, alphabet, extra_phyche_index=None):
    """Generate DNA or RNA phyche_value.

    :param k: int, the value of k-tuple.
    :param phyche_list: physicochemical properties list.
    :param extra_phyche_index: dict, the key is the olinucleotide (string),
                                     the value is its physicochemical property value (list).
                               It means the user-defined physicochemical indices.
    """
    if extra_phyche_index is None:
        extra_phyche_index = {}

    phyche_value = extend_phyche_index(get_phyche_index(k, phyche_list, alphabet), extra_phyche_index)

    return phyche_value


def extend_phyche_index(original_index, extend_index):
    """Extend DNA or RNA {phyche:[value, ... ]}"""
    if extend_index is None or len(extend_index) == 0:
        return original_index
    for key in list(original_index.keys()):
        original_index[key].extend(extend_index[key])
    return original_index


def get_phyche_factor_dic(k, alphabet):
    """Get all DNA or RNA {nucleotide: [(phyche, value), ...]} dict."""
    full_path = os.path.realpath(__file__)
    if 2 == k and alphabet == DNA:
        file_path = "%s/data/didna.data" % os.path.dirname(full_path)
    elif 2 == k and alphabet == RNA:
        file_path = "%s/data/dirna.data" % os.path.dirname(full_path)
    elif 3 == k:
        file_path = "%s/data/mmc4.data" % os.path.dirname(full_path)
    else:
        sys.stderr.write("The k can just be 2 or 3.")
        sys.exit(0)

    try:
        with open(file_path, 'rb') as f:
            phyche_factor_dic = pickle.load(f)
    except:
        with open(file_path, 'r') as f:
            phyche_factor_dic = pickle.load(f)

    return phyche_factor_dic


def get_phyche_index(k, phyche_list, alphabet):
    """get phyche_value according phyche_list."""
    phyche_value = {}
    if 0 == len(phyche_list):
        for nucleotide in make_km_list(k, alphabet):
            phyche_value[nucleotide] = []
        return phyche_value

    nucleotide_phyche_value = get_phyche_factor_dic(k, alphabet)
    for nucleotide in make_km_list(k, alphabet):
        if nucleotide not in phyche_value:
            phyche_value[nucleotide] = []
        for e in nucleotide_phyche_value[nucleotide]:
            if e[0] in phyche_list:
                phyche_value[nucleotide].append(e[1])

    return phyche_value


def get_theta(k, lamada, sequence, alphabet):
    """Get the  theta list which use frequency to replace physicochemical properties(the kernel of ZCPseKNC method."""
    theta = []
    L = len(sequence)
    kmer = make_km_list(k, alphabet)
    fre_list = [frequency_p(sequence, str(key))[0] for key in kmer]
    fre_sum = float(sum(fre_list))
    for i in range(1, lamada + 1):
        temp_sum = 0.0
        for j in range(0, L - k - i + 1):
            nucleotide1 = sequence[j: j + k]
            nucleotide2 = sequence[j + i: j + i + k]
            if alphabet == DNA:
                fre_nucleotide1 = frequency_p(sequence, str(nucleotide1))[0] / fre_sum
                fre_nucleotide2 = frequency_p(sequence, str(nucleotide2))[0] / fre_sum
                temp_sum += pow(float(fre_nucleotide1) - float(fre_nucleotide2), 2)
            else:
                sys.stderr.write("The ZCPseKNC method just for DNA.")
                sys.exit(0)

        theta.append(temp_sum / (L - k - i + 1))

    return theta


def make_zcpseknc_vector(sequence_list, k=2, w=0.05, lamada=1, alphabet=DNA):
    # use theta_type=1 variable can distinguish method
    """Generate the ZCPseKNC vector."""
    kmer = make_km_list(k, alphabet)
    vector = []

    for sequence in sequence_list:
        if len(sequence) < k or lamada + k > len(sequence):
            error_info = "Sorry, the sequence length must be larger than " + str(lamada + k)
            sys.stderr.write(error_info)
            sys.exit(0)

        # Get the nucleotide frequency in the DNA sequence.
        fre_list = [frequency_p(sequence, str(key))[0] for key in kmer]
        fre_sum = float(sum(fre_list))
        fre_list = z_curve(sequence, k, alphabet)

        # Get the normalized occurrence frequency of nucleotide in the DNA sequence.
        fre_list = [e / fre_sum for e in fre_list]
        fre_sum = float(sum(fre_list))

        # Get the theta_list.
        theta_list = get_theta(k, lamada, sequence, alphabet)
        theta_sum = sum(theta_list)

        # Generate the vector according the Equation .
        denominator = fre_sum + w * theta_sum

        temp_vec = [round(f / denominator, 8) for f in fre_list]
        for theta in theta_list:
            temp_vec.append(round(w * theta / denominator, 8))

        vector.append(temp_vec)

    return array(vector)
