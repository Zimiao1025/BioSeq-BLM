#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *


def frequency_p(tol_str, tar_str):
    """Generate the frequency of tar_str in tol_str.

    :param tol_str: mother string.
    :param tar_str: substring.
    """
    i, j, tar_count, tar1_count, tar2_count, tar3_count = 0, 0, 0, 0, 0, 0
    tar_list = []
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
                if (
                        i + 1) % 3 == 1:
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


def Z_curve(sequence, k, alphabet):
    kmer = make_kmer_list(k, alphabet)
    len_kmer = len(kmer)
    i = 0
    f_ZC = []
    fx_list = []
    fy_list = []
    fz_list = []
    while i < len_kmer:
        j = 1
        fre1_list = []
        fre2_list = []
        fre3_list = []
        while j <= 4:
            fre1 = frequency_p(sequence, str(kmer[i]))[1]
            fre2 = frequency_p(sequence, str(kmer[i]))[2]
            fre3 = frequency_p(sequence, str(kmer[i]))[3]
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
        f_ZC.append(fx_list[i])
    for i in range(0, len(fy_list)):
        f_ZC.append(fy_list[i])
    for i in range(0, len(fz_list)):
        f_ZC.append(fz_list[i])

    return f_ZC


def convert_phyche_index_to_dict(phyche_index, alphabet):
    """Convert phyche index from list to dict."""
    # for e in phyche_index:
    #     print e
    len_index_value = len(phyche_index[0])
    k = 0
    for i in range(1, 10):
        if len_index_value < 4 ** i:
            sys.exit("Sorry, the number of each index value is must be 4^k.")
        if len_index_value == 4 ** i:
            k = i
            break
    kmer_list = make_kmer_list(k, alphabet)
    # print kmer_list
    len_kmer = len(kmer_list)
    phyche_index_dict = {}
    for kmer in kmer_list:
        phyche_index_dict[kmer] = []
    # print phyche_index_dict
    phyche_index = list(zip(*phyche_index))
    for i in range(len_kmer):
        phyche_index_dict[kmer_list[i]] = list(phyche_index[i])

    return phyche_index_dict


def make_kmer_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError


def standard_deviation(value_list):
    """Return standard deviation."""
    from math import sqrt
    from math import pow
    n = len(value_list)
    average_value = sum(value_list) * 1.0 / n
    return sqrt(sum([pow(e - average_value, 2) for e in value_list]) * 1.0 / (n - 1))


def normalize_index(phyche_index, alphabet, is_convert_dict=False):
    """Normalize the physicochemical index."""
    normalize_phyche_value = []
    for phyche_value in phyche_index:
        average_phyche_value = sum(phyche_value) * 1.0 / len(phyche_value)
        sd_phyche = standard_deviation(phyche_value)
        normalize_phyche_value.append([round((e - average_phyche_value) / sd_phyche, 2) for e in phyche_value])

    if is_convert_dict is True:
        return convert_phyche_index_to_dict(normalize_phyche_value, alphabet)

    print(normalize_phyche_value)
    return normalize_phyche_value



