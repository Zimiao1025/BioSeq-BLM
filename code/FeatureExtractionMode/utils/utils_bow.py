from itertools import combinations_with_replacement, permutations
import numpy as np


def get_km_dict(k, alphabet):
    """ Func: get km dictionary -> {'AA': 0, 'AC': 1, ..., 'TT': 15}"""
    km_list = []
    part_km = list(combinations_with_replacement(alphabet, k))
    for element in part_km:
        ele_set = set(permutations(element, k))
        str_list = [''.join(ele) for ele in ele_set]
        km_list += str_list
    km_list = np.sort(km_list)
    km_dict = {km_list[i]: i for i in range(len(km_list))}
    return km_dict


def frequency(tol_str, tar_str):
    """Generate the frequency of tar_str in tol_str.

    :param tol_str: mother string.
    :param tar_str: substring.
    """
    i, j, tar_count = 0, 0, 0
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
        else:
            i = i - j + 1
            j = 0

    return tar_count
