import numpy as np
from ..utils.utils_bow import get_km_dict
from ..utils.utils_fasta import get_seqs


def mismatch_bow(input_file, alphabet, k, m):
    alphabet = list(alphabet)
    p = len(alphabet)
    km_dict = get_km_dict(k, alphabet)
    features = []
    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, alphabet)
    if m == 0 and m < k:
        for sequence in seq_list:
            vector = get_spectrum(sequence, km_dict, p, k)
            features.append(vector)
    else:
        for sequence in seq_list:
            vector = get_mismatch(sequence, alphabet, km_dict, p, k)
            features.append(vector)
        return np.array(features)


def get_spectrum(sequence, km_dict, p, k):
    vector = np.zeros((1, p ** k))
    n = len(sequence)
    for i in range(n - k + 1):
        subsequence = sequence[i:i + k]
        position = km_dict.get(subsequence)
        vector[0, position] += 1
    return list(vector[0])


def get_mismatch(sequence, alphabet, km_dict, p, k):
    n = len(sequence)
    vector = np.zeros((1, p ** k))
    for i in range(n - k + 1):
        subsequence = sequence[i:i + k]
        position = km_dict.get(subsequence)
        vector[0, position] += 1
        for j in range(k):
            substitution = subsequence
            for letter in list(set(alphabet) ^ set(subsequence[j])):  # 求字母并集
                substitution = list(substitution)
                substitution[j] = letter
                substitution = ''.join(substitution)
                position = km_dict.get(substitution)
                vector[0, position] += 1
    return list(vector[0])
