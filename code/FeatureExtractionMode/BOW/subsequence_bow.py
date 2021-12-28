# import numpy as np
# import threading
# import multiprocessing
# from itertools import combinations
# from ..utils.utils_bow import get_km_dict
# from ..utils.utils_fasta import get_seqs
#
#
# def subsequence_bow(input_file, alphabet, k, delta):
#     alphabet = list(alphabet)
#     # cpu_num = int(multiprocessing.cpu_count() - 1)
#     threads = []
#     # sem = threading.Semaphore(cpu_num)
#
#     with open(input_file, 'r') as f:
#         seq_list = get_seqs(f, alphabet)
#
#     km_dict = get_km_dict(k, alphabet)
#     results = np.zeros((len(seq_list), len(km_dict)))
#
#     for i in range(len(seq_list)):
#         print('sequence[%d]' % i)
#         sequence = np.array(list(seq_list[i]))
#         threads.append(threading.Thread(target=get_one_subsequence,
#                                         args=(sequence, i, km_dict, k, delta, results)))
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()
#
#     return results
#
#
# def get_one_subsequence(sequence, index, km_dict, k, delta, results):
#     # sem.acquire()
#
#     vector = np.zeros(len(km_dict))
#     n = len(sequence)
#
#     for sub_seq_index in combinations(list(range(n)), k):
#         # [(0, 1, 2) ,(0, 1, 3) ,(0, 1, 4), ...,(4, 6, 7) ,(5, 6, 7)]
#         sub_seq_index = list(sub_seq_index)
#         subsequence = sequence[sub_seq_index]
#         position = km_dict.get(''.join(subsequence))
#         sub_seq_length = sub_seq_index[-1] - sub_seq_index[0] + 1
#         sub_seq_score = 1 if sub_seq_length == k else delta ** sub_seq_length
#         vector[position] += sub_seq_score
#
#     results[index] = vector
#
#     # time.sleep(2)
#     # sem.release()

import multiprocessing
import threading
import numpy as np
from itertools import combinations

from ..utils.utils_bow import get_km_dict
from ..utils.utils_fasta import get_seqs


def subsequence_bow(filename, alphabet, k, delta):
    alphabet = list(alphabet)
    with open(filename) as f:
        seq_list = get_seqs(f, alphabet)
        cpu_num = int(multiprocessing.cpu_count() / 3)
        batches = construct_partitions(seq_list, cpu_num)
        threads = []
        sem = threading.Semaphore(cpu_num)
        km_dict = get_km_dict(k, alphabet)
        results = np.zeros((len(seq_list), len(km_dict)))

        for batch in batches:
            # temp = pool.apply_async(get_subsequence_profile, (batch, alphabet, k, delta))
            # results.append(temp)

            threads.append(threading.Thread(target=get_subsequence_profile,
                                            args=(seq_list, batch, km_dict, k, delta, results, sem)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results


def construct_partitions(seq_list, cpu_num):
    cpu_num = int(cpu_num)
    seqs_num = len(seq_list)
    batch_num = seqs_num // cpu_num
    batches = []
    for i in range(cpu_num - 1):
        # batch = seq_list[i * batch_num:(i + 1) * batch_num]
        batch = list(range(i * batch_num, (i + 1) * batch_num))
        batches.append(batch)
    # batch = seq_list[(cpu_num - 1) * batch_num:]
    batch = list(range((cpu_num - 1) * batch_num, seqs_num))
    batches.append(batch)
    return batches


def get_subsequence_profile(seq_list, batch, km_dict, k, delta, results, sem):
    sem.acquire()

    for seq_ind in batch:
        print('sequence index: %d\n' % seq_ind)
        sequence = seq_list[seq_ind]
        vector = np.zeros((1, len(km_dict)))
        sequence = np.array(list(sequence))
        n = len(sequence)
        # index_lst = list(combinations(range(n), k))
        for sub_seq_index in combinations(list(range(n)),
                                          k):  # [(0, 1, 2) ,(0, 1, 3) ,(0, 1, 4), ...,(4, 6, 7) ,(5, 6, 7)]
            sub_seq_index = list(sub_seq_index)
            subsequence = sequence[sub_seq_index]
            position = km_dict.get(''.join(subsequence))
            sub_seq_length = sub_seq_index[-1] - sub_seq_index[0] + 1
            sub_seq_score = 1 if sub_seq_length == k else delta ** sub_seq_length
            vector[0, position] += sub_seq_score
        results[seq_ind] = vector
    sem.release()
