from ..utils.utils_topic import lsa, PLsa, lda
from ..TR.TR4vec import text_rank


def text_rank_tm(tm_method, input_file, labels, category, words, fixed_len, sample_num_list, out_format, out_file_list,
                 cur_dir, **param_dict):
    vectors = text_rank(input_file, category, words, fixed_len, sample_num_list, out_format, out_file_list, cur_dir,
                        True, **param_dict)
    if tm_method == 'LSA':
        tm_vectors = lsa(vectors, com_prop=param_dict['com_prop'])
    elif tm_method == 'PLSA':
        _, tm_vectors = PLsa(vectors, com_prop=param_dict['com_prop']).em_algorithm()
    elif tm_method == 'LDA':
        tm_vectors = lda(vectors, labels=None, com_prop=param_dict['com_prop'])
    elif tm_method == 'Labeled-LDA':
        tm_vectors = lda(vectors, labels=labels, com_prop=param_dict['com_prop'])
    else:
        print('Topic model method error!')
        return False
    return tm_vectors
