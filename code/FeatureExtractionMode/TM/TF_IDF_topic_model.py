from ..utils.utils_topic import lsa, PLsa, lda
from ..TF_IDF.TF_IDF4vec import tf_idf


def tf_idf_lsa(input_file, alphabet, words, **param_dict):
    bow_vectors = tf_idf(input_file, alphabet, words, **param_dict)
    lsa_vectors = lsa(bow_vectors, com_prop=param_dict['com_prop'])
    return lsa_vectors


def tf_idf_plsa(input_file, alphabet, words, **param_dict):
    bow_vectors = tf_idf(input_file, alphabet, words, **param_dict)
    _, plsa_vectors = PLsa(bow_vectors, com_prop=param_dict['com_prop']).em_algorithm()
    return plsa_vectors


def tf_idf_lda(input_file, alphabet, words, **param_dict):
    bow_vectors = tf_idf(input_file, alphabet, words, **param_dict)
    lda_vectors = lda(bow_vectors, labels=None, com_prop=param_dict['com_prop'])
    return lda_vectors


def tf_idf_label_lda(input_file, labels, alphabet, words, **param_dict):
    bow_vectors = tf_idf(input_file, alphabet, words, **param_dict)
    lda_vectors = lda(bow_vectors, labels=labels, com_prop=param_dict['com_prop'])
    return lda_vectors


def tf_idf_tm(tm_method, input_file, labels, category, words, fixed_len, sample_num_list, out_format, out_file_list,
              cur_dir, **param_dict):
    vectors = tf_idf(input_file, category, words, fixed_len, sample_num_list, out_format, out_file_list, cur_dir,
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
