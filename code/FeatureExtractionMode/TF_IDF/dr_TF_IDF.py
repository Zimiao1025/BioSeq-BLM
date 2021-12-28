from ..utils.utils_words import dr_words
from ..utils.utils_algorithm import tf_idf


def dr_tf_idf(input_file, alphabet, fixed_len, max_dis, fixed=True):
    corpus = dr_words(input_file, alphabet, fixed_len, max_dis, fixed)
    print(corpus)
    return tf_idf(corpus)
