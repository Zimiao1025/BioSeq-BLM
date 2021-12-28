from ..utils.utils_words import km_words
from ..utils.utils_algorithm import tf_idf


def km_tf_idf(input_file, alphabet, fixed_len, word_size, fixed=True):
    corpus = km_words(input_file, alphabet, fixed_len, word_size, fixed)
    return tf_idf(corpus)
