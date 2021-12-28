from ..utils.utils_words import mismatch_words
from ..utils.utils_algorithm import tf_idf


def mismatch_tf_idf(input_file,  alphabet, fixed_len, word_size, fixed=True):
    corpus = mismatch_words(input_file,  alphabet, fixed_len, word_size, fixed)
    return tf_idf(corpus)
