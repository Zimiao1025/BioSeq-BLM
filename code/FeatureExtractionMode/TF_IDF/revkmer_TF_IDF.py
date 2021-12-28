from ..utils.utils_words import rev_km_words
from ..utils.utils_algorithm import tf_idf


def rev_km_tf_idf(input_file, alphabet, fixed_len, word_size, fixed=True):
    corpus = rev_km_words(input_file,  alphabet, fixed_len, word_size, fixed)
    return tf_idf(corpus)
