from ..utils.utils_words import subsequence_words
from ..utils.utils_algorithm import tf_idf


def subsequence_tf_idf(input_file, alphabet, fixed_len, word_size, fixed=True):
    corpus = subsequence_words(input_file, alphabet, fixed_len, word_size, fixed)
    return tf_idf(corpus)
