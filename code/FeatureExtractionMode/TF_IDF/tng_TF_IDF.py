from ..utils.utils_words import tng_words
from ..utils.utils_algorithm import tf_idf


def tng_tf_idf(input_file, fixed_len, word_size, n, process_num, cur_dir, fixed=True):
    corpus = tng_words(input_file, fixed_len, word_size, n, process_num, cur_dir, fixed)
    return tf_idf(corpus)
