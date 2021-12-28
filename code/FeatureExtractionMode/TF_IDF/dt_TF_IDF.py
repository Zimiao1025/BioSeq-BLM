from ..utils.utils_words import dt_words
from ..utils.utils_algorithm import tf_idf


def dt_tf_idf(input_file, fixed_len, max_dis, process_num, cur_dir, fixed=True):
    corpus = dt_words(input_file, fixed_len, max_dis, process_num, cur_dir, fixed)
    return tf_idf(corpus)
