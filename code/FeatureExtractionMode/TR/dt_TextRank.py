from ..utils.utils_words import dt_words
from ..utils.utils_algorithm import text_rank


def dt_text_rank(input_file, fixed_len, max_dis, process_num, alpha, cur_dir, fixed=True):
    corpus = dt_words(input_file, fixed_len, max_dis, process_num, cur_dir, fixed)
    return text_rank(corpus, alpha)
