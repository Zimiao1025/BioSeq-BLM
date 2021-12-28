from ..utils.utils_words import tng_words
from ..utils.utils_algorithm import text_rank


def tng_text_rank(input_file, fixed_len, word_size, n, process_num, alpha, cur_dir, fixed=True):
    corpus = tng_words(input_file, fixed_len, word_size, n, process_num, cur_dir, fixed)
    return text_rank(corpus, alpha)
