from ..utils.utils_words import dr_words
from ..utils.utils_algorithm import text_rank


def dr_text_rank(input_file, alphabet, fixed_len, max_dis, alpha, fixed=True):
    corpus = dr_words(input_file, alphabet, fixed_len, max_dis, fixed)
    return text_rank(corpus, alpha)
