from ..utils.utils_words import km_words
from ..utils.utils_algorithm import text_rank


def km_text_rank(input_file, alphabet, fixed_len, word_size, alpha, fixed=True):
    corpus = km_words(input_file, alphabet, fixed_len, word_size, fixed)
    return text_rank(corpus, alpha)
