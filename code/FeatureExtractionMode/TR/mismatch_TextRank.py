from ..utils.utils_words import mismatch_words
from ..utils.utils_algorithm import text_rank


def mismatch_text_rank(input_file,  alphabet, fixed_len, word_size, alpha, fixed=True):
    corpus = mismatch_words(input_file, alphabet, fixed_len, word_size, fixed)
    return text_rank(corpus, alpha)
