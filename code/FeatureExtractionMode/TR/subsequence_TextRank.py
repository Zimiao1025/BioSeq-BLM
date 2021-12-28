from ..utils.utils_words import subsequence_words
from ..utils.utils_algorithm import text_rank


def subsequence_text_rank(input_file, alphabet, fixed_len, word_size, alpha, fixed=True):
    corpus = subsequence_words(input_file, alphabet, fixed_len, word_size, fixed)
    return text_rank(corpus, alpha)
