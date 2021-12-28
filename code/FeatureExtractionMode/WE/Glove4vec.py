from ..utils.utils_algorithm import glove


def glove4vec(corpus, sample_size_list, fixed_len, **param_dict):

    corpus_out = glove(corpus, sample_size_list, fixed_len, word_size=param_dict['word_size'],
                       win_size=param_dict['win_size'], vec_dim=param_dict['vec_dim'])
    return corpus_out
