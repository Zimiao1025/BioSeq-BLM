from ..utils.utils_algorithm import fast_text


def fast_text4vec(corpus, sample_size_list, fixed_len, **param_dict):
    corpus_out = fast_text(corpus, sample_size_list, fixed_len, word_size=param_dict['word_size'],
                           win_size=param_dict['win_size'], vec_dim=param_dict['vec_dim'], skip_gram=param_dict['sg'])
    return corpus_out
