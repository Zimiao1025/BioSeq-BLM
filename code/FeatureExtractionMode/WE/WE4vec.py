from .Glove4vec import glove4vec
from .fastText4vec import fast_text4vec
from .word4vec import word4vec
from ..utils.utils_write import vectors2files
from ..utils.utils_words import DNA_X, RNA_X, PROTEIN_X
from ..utils.utils_words import dr_words, dt_words, km_words, mismatch_words, rev_km_words, subsequence_words, tng_words


def word_emb(emb_method, input_file, category, words, fixed_len, sample_num_list, out_format, out_file_list, cur_dir,
             **param_dict):
    if category == 'DNA':
        alphabet = DNA_X
    elif category == 'RNA':
        alphabet = RNA_X
    else:
        alphabet = PROTEIN_X

    if words == 'Kmer':
        corpus = km_words(input_file, alphabet, fixed_len, word_size=param_dict['word_size'], fixed=True)
    elif words == 'RevKmer':
        corpus = rev_km_words(input_file, alphabet, fixed_len, word_size=param_dict['word_size'], fixed=True)
    elif words == 'Mismatch':
        corpus = mismatch_words(input_file, alphabet, fixed_len, word_size=param_dict['word_size'], fixed=True)
    elif words == 'Subsequence':
        corpus = subsequence_words(input_file, alphabet, fixed_len, word_size=param_dict['word_size'], fixed=True)
    elif words == 'Top-N-Gram':
        corpus = tng_words(input_file, fixed_len, word_size=param_dict['word_size'], n=param_dict['top_n'],
                           process_num=param_dict['cpu'], cur_dir=cur_dir, fixed=True)
    elif words == 'DR':
        corpus = dr_words(input_file, alphabet, fixed_len, max_dis=param_dict['max_dis'], fixed=True)
    elif words == 'DT':
        corpus = dt_words(input_file, fixed_len, max_dis=param_dict['max_dis'], process_num=param_dict['cpu'],
                          cur_dir=cur_dir, fixed=True)
    else:
        print('word segmentation method error!')
        return False

    if emb_method == 'fastText':
        emb_vectors = fast_text4vec(corpus, sample_num_list, fixed_len, **param_dict)
    elif emb_method == 'Glove':
        emb_vectors = glove4vec(corpus, sample_num_list, fixed_len, **param_dict)
    elif emb_method == 'word2vec':
        # word4vec(input_file, alphabet, sample_size_list, words, fixed_len, cur_dir, **param_dict):
        emb_vectors = word4vec(corpus, sample_num_list, fixed_len, **param_dict)
    else:
        print('Word embedding method error!')
        return False
    vectors2files(emb_vectors, sample_num_list, out_format, out_file_list)
