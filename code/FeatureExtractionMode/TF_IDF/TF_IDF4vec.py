from .kmer_TF_IDF import km_tf_idf
from .revkmer_TF_IDF import rev_km_tf_idf
from .mismatch_TF_IDF import mismatch_tf_idf
from .subsequence_TF_IDF import subsequence_tf_idf
from .tng_TF_IDF import tng_tf_idf
from .dr_TF_IDF import dr_tf_idf
from .dt_TF_IDF import dt_tf_idf
from ..utils.utils_write import vectors2files
from ..utils.utils_words import DNA_X, RNA_X, PROTEIN_X


def tf_idf(input_file, category, words, fixed_len, sample_num_list, out_format, out_file_list, cur_dir, tm=False,
           **param_dict):
    if category == 'DNA':
        alphabet = DNA_X
    elif category == 'RNA':
        alphabet = RNA_X
    else:
        alphabet = PROTEIN_X

    if words == 'Kmer':
        tf_vectors = km_tf_idf(input_file, alphabet, fixed_len, word_size=param_dict['word_size'], fixed=True)
    elif words == 'RevKmer':
        tf_vectors = rev_km_tf_idf(input_file, alphabet, fixed_len, word_size=param_dict['word_size'], fixed=True)
    elif words == 'Mismatch':
        tf_vectors = mismatch_tf_idf(input_file, alphabet, fixed_len, word_size=param_dict['word_size'], fixed=True)
    elif words == 'Subsequence':
        tf_vectors = subsequence_tf_idf(input_file, alphabet, fixed_len, word_size=param_dict['word_size'], fixed=True)
    elif words == 'Top-N-Gram':
        tf_vectors = tng_tf_idf(input_file, fixed_len, word_size=param_dict['word_size'], n=param_dict['top_n'],
                                process_num=param_dict['cpu'], cur_dir=cur_dir, fixed=True)
    elif words == 'DR':
        tf_vectors = dr_tf_idf(input_file, alphabet, fixed_len, max_dis=param_dict['max_dis'], fixed=True)
    elif words == 'DT':
        # input_file, fixed_len, max_dis, process_num, cur_dir, fixed=True
        tf_vectors = dt_tf_idf(input_file, fixed_len, max_dis=param_dict['max_dis'],
                               process_num=param_dict['cpu'], cur_dir=cur_dir, fixed=True)
    else:
        print('word segmentation method error!')
        return False
    if tm is False:
        vectors2files(tf_vectors, sample_num_list, out_format, out_file_list)
    else:
        return tf_vectors
