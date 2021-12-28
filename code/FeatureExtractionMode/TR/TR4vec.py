from .kmer_TextRank import km_text_rank
from .revkmer_TextRank import rev_km_text_rank
from .mismatch_TextRank import mismatch_text_rank
from .subsequence_TextRank import subsequence_text_rank
from .tng_TextRank import tng_text_rank
from .dr_TextRank import dr_text_rank
from .dt_TextRank import dt_text_rank
from ..utils.utils_write import vectors2files
from ..utils.utils_words import DNA_X, RNA_X, PROTEIN_X


def text_rank(input_file, category, words, fixed_len, sample_num_list, out_format, out_file_list, cur_dir, tm=False,
              **param_dict):
    if category == 'DNA':
        alphabet = DNA_X
    elif category == 'RNA':
        alphabet = RNA_X
    else:
        alphabet = PROTEIN_X

    if words == 'Kmer':
        tr_vectors = km_text_rank(input_file, alphabet, fixed_len, word_size=param_dict['word_size'],
                                  alpha=param_dict['alpha'], fixed=True)
    elif words == 'RevKmer':
        tr_vectors = rev_km_text_rank(input_file, alphabet, fixed_len, word_size=param_dict['word_size'],
                                      alpha=param_dict['alpha'], fixed=True)
    elif words == 'Mismatch':
        tr_vectors = mismatch_text_rank(input_file, alphabet, fixed_len, word_size=param_dict['word_size'],
                                        alpha=param_dict['alpha'], fixed=True)
    elif words == 'Subsequence':
        tr_vectors = subsequence_text_rank(input_file, alphabet, fixed_len, word_size=param_dict['word_size'],
                                           alpha=param_dict['alpha'], fixed=True)
    elif words == 'Top-N-Gram':
        tr_vectors = tng_text_rank(input_file, fixed_len, word_size=param_dict['word_size'], n=param_dict['top_n'],
                                   process_num=param_dict['cpu'], alpha=param_dict['alpha'], cur_dir=cur_dir,
                                   fixed=True)
    elif words == 'DR':
        tr_vectors = dr_text_rank(input_file, alphabet, fixed_len, max_dis=param_dict['max_dis'],
                                  alpha=param_dict['alpha'], fixed=True)
    elif words == 'DT':
        tr_vectors = dt_text_rank(input_file, fixed_len, max_dis=param_dict['max_dis'], process_num=param_dict['cpu'],
                                  alpha=param_dict['alpha'], cur_dir=cur_dir,
                                  fixed=True)
    else:
        print('word segmentation method error!')
        return False
    if tm is False:
        vectors2files(tr_vectors, sample_num_list, out_format, out_file_list)
    else:
        return tr_vectors
