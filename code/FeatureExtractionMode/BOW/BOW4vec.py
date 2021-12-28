from .kmer_bow import km_bow
from .mismatch_bow import mismatch_bow
from .subsequence_bow import subsequence_bow
from .tng_bow import tng_bow
from .dr_bow import dr_bow
from .dt_bow import dt_bow
from ..utils.utils_write import vectors2files
from ..utils.utils_const import DNA, RNA, PROTEIN


def bow(input_file, category, words, sample_num_list, out_format, out_file_list, cur_dir, tm=False, **param_dict):
    if category == 'DNA':
        alphabet = DNA
    elif category == 'RNA':
        alphabet = RNA
    else:
        alphabet = PROTEIN
    if words == 'Kmer':
        bow_vectors = km_bow(input_file, k=param_dict['word_size'], alphabet=alphabet, rev_comp=False)
    elif words == 'RevKmer':
        bow_vectors = km_bow(input_file, k=param_dict['word_size'], alphabet=alphabet, rev_comp=True)
    elif words == 'Mismatch':
        bow_vectors = mismatch_bow(input_file, alphabet, k=param_dict['word_size'], m=param_dict['mis_num'])
    elif words == 'Subsequence':
        bow_vectors = subsequence_bow(input_file, alphabet, k=param_dict['word_size'], delta=param_dict['delta'])
    elif words == 'Top-N-Gram':
        bow_vectors = tng_bow(input_file, n=param_dict['top_n'], cur_dir=cur_dir, process_num=param_dict['cpu'])
    elif words == 'DR':
        bow_vectors = dr_bow(input_file, max_dis=param_dict['max_dis'])
    elif words == 'DT':
        bow_vectors = dt_bow(input_file, max_dis=param_dict['max_dis'], cur_dir=cur_dir, process_num=param_dict['cpu'])
    else:
        print('word segmentation method error!')
        return False
    if tm is False:
        vectors2files(bow_vectors, sample_num_list, out_format, out_file_list)
    else:
        return bow_vectors
