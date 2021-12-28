from ..utils.utils_pssm import sep_file, produce_all_frequency
from ..utils.utils_words import convert_tng_to_fasta
from .dr_bow import dr_bow


def dt_bow(inputfile, max_dis, cur_dir, process_num):
    """Generate DT method feature vectors.
    :param inputfile: input sequence file in FASTA format.
    :param max_dis: the maximum distance between top-1-gram pairs.
    :param cur_dir: the main dir of code.
    :param process_num: the number of processes used for multiprocessing.
    """
    dir_name, seq_name = sep_file(inputfile)
    sw_dir = cur_dir + '/software/'
    pssm_dir = produce_all_frequency(dir_name, sw_dir, process_num)

    tng_seq_file = convert_tng_to_fasta(pssm_dir, seq_name, inputfile, 1, sw_dir)
    return dr_bow(tng_seq_file, max_dis)
