import heapq
import itertools
import os
import pickle
import sys

from .utils_fasta import get_seqs
from ..utils.utils_const import PROTEIN
from ..utils.utils_pssm import sep_file, produce_all_frequency

PROTEIN_X = "ACDEFGHIKLMNPQRSTVWYX"
DNA_X = "ACGTX"
RNA_X = "ACGUX"


def make_km_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError


def seq_length_fixed(seq_list, fixed_len):
    sequence_list = []
    for seq in seq_list:
        seq_len = len(seq)
        if seq_len <= fixed_len:
            for i in range(fixed_len - seq_len):
                seq += 'X'
        else:
            seq = seq[:fixed_len]
        sequence_list.append(seq)

    return sequence_list


def km_words(input_file, alphabet, fixed_len, word_size, fixed=True):
    """ convert sequence to corpus """
    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, alphabet)
    if fixed is True:
        seq_list = seq_length_fixed(seq_list, fixed_len)  # 注意这里的fixed_len对于蛋白质一般是400， RNA一般为1001
    corpus = []
    for sequence in seq_list:
        word_list = []
        # windows slide along sequence to generate gene/protein words
        for i in range(len(sequence) - word_size + 1):
            word = sequence[i:i + word_size]
            word_list.append(word)
        corpus.append(word_list)
    return corpus


def get_rev_comp_dict(k):
    """ get reverse composition dictionary
    :return: {'ACG': 'CGT', ...} """
    km_list = make_km_list(k, "ACGTX")  # 只能是DNA序列

    rev_comp_dict = {}
    # Make a reversed version of the string.
    for km in km_list:
        rev_sequence = list(km)
        rev_sequence.reverse()
        rev_sequence = ''.join(rev_sequence)

        return_value = ""
        for letter in rev_sequence:
            if letter == "A":
                return_value += "T"
            elif letter == "C":
                return_value += "G"
            elif letter == "G":
                return_value += "C"
            elif letter == "T":
                return_value += "A"
            elif letter == "X":
                return_value += "X"
            else:
                error_info = ("Unknown DNA character (%s)\n" % letter)
                sys.exit(error_info)

        # Store this value for future use.
        rev_comp_dict[km] = return_value

    return rev_comp_dict


def rev_km_words(input_file,  alphabet, fixed_len, word_size, fixed=True):
    """ convert sequence to corpus """
    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, alphabet)
    if fixed is True:
        seq_list = seq_length_fixed(seq_list, fixed_len)  # 注意这里的fixed_len对于蛋白质一般是400， RNA一般为1001
    rev_comp_dict = get_rev_comp_dict(word_size)
    corpus = []   # 构建一个不同单词的语料库，来方便后续的operation
    for sequence in seq_list:
        word_list = []
        # windows slide along sequence to generate gene/protein words
        for i in range(len(sequence) - word_size + 1):
            word = sequence[i:i + word_size]
            rev_km_word = rev_comp_dict[word]
            word_list.append(rev_km_word)
        corpus.append(word_list)
    return corpus


def get_mismatch_dict(k, alphabet):
    """ Attention: Fixed m = 1
    :return: {'AA': ['AA', 'TA', 'GA', 'CA', 'AT', 'AG', 'AC'], ...}"""
    # 这里在产生mismatch words的过程中固定m=1, 否则产生的语料库过大且不合理!
    # 如：k=3, m=2时，len(dict['ACT']) = 4*4 -3*3, 意味着有37个3苷酸能take them as the occurrences of 'ACT'.
    mismatch_dict = {}
    km_list = make_km_list(k, alphabet)
    for km in km_list:
        sub_list = [''.join(km)]
        for j in range(k):
            for letter in list(set(alphabet) ^ set(km[j])):  # 求字母并集
                substitution = list(km)
                substitution[j] = letter
                substitution = ''.join(substitution)
                sub_list.append(substitution)
        mismatch_dict[km] = sub_list
    return mismatch_dict


def mismatch_words(input_file, alphabet, fixed_len, word_size, fixed=True):
    """ convert sequence to corpus """
    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, alphabet)
    if fixed is True:
        seq_list = seq_length_fixed(seq_list, fixed_len)  # 注意这里的fixed_len对于蛋白质一般是400， RNA一般为1001
    mismatch_dict = get_mismatch_dict(word_size, alphabet)
    corpus = []   # 构建一个不同单词的语料库，来方便后续的operation
    for sequence in seq_list:
        word_list = []
        # windows slide along sequence to generate gene/protein words
        for i in range(len(sequence) - word_size + 1):
            word = sequence[i:i + word_size]
            mis_words = mismatch_dict[word]
            word_list += mis_words
        corpus.append(word_list)
    return corpus


def subsequence_words(input_file, alphabet, fixed_len, word_size, fixed=True):
    """ convert sequence to corpus """
    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, alphabet)
    if fixed is True:
        seq_list = seq_length_fixed(seq_list, fixed_len)  # 注意这里的fixed_len对于蛋白质一般是400， RNA一般为1001
    corpus = []
    for seq in seq_list:
        corpus.append(combination_dck(seq, word_size, dc=5))  # 这里非常特殊，需要注意！

    return corpus


def combination_dck(s, k, dc):
    # dc: 距离控制参数

    if k == 0:
        return ['']
    sub_letters = []
    # 此处涉及到一个 python 遍历循环的特点：当遍历的对象为空（列表，字符串...）时，循环不会被执行，range(0) 也是一样

    for i in range(len(s)):

        for letter in combination_dck(s[i + 1: i + dc], k - 1, dc):
            sub_letters += [s[i] + letter]

    return sub_letters


def produce_one_top_n_gram(pssm_file, n):
    """Produce top-n-gram for one pssm file.
    :param pssm_file: the pssm file used to generate top-n-gram.
    :param n: the top n most frequency amino acids in the corresponding column of a frequency profile
    """
    tng_list = []
    new_alpha_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
                      'P', 'S', 'T', 'W', 'Y', 'V']
    with open(pssm_file, 'r') as f:
        count = 0
        for line in f:
            count += 1
            if count <= 3 or len(line.strip().split()) < 20:
                pass
            else:
                line = line.strip().split()[22:-2]
                line = list(map(eval, line))
                # print line
                data = heapq.nlargest(n, enumerate(line), key=lambda x: x[1])
                # print data
                if n == 1:
                    new_alpha = new_alpha_list[data[0][0]]
                else:
                    new_alpha = ''
                    indices, val = list(zip(*data))
                    for i in indices:
                        new_alpha += new_alpha_list[i]
                tng_list.append(new_alpha)

    return tng_list


def produce_tng_blosum(seq_file, blosum_dict, n):
    """Generate top-n-gram by blosum62 matrix.
    :param seq_file: the sequence file containing one sequence.
    :param blosum_dict: the dict which stores the blosum62 matrix.
    :param n: the top n most frequency amino acids in the corresponding column of a frequency profile.
    """
    tng_list = []
    with open(seq_file, 'r') as f:
        for line in f:
            if line.strip().startswith('>'):
                continue
            else:
                line = line.strip()
                for amino in line:
                    amino_blosum = blosum_dict[amino]
                    data = heapq.nlargest(n, enumerate(amino_blosum), key=lambda x: x[1])
                    if n == 1:
                        index = data[0][0]
                        new_alpha = blosum_dict['alphas'][index]
                    else:
                        new_alpha = ''
                        indices, val = list(zip(*data))
                        for i in indices:
                            new_alpha += blosum_dict['alphas'][i]
                    tng_list.append(new_alpha)
    return tng_list


def produce_top_n_gram(pssm_dir, seq_name, n, sw_dir):
    """Produce top-n-gram for all the pssm files.
    :param pssm_dir: the directory used to store pssm files.
    :param seq_name: the name of sequences.
     :param n: the top n most frequency amino acids in the corresponding column of a frequency profile
     :param sw_dir: the main dir of software.
    """
    dir_name = os.path.split(pssm_dir)[0]

    dir_list = os.listdir(pssm_dir)
    index_list = []
    for elem in dir_list:
        pssm_full_path = ''.join([pssm_dir, '/', elem])
        name, suffix = os.path.splitext(elem)
        if os.path.isfile(pssm_full_path) and suffix == '.pssm':
            index_list.append(int(name))

    index_list.sort()

    if len(index_list) != len(seq_name):
        BLOSUM62 = sw_dir + 'psiblast/blosum62.pkl'
        with open(BLOSUM62, 'rb') as f:
            blosum_dict = pickle.load(f)

    # print tng_all_list
    for i in range(1, len(seq_name) + 1):
        if i in index_list:
            pssm_full_path = ''.join([pssm_dir, '/', str(i), '.pssm'])
            tng = produce_one_top_n_gram(pssm_full_path, n)
        else:
            seq_file = ''.join([dir_name, '/', str(i), '.txt'])
            tng = produce_tng_blosum(seq_file, blosum_dict, n)

        yield tng


def convert_tng_to_fasta(pssm_dir, seq_name, origin_file_name, n, sw_dir):
    """Convert top-n-gram to fasta format.
    :param n: the top n most frequency amino acids in the corresponding column of a frequency profile
    :param pssm_dir: pssm directory.
    :param seq_name: the name of sequences.
    :param origin_file_name: the name of the input file in FASTA format.
    :param sw_dir: the main dir of software.
    """

    file_name, suffix = os.path.splitext(origin_file_name)

    tng_file = ''.join([file_name, '_tng', suffix])

    with open(tng_file, 'w') as f:
        for index, tng in enumerate(produce_top_n_gram(pssm_dir, seq_name, n, sw_dir)):
            f.write('>')
            f.write(seq_name[index])
            f.write('\n')
            for elem in tng:
                f.write(elem)
            f.write('\n')
    return tng_file


def tng_words(input_file, fixed_len, word_size, n, process_num, cur_dir, fixed=True):
    """Generate DT words.
    tng: By replacing all the amino acids
    in a protein with their corresponding Top-n-grams,
    a protein sequence can be represented as a sequence of
    Top-n-grams instead of a sequence of amino acids.
    """
    dir_name, seq_name = sep_file(input_file)
    sw_dir = cur_dir + '/software/'
    pssm_dir = produce_all_frequency(dir_name, sw_dir, process_num)

    tng_seq_file = convert_tng_to_fasta(pssm_dir, seq_name, input_file, n, sw_dir)

    fixed_len = fixed_len*n
    return km_words(tng_seq_file, PROTEIN, fixed_len, word_size, fixed)


def dr_words(input_file, alphabet, fixed_len, max_dis, fixed=True):
    """
    The Distance Residue method.
    :param input_file: 输入满足格式的序列文件
    :param alphabet: DNA/RNA/Protein的字母表
    :param fixed: fix or not, 默认未固定长度，但对于一些词袋模型的词，是不固定的
    :param fixed_len: fixed length for protein sequence
    :param max_dis: the value of the maximum distance.
    """
    assert int(max_dis) > 0
    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, alphabet)
    if fixed is True:
        seq_list = seq_length_fixed(seq_list, fixed_len)  # 注意这里的fixed_len对于蛋白质一般是400， RNA一般为1001
    corpus = []
    for sequence in seq_list:
        corpus_temp = []
        for i in range(fixed_len):
            corpus_temp.append(sequence[i])
            # Paper: Using distances between Top-n-gram and residue pairs for protein remote homology detection
            # 代码是基于以上文章得到，与BioSeq-Analysis2.0中的dr_method部分有差别，需要注意！
            for j in range(1, max_dis + 1):
                if i + j < fixed_len:
                    corpus_temp.append(sequence[i] + sequence[i + j])
        corpus.append(corpus_temp)

    return corpus


def dt_words(input_file, fixed_len, max_dis, process_num, cur_dir, fixed=True):
    """Generate DT words.
    DT: replacing all the amino acids in a protein with their corresponding Top-n-grams.
    """
    dir_name, seq_name = sep_file(input_file)
    sw_dir = cur_dir + '/software/'
    pssm_dir = produce_all_frequency(dir_name, sw_dir, process_num)
    tng_seq_file = convert_tng_to_fasta(pssm_dir, seq_name, input_file, 1, sw_dir)
    return dr_words(tng_seq_file, PROTEIN, fixed_len, max_dis, fixed)
