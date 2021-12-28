import numpy as np

from .ei import EvolutionaryInformation2Vectors
from .pp import PhyChemicalProperty2vectors
from .rc import ResidueComposition2Vectors
from .sc import rss_method, ss_method, sasa_method, cs_method
from ..utils.utils_write import vectors2files, dl_vec2file, write_res_base_vec, res_vectors2file


def one_hot_enc(input_file, alphabet, enc_method, cur_dir, chosen_file, rss_file, cpu):
    # return_mat: 对于one-hot方法返回向量， 对于Automatic features和深度学习模型返回矩阵
    # TODO: ResidueComposition
    if enc_method == 'One-hot':
        vec_mat_list = ResidueComposition2Vectors(alphabet).one_hot(input_file)
    elif enc_method == 'One-hot-6bit':
        vec_mat_list = ResidueComposition2Vectors(alphabet).one_hot_six_bits(input_file)
    elif enc_method == 'Binary-5bit':
        vec_mat_list = ResidueComposition2Vectors(alphabet).one_hot_five(input_file)
    elif enc_method == 'DBE':
        vec_mat_list = ResidueComposition2Vectors(alphabet).dbe(input_file)
    elif enc_method == 'Position-specific-2':
        vec_mat_list = ResidueComposition2Vectors(alphabet).position_specific(2, input_file)
    elif enc_method == 'Position-specific-3':
        vec_mat_list = ResidueComposition2Vectors(alphabet).position_specific(3, input_file)
    elif enc_method == 'Position-specific-4':
        vec_mat_list = ResidueComposition2Vectors(alphabet).position_specific(4, input_file)
    elif enc_method == 'AESNN3':
        vec_mat_list = ResidueComposition2Vectors(alphabet).aesnn3(input_file)
    elif enc_method == 'NCP':
        vec_mat_list = ResidueComposition2Vectors(alphabet).ncp(input_file)

    # TODO: PhyChemicalProperty
    elif enc_method == 'DPC':
        vec_mat_list = PhyChemicalProperty2vectors(enc_method, alphabet, chosen_file).dpc(input_file)
    elif enc_method == 'TPC':
        vec_mat_list = PhyChemicalProperty2vectors(enc_method, alphabet, chosen_file).tpc(input_file)
    elif enc_method == 'PP':
        vec_mat_list = PhyChemicalProperty2vectors(enc_method, alphabet, chosen_file).pp(input_file)

    # TODO: EvolutionaryInformation
    elif enc_method == 'BLAST-matrix':
        vec_mat_list = EvolutionaryInformation2Vectors(alphabet, cur_dir).blast_matrix(input_file)
    elif enc_method == 'PAM250':
        vec_mat_list = EvolutionaryInformation2Vectors(alphabet, cur_dir).pam250(input_file)
    elif enc_method == 'BLOSUM62':
        vec_mat_list = EvolutionaryInformation2Vectors(alphabet, cur_dir).blosum62(input_file)
    elif enc_method == 'PSSM':
        vec_mat_list = EvolutionaryInformation2Vectors(alphabet, cur_dir).pssm(input_file, cpu)
    elif enc_method == 'PSFM':
        vec_mat_list = EvolutionaryInformation2Vectors(alphabet, cur_dir).psfm(input_file, cpu)

    # TODO: SecondStructure and ConservationScore
    # mat_return 后期需要修改
    elif enc_method == 'RSS':
        vec_mat_list = rss_method(rss_file)
    elif enc_method == 'SS':
        vec_mat_list = ss_method(input_file, cur_dir, cpu)
    elif enc_method == 'SASA':
        vec_mat_list = sasa_method(input_file, cur_dir, cpu)
    elif enc_method == 'CS':
        vec_mat_list = cs_method(input_file, cur_dir, cpu)
    else:
        print('Method for One-hot encoding mode error!')
        return False
    return vec_mat_list


def ohe2seq_vec(input_file, alphabet, enc_method, cur_dir, chosen_file, rss_file, sp_num_list, fixed_len, out_format,
                out_files, cpu):
    vec_mat_list = one_hot_enc(input_file, alphabet, enc_method, cur_dir, chosen_file, rss_file, cpu)
    ohe_array = mat_list2mat_array(vec_mat_list, fixed_len)
    vectors2files(ohe_array, sp_num_list, out_format, out_files)


def ohe2seq_mat(input_file, alphabet, enc_method, cur_dir, chosen_file, rss_file, sp_num_list, out_files, cpu):
    vec_mat_list = one_hot_enc(input_file, alphabet, enc_method, cur_dir, chosen_file, rss_file, cpu)
    dl_vec2file(vec_mat_list, sp_num_list, out_files)


def ohe2res_base(input_file, alphabet, enc_method, cur_dir, chosen_file, rss_file, out_file, cpu):
    vec_mat_list = one_hot_enc(input_file, alphabet, enc_method, cur_dir, chosen_file, rss_file, cpu)
    write_res_base_vec(vec_mat_list, out_file)


def mat_list2mat_array(mat_list, fixed_len):
    mat_array = []
    try:
        width = mat_list[0].shape[1]
    except IndexError:
        width = 1
    for i in range(len(mat_list)):
        temp_arr = np.zeros((fixed_len, width))
        temp_len = mat_list[i].shape[0]
        if temp_len <= fixed_len:
            # temp_arr[:temp_len, :] = mat_list[i]
            try:
                temp_arr[:temp_len, :] = mat_list[i]
            except ValueError:
                temp_arr[:temp_len, 0] = mat_list[i]
        else:
            temp_arr = mat_list[i][:fixed_len, :]

        mat_array.append(temp_arr.flatten().tolist())
    mat_array = np.array(mat_array)
    return mat_array


def sliding_win2files(res_mats, res_labels_list, win_size, out_format, out_files):
    width = res_mats[0].shape[1]
    win_ctrl = win_size // 2
    pos_vec = []
    neg_vec = []
    for i in range(len(res_mats)):
        res_mat = res_mats[i]
        seq_len = len(res_mat)
        assert seq_len > win_size, "The size of window should be no more than the length of sequence[%d]." % i
        # print('seq_len: %d' % seq_len)
        for j in range(seq_len):
            temp_mat = np.zeros((win_size, width))
            if j <= win_ctrl:
                # print(j+win_ctrl+1)
                temp_mat[:j+win_ctrl+1, :] = res_mat[: j+win_ctrl+1, :]
            elif j >= seq_len - win_ctrl:
                temp_mat[:seq_len-j+win_ctrl, :] = res_mat[j-win_ctrl: seq_len, :]
            else:
                temp_mat[:, :] = res_mat[j-win_ctrl: j+win_ctrl+1]
            temp_vec = temp_mat.flatten().tolist()
            if res_labels_list[i][j] == 0:
                neg_vec.append(temp_vec)
            else:
                pos_vec.append(temp_vec)
    print('The output files can be found here:')
    res_vectors2file(np.array(pos_vec), out_format, out_files[0])
    res_vectors2file(np.array(neg_vec), out_format, out_files[1])
    print('\n')


def mat_list2frag_array(mat_list, res_labels_list, fixed_len, out_format, out_files):
    frag_array = []
    width = mat_list[0].shape[1]
    for i in range(len(mat_list)):
        temp_arr = np.zeros((fixed_len, width))
        temp_len = mat_list[i].shape[0]
        if temp_len <= fixed_len:
            temp_arr[:temp_len, :] = mat_list[i]
        else:
            temp_arr = mat_list[i][:fixed_len, :]

        frag_array.append(temp_arr.flatten().tolist())

    pos_num = 0
    neg_num = 0
    pos_vec = []
    neg_vec = []
    for i in range(len(res_labels_list)):
        if int(res_labels_list[i][0]) == 1:
            pos_num += 1
            pos_vec.append(frag_array[i])
        else:
            neg_num += 1
            neg_vec.append(frag_array[i])
    assert pos_num + neg_num == len(res_labels_list), 'Please check label file, there contains ont only two labels!'

    print('The output files can be found here:')
    res_vectors2file(np.array(pos_vec), out_format, out_files[0])
    res_vectors2file(np.array(neg_vec), out_format, out_files[1])
    print('\n')
