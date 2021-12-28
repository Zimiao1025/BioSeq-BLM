import numpy as np
import torch
from torch.utils.data import DataLoader
from ..utils.utils_motif import MotifFile2Matrix, motif_init
from ..OHE.ei import EvolutionaryInformation2Vectors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多


def mat_list2mat(mat_list, fixed_len):
    mat_array = []
    width = mat_list[0].shape[1]
    for i in range(len(mat_list)):
        temp_arr = np.zeros((fixed_len, width))
        temp_len = mat_list[i].shape[0]
        if temp_len <= fixed_len:
            temp_arr[:temp_len, :] = mat_list[i]
        else:
            temp_arr = mat_list[i][:fixed_len, :]

        mat_array.append(temp_arr)
    mat_array = np.array(mat_array)
    return mat_array


def motif_pssm(input_file, alphabet, process_num, batch_size, motif_file, motif_database, fixed_len, cur_dir):
    # all_data = EvolutionaryInformation2Vectors(alphabet, fixed_len, cur_dir, True,
    #                                            mat=True).pssm(input_file, process_num)

    vec_mat_list = EvolutionaryInformation2Vectors(alphabet, cur_dir).pssm(input_file, process_num)
    all_data = mat_list2mat(vec_mat_list, fixed_len)

    data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False)
    if motif_database == 'Mega':
        motifs = MotifFile2Matrix(motif_file).mega_motif_to_matrix()
    else:
        motifs = MotifFile2Matrix(motif_file).elm_motif_to_matrix()
    # print(all_data.shape)  # (20, 100, 20)

    motif_features = []
    with torch.no_grad():
        for mat in data_loader:
            # print(mat.size())  # torch.Size([5, 100, 20]) [batch_size, seq_len,

            tensor = mat.to(DEVICE)

            batch_motif_feature = motif_init(tensor, motifs)
            motif_features += batch_motif_feature.tolist()
    return np.array(motif_features)
