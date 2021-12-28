import re

import numpy as np
import torch
import torch.nn.functional as func


def mega_motif2mat(motif_string):
    alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
                'P', 'S', 'T', 'W', 'Y', 'V']  # Just for protein now
    motif_mat = []
    bracket_match = re.compile(r'[\[](.*?)[\]]', re.S)  # 最小匹配，去掉问号为贪婪匹配
    frag_list = re.findall(bracket_match, motif_string)
    for frag in frag_list:
        vec_tmp = np.zeros(20, dtype=np.float32)
        for i in range(len(frag)):
            index = alphabet.index(frag[i])
            vec_tmp[index] = 1 / float(1.5 ** i)
        nom_vec = vec_tmp / np.sum(vec_tmp)
        motif_mat.append(nom_vec)
    return np.array(motif_mat)


class MotifFile2Matrix(object):
    def __init__(self, motif_file):
        self.input = motif_file

    def elm_motif_to_matrix(self):
        motifs = []
        # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
        alp1 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
                'P', 'S', 'T', 'W', 'Y', 'V']

        alp2 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
                'R', 'S', 'T', 'W', 'Y', 'V']

        with open(self.input, 'r') as f:
            lines = f.readlines()
            i = 0
            while i >= 0:
                if len(lines[i].split()) < 1:
                    i += 1
                elif lines[i].split()[0] == 'END':
                    break
                elif lines[i].split()[0] == 'MOTIF':
                    tmp = []
                    i += 3
                elif lines[i].split()[0] == 'URL':
                    motifs.append(tmp)
                    i += 2
                else:
                    tmp.append(lines[i].split())
                    i += 1
        frequency_matrices = []
        for motif in motifs:
            tmp = np.asarray(motif, dtype=np.float32)
            print('motif shape:', tmp.shape)
            for i in range(tmp.shape[0]):
                for j in range(20):
                    ti = alp1.index(alp2[j])
                    tmp[i][ti] = float(motif[i][j])
            frequency_matrices.append(tmp)
        return frequency_matrices

    def mega_motif_to_matrix(self):
        frequency_matrices = []
        with open(self.input, 'r') as f:
            lines = f.readlines()
            i = 0
            while i >= 0:
                if len(lines[i].split('	')) < 2:
                    i += 6
                elif lines[i].split('	')[0] == 'END':
                    break
                elif lines[i].split('	')[0] == 'MT':
                    motif_reg = lines[i].split(':')[1]
                    motif = mega_motif2mat(motif_reg)
                    # 删除那些长度小于3的motif
                    # if len(motif) > 3:
                    frequency_matrices.append(motif)
                    i += 1
                else:
                    i += 1
        return frequency_matrices


def motif_init(x, kernels):
    motif_out = []
    for kernel in kernels:
        # x: torch.Size([5, 100, 20])
        # print('size of kernel:', kernel.shape)  # [5, 20]
        out = x.unsqueeze(1)  # [batch_size, 1, 100, 20]
        # inputs = torch.randn(64, 3, 244, 244)
        # weight = torch.randn(64, 3, 3, 3)
        # bias = torch.randn(64)
        # outputs = func.conv2d(inputs, weight, bias)
        # print('size of mat:', out.size())  # torch.Size([5, 1, 100, 20])
        # print(outputs.size())  # torch.Size([64, 64, 242, 242])
        weight = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).double()  # 注意格式转换
        # print('size of weight:', weight.size())

        out = func.conv2d(out,
                          weight=weight)

        out = func.relu(out)
        out = func.max_pool2d(out, (2, 1))  # torch.Size([50, 1, 7, 1])
        out = out.view(out.size()[0], -1)
        out_mean = torch.mean(out, dim=1, keepdim=True)
        out_max = torch.max(out, dim=1, keepdim=True)[0]  # 0 for value; 1 for index
        out_mm = torch.cat([out_mean, out_max], 1)
        motif_out.append(out_mm)
    return torch.cat(motif_out, 1)
