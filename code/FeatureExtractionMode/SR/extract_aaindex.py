__author__ = 'Fule Liu'

import math
import os
import pickle


class AAIndex:
    def __init__(self, head, index_dict):
        self.head = head
        self.index_dict = index_dict

    def __str__(self):
        return "%s\n%s" % (self.head, self.index_dict)


def extra_aaindex(filename):
    """Return AAIndex obj list.
    """
    index_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    aaindex = []
    with open(filename, 'r') as f:
        temp_h = ""
        lines = f.readlines()
        for ind, line in enumerate(lines):
            if line[0] == 'H':
                temp_h = line[2:].rstrip()
            elif line[0] == 'I':
                vals = lines[ind+1].rstrip().split() + lines[ind+2].rstrip().split()
                index_val = {}
                try:
                    index_val = {index: float(val) for index, val in zip(index_list, vals)}
                except:
                    _sum = sum([float(val) for val in vals if val != 'NA'])
                    for ind, val in enumerate(vals):
                        if val != 'NA':
                            index_val[index_list[ind]] = float(vals[ind])
                        else:
                            index_val[index_list[ind]] = round(_sum / 20, 3)
                            # print(temp_h, vals)
                aaindex.append(AAIndex(temp_h, index_val))

    return aaindex


def norm_index_vals(index_vals):
    """Normalize index_vals.

    :param index_vals: dict, {index: vals}.
    """
    _norm_index_vals = {}

    avg = float(sum(index_vals.values())) / 20
    for index, val in list(index_vals.items()):
        numerator = val - avg
        denominator = math.sqrt((sum([pow(temp_val - avg, 2) for temp_val in list(index_vals.values())]) / 20))
        _norm_index_vals[index] = round(numerator / denominator, 2)

    return _norm_index_vals


def write_aaindex(aaindex, filename):
    with open(filename, 'wb') as f:
        pickle.dump(aaindex, f, protocol=2)


if __name__ == '__main__':
    h1 = {'A': 0.620, 'R': -2.530, 'N': -0.780, 'D': -0.090, 'C': 0.290, 'Q': -0.850, 'E': -0.740, 'G': 0.480,
          'H': -0.400, 'I': 1.380, 'L': 1.530, 'K': -1.500, 'M': 0.640, 'F': 1.190, 'P': 0.120, 'S': -0.180,
          'T': -0.050, 'W': 0.810, 'Y': 0.260, 'V': 1.800}
    h2 = {'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0, 'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
          'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0, 'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5}
    m = {'A': 71.079, 'R': 156.188, 'N': 114.104, 'D': 115.086, 'C': 103.145, 'Q': 128.131, 'E': 129.116, 'G': 57.0521,
         'H': 137.141, 'I': 113.160, 'L': 113.160, 'K': 128.170, 'M': 131.99, 'F': 147.177, 'P': 97.177, 'S': 87.078,
         'T': 101.105, 'W': 186.123, 'Y': 163.176, 'V': 99.133}

    file_path = os.path.abspath('..') + "/data/aaindex3.txt"
    print(file_path)
    aaindex = extra_aaindex(file_path)
    aaindex.extend([AAIndex('Hydrophobicity', h1), AAIndex('Hydrophilicity', h2), AAIndex('Mass', m)])

    for ind, e in enumerate(aaindex):
        aaindex[ind] = AAIndex(e.head, norm_index_vals(e.index_dict))

    for e in aaindex:
        if e.head == 'Hydrophobicity':
            print((e.index_dict))

    file_path = os.path.abspath('..') + "/data/aaindex.data"
    write_aaindex(aaindex, file_path)

    file_path = os.path.abspath('..') + "/data/aaindex.data"
    with open(file_path, 'rb') as f:
        norm_aaindex = pickle.load(f)
    print('\n')

    heads = [e.head for e in norm_aaindex]
    # print(heads)

    print((len(norm_aaindex)))

    norm_h1 = norm_index_vals(h1)
    norm_h2 = norm_index_vals(h2)
    norm_m = norm_index_vals(m)