from collections import OrderedDict
import numpy as np
import os


class PhyChemicalProperty2vectors(object):
    # TODO: 在模型的初始化函数中定义模型要用到的变量
    def __init__(self, method, alphabet, chosen_file=None):
        """
        Initialize the object.
        :param alphabet: DNA, RNA or Protein
        """
        full_path = os.path.realpath(__file__)
        self.pp_dir = os.path.dirname(full_path) + '/data/'

        if alphabet == 'DNA':
            self.indicators = self.pp_dir + 'DDi_index.txt'
            self.indicators_name = self.pp_dir + 'DDi_name.txt'
            # sum = 90  # total number of the DNA indicators
        elif alphabet == 'RNA':
            self.indicators = self.pp_dir + 'RDi_index.txt'
            self.indicators_name = self.pp_dir + 'RDi_name.txt'
            # sum = 11  # total number of the RNA indicators
        else:
            self.indicators = self.pp_dir + 'aaindex.txt'
            self.indicators_name = self.pp_dir + 'Phy_HeadList.txt'
            # sum = ?  # total number of the Protein indicators

        if chosen_file is None:
            print('\nThe pp_file is None, select default physicochemical properties.')
            if method == 'DPC':
                self.chosen_name = ['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise']
            elif method == 'TPC':
                self.chosen_name = ['Dnase I', 'Nucleosome positioning']
            elif method == 'PP':
                self.chosen_name = ['Hydrophobicity', 'Hydrophilicity', 'Mass']
        else:
            print('\nThe pp_file is: %s' % chosen_file)
            with open(chosen_file) as r:
                self.chosen_name = [i.replace('\r', '') for i in r.read().split('\n')]

        self.alphabet = alphabet
        self.aaList_Index = []
        self.vec_mat_list = []

    def dpc(self, file_path):
        indicators_value = OrderedDict()
        with open(self.indicators) as f:
            line = f.readlines()
            for i in range(32):
                if i % 2 == 0:
                    indicators_value[line[i].strip()] = ''
                else:
                    indicators_value[line[i - 1].strip()] = line[i].strip().split()
        print('The physicochemical properties file is %s\n' % self.indicators_name)
        with open(self.indicators_name) as r:
            indicators_list = [i.replace('\r', '') for i in r.read().split('\n')]
        # chosen_index = [indicators_list.index(i) for i in self.chosen_name]
        chosen_index = [indicators_list.index(i) for i in self.chosen_name if i != '']
        with open(file_path) as f:
            for line in f:
                if line[0] != '>':
                    vec_mat = []
                    line = line.strip().upper()
                    for n in range(len(line) - 1):
                        i = line[n] + line[n + 1]
                        vector = []
                        for j in chosen_index:
                            vector.append(indicators_value[i][j])
                        vector = list(map(float, vector))
                        vec_mat.append(vector)
                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def tpc(self, file_path):
        assert self.alphabet == 'DNA', 'TPC method is only for DNA sequence!'
        indicators_value = OrderedDict()
        with open(self.pp_dir + 'DTi_index.txt') as f:
            line = f.readlines()
            for i in range(128):
                if i % 2 == 0:
                    indicators_value[line[i].strip()] = ''
                else:
                    indicators_value[line[i - 1].strip()] = line[i].strip().split()
        print('The physicochemical properties file is %s\n' % (self.pp_dir + 'DTi_name.txt'))
        with open(self.pp_dir + 'DTi_name.txt') as r:
            indicators_list = [i.replace('\r', '') for i in r.read().split('\n')]

        # chosen_index = [indicators_list.index(i) for i in self.chosen_name]
        chosen_index = [indicators_list.index(i) for i in self.chosen_name if i != '']
        with open(file_path) as f:
            for line in f:
                if line[0] != '>':
                    vec_mat = []
                    line = line.strip().upper()
                    for n in range(len(line) - 1):
                        i = line[n] + line[n + 1]
                        vector = []
                        for j in chosen_index:
                            vector.append(indicators_value[i][j])
                        vector = list(map(float, vector))
                        vec_mat.append(vector)
                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list

    def pp(self, file_path):
        indicators_value = OrderedDict()
        with open(self.indicators) as f:
            line = f.readlines()
            for i in range(40):
                if i % 2 == 0:
                    indicators_value[line[i].strip()] = ''
                else:
                    indicators_value[line[i - 1].strip()] = line[i].strip().split()
        print('The physicochemical properties file is %s\n' % self.indicators_name)
        with open(self.indicators_name) as r:
            indicators_list = [i.replace('\r', '') for i in r.read().split('\n')]
        # chosen_index = [indicators_list.index(i) for i in self.chosen_name]
        chosen_index = [indicators_list.index(i) for i in self.chosen_name if i != '']
        with open(file_path) as lines:
            for line in lines:
                if line[0] != '>':
                    vec_mat = []
                    line = line.strip().upper()
                    for i in range(len(line)):
                        vector = []
                        for j in chosen_index:
                            vector.append(indicators_value[line[i]][j])
                        vector = list(map(float, vector))
                        vec_mat.append(vector)
                    self.vec_mat_list.append(np.array(vec_mat))
        return self.vec_mat_list
