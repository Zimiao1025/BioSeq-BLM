import os
import sys
import time
import subprocess
import threading

import numpy as np
from xml.etree import ElementTree

from ..SR.profile import get_blosum62
from ..utils.utils_pssm import produce_all_frequency, sep_file


def generate_bracket_seq(receive_file_path, bracket_file_path):
    """ This is a system command to generate bracket_seq file according receive_file. """

    cmd = "RNAfold <" + receive_file_path + " >" + bracket_file_path + ' --noPS'
    subprocess.Popen(cmd, shell=True).wait()


def rss_method(input_file):

    dir_name, suffix = os.path.splitext(input_file)
    bracket_file = dir_name + '_bracket' + suffix
    print("bracket_file: ", bracket_file)
    generate_bracket_seq(input_file, bracket_file)

    vectors = []
    with open(bracket_file) as r:
        line = r.readlines()
        for i in range(1, len(line), 3):
            num = len(line[i]) - 1
            sc = line[i + 1][:num]

            sc_fe = ''
            for j in sc:
                if j == '.':
                    sc_fe += '0 '  # keep the space
                else:
                    sc_fe += '1 '
            vec_temp = list(map(float, sc_fe.split()))  # RSS特征的维度等于序列长度
            vector = []
            for k in range(len(vec_temp)):
                vector.append(vec_temp[k])
            vectors.append(np.array(vector))
    return vectors


# SS starts
def ss_method(input_file, cur_dir, process_num):
    sw_dir = cur_dir + '/software/'

    dir_name, seq_name = sep_file(input_file)

    threads = []
    file_path_list = []
    sem = threading.Semaphore(process_num)

    for parent, dir_names, file_names in os.walk(dir_name):
        for filename in file_names:
            file_path = os.path.join(parent, filename)
            print('file_path: ', file_path)
            file_path_list.append(file_path)
            threads.append(threading.Thread(target=gen_ss_vector,
                                            args=(file_path, sw_dir, sem)))
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    ss_mat_dict = {'C': [0, 0, 1], 'E': [0, 1, 0], 'H': [1, 0, 0]}

    for temp_file in file_path_list:
        file_name = str(temp_file).split('/')[-1].split('.')[0]
        ss_file = os.getcwd() + '/' + file_name + '.ss'
        print(ss_file)

        with open(ss_file) as f:
            ss = [line[7] for line in f.readlines()]

        vector = []
        for i in range(len(ss)):
            vector.append(ss_mat_dict[ss[i]])

        os.remove(ss_file)
        os.remove(ss_file[:-2] + 'ss2')
        os.remove(ss_file[:-2] + 'horiz')
        with open(temp_file) as f:
            lines = f.readlines()
            seq_name[seq_name.index(lines[0].strip()[1:])] = np.array(vector)
    ss_vector = seq_name
    return ss_vector


def gen_ss_vector(temp_file, sw_dir, sem):
    sem.acquire()
    # 只支持Linux系统
    if sys.platform.startswith('win'):
        error_info = 'The SS method for One-hot encoding mode only support Linux/Unix system!'
        sys.stderr.write(error_info)
        return False

    else:
        os.chdir(sw_dir)
        cmd = sw_dir + 'psipred/runpsipred_single ' + temp_file
        subprocess.call(cmd, shell=True)
    time.sleep(2)
    sem.release()


# SS ends

def sasa_method(input_file, cur_dir, process_num):

    sw_dir = cur_dir + '/software/'

    pssm_path, seq_name = sep_file(input_file)
    pssm_dir = produce_all_frequency(pssm_path, sw_dir, process_num)
    # 调试模式 on/off
    # pssm_dir = cur_dir + "/data/results/Protein/sequence/OHE/SVM/SASA/all_seq/pssm"

    dir_name = os.path.split(pssm_dir)[0]

    xml_dir = dir_name + '/xml'

    final_result = ''.join([dir_name, '/final_result'])

    if not os.path.isdir(final_result):
        os.mkdir(final_result)

    dir_list = os.listdir(xml_dir)

    threads = []
    sem = threading.Semaphore(process_num)

    index_list = []
    for elem in dir_list:
        xml_full_path = ''.join([xml_dir, '/', elem])
        name, suffix = os.path.splitext(elem)

        if os.path.isfile(xml_full_path) and suffix == '.xml':
            index_list.append(int(name))

    index_list.sort()
    # index_list: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    spd3_file_list = []
    for index in index_list:
        pssm_file = pssm_dir + '/' + str(index) + '.pssm'
        pssm_file_list = os.path.split(pssm_file)
        spd3_file = os.path.split(pssm_file_list[0])[0] + '/' + pssm_file_list[1].split('.')[0] + '.spd3'
        spd3_file_list.append(spd3_file)
        threads.append(threading.Thread(target=gen_sa_vec,
                                        args=(pssm_file, dir_name, sw_dir, sem)))
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    sasa_vectors = []
    for spd3_file in spd3_file_list:
        with open(spd3_file) as f:
            lines = f.readlines()
            vec_temp = [float(line.strip().split()[3]) for line in lines[1:]]

            vector = []
            for i in range(len(vec_temp)):
                vector.append(vec_temp[i])

        sasa_vectors.append(np.array(vector))
    return np.array(sasa_vectors)


def gen_sa_vec(pssm_file, dir_name, sw_dir, sem):
    sem.acquire()

    if not os.path.exists(pssm_file):
        seq_name = os.path.basename(pssm_file).split('.')[0]
        seq_path = dir_name + '/' + seq_name + '.txt'
        with open(seq_path) as f, open(pssm_file, 'w') as w:
            lines = f.readlines()
            protein_seq = lines[1].strip().upper()
            pssm = get_blosum62(protein_seq)
            pssm = np.array(pssm)

            protein_seq = [np.array([1, x]) for x in list(protein_seq)]
            protein_seq = np.array(protein_seq)
            pssm_lists = np.hstack((protein_seq, pssm)).tolist()
            for pssm_list in pssm_lists:
                w.writelines('\t'.join(pssm_list) + '\n')

    pwd = os.getcwd()
    os.chdir(dir_name)
    cmd = 'python ' + sw_dir + 'SPIDER2_local/misc/pred_pssm.py ' + pssm_file
    subprocess.call(cmd, shell=True)
    os.chdir(pwd)

    time.sleep(2)
    sem.release()


def cs_method(input_file, cur_dir, process_num):

    sw_dir = cur_dir + '/software/'
    dirname, seq_name = sep_file(input_file)
    pssm_dir = produce_all_frequency(dirname, sw_dir, process_num)

    dir_name = os.path.split(pssm_dir)[0]

    xml_dir = dir_name + '/xml'
    final_result = ''.join([dir_name, '/final_result'])
    if not os.path.isdir(final_result):
        os.mkdir(final_result)

    dir_list = os.listdir(xml_dir)
    index_list = []
    for elem in dir_list:
        xml_full_path = ''.join([xml_dir, '/', elem])
        name, suffix = os.path.splitext(elem)
        if os.path.isfile(xml_full_path) and suffix == '.xml':
            index_list.append(int(name))

    index_list.sort()

    threads = []
    sem = threading.Semaphore(process_num)

    ec_vector = []
    cs_file_list = []
    for index in index_list:
        fasta_file = dir_name + '/' + str(index) + '.fasta'
        cs_file = os.path.splitext(fasta_file)[0] + '_cs.txt'
        cs_file_list.append(cs_file)
        threads.append(threading.Thread(target=gen_ec_vector,
                                        args=(xml_dir, index, dir_name, sw_dir, sem)))
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    for cs_file in cs_file_list:
        with open(cs_file) as f:
            lines = f.readlines()
            if sys.platform.startswith('win'):
                vec_temp = [0.0000 if line.strip().split()[2] == '-nan' else float(line.strip().split()[2]) for line in
                            lines if line[0].isdigit()]
            else:
                vec_temp = [0.0000 if line.strip().split()[2] == '-nan' else float(line.strip().split()[2]) for line in
                            lines if line[0] == ' ']

        ec_vector.append(np.array(vec_temp))

    return np.array(ec_vector)


def gen_ec_vector(xml_dir, index, dir_name, sw_dir, sem):
    sem.acquire()

    xml_full_path = xml_dir + '/' + str(index) + '.xml'
    txt_full_path = dir_name + '/' + str(index) + '.txt'
    msa_file = get_msa(xml_full_path, txt_full_path)

    fasta_file = dir_name + '/' + str(index) + '.fasta'

    with open(msa_file) as f, open(fasta_file, 'w') as w:
        lines = f.readlines()
        c = ''
        for i, line in enumerate(lines):
            if i < 290:
                c += '>' + str(i) + '\n' + line
        w.writelines(c)

    cs_file = os.path.splitext(fasta_file)[0] + '_cs.txt'
    if sys.platform.startswith('win'):
        cmd = sw_dir + 'psiblast/rate4site_slow.exe -s ' + fasta_file + ' -o ' + cs_file
        subprocess.call(cmd, shell=True)
    else:
        cmd = sw_dir + 'psiblast/rate4site -s ' + fasta_file + ' -o ' + cs_file
        subprocess.call(cmd, shell=True)
    time.sleep(2)
    sem.release()


def get_msa(xml_full_path, file_path):
    MSA = []
    evalue_threshold = 0.0001
    tree = ElementTree.ElementTree(file=xml_full_path)

    query_len = tree.find('BlastOutput_query-len').text

    iteration = tree.findall('BlastOutput_iterations/Iteration')[-1]  # get the last iteration
    Iteration_hits = iteration.find('Iteration_hits')
    for Hit in list(Iteration_hits):
        Hsp_evalue = Hit.find('Hit_hsps/Hsp/Hsp_evalue').text
        # only parser the hits that e-value < threshold
        if float(Hsp_evalue) > evalue_threshold:
            continue

        Hsp_query_from = Hit.find('Hit_hsps/Hsp/Hsp_query-from').text
        Hsp_query_to = Hit.find('Hit_hsps/Hsp/Hsp_query-to').text

        Hsp_qseq = Hit.find('Hit_hsps/Hsp/Hsp_qseq').text
        Hsp_hseq = Hit.find('Hit_hsps/Hsp/Hsp_hseq').text

        # alignment sequence by add prefix, suffix
        prefix = "-" * (int(Hsp_query_from) - 1)
        suffix = "-" * (int(query_len) - int(Hsp_query_to))
        # delete the space in protein_name and the corresponding position of hits
        pos = -1
        for aa in Hsp_qseq:
            pos = pos + 1
            if aa == '-':
                Hsp_hseq = Hsp_hseq[:pos] + '*' + Hsp_hseq[pos + 1:]
        Hsp_hseq = Hsp_hseq.replace('*', '')

        if 'X' in Hsp_hseq:
            Hsp_hseq = Hsp_hseq.replace('X', '-')
        if 'B' in Hsp_hseq:
            Hsp_hseq = Hsp_hseq.replace('B', '-')
        if 'Z' in Hsp_hseq:
            Hsp_hseq = Hsp_hseq.replace('Z', '-')
        if 'U' in Hsp_hseq:
            Hsp_hseq = Hsp_hseq.replace('U', '-')
        if 'J' in Hsp_hseq:
            Hsp_hseq = Hsp_hseq.replace('J', '-')
        if 'O' in Hsp_hseq:
            Hsp_hseq = Hsp_hseq.replace('O', '-')
        # combine prefix, modified hits, suffix
        hit_sequence = prefix + Hsp_hseq + suffix
        # append in MSA
        MSA.append(hit_sequence)
    ff = open(file_path, 'r')
    ff.readline()  # skip the id
    fasta_seq = ff.readline().strip().upper()
    ff.close()
    if not MSA:
        # append the protein-self
        if 'X' in fasta_seq:
            fasta_seq = fasta_seq.replace('X', '-')
        if 'B' in fasta_seq:
            fasta_seq = fasta_seq.replace('B', '-')
        if 'Z' in fasta_seq:
            fasta_seq = fasta_seq.replace('Z', '-')
        if 'U' in fasta_seq:
            fasta_seq = fasta_seq.replace('U', '-')
        if 'J' in fasta_seq:
            fasta_seq = fasta_seq.replace('J', '-')
        if 'O' in fasta_seq:
            fasta_seq = fasta_seq.replace('O', '-')
        MSA.append(fasta_seq)

    if len(MSA) == 1:
        MSA.append(MSA[0])
    if MSA[0] != fasta_seq:
        if fasta_seq in MSA:
            index_seq = MSA.index(fasta_seq)
            MSA[0], MSA[index_seq] = MSA[index_seq], MSA[0]
        else:
            MSA.insert(0, fasta_seq)
    # write file
    msa_file = os.path.splitext(file_path)[0] + '.msa'
    output = open(msa_file, 'w')
    output.write('\n'.join(MSA))
    output.close()
    return msa_file
