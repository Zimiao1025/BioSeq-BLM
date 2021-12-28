import collections
import itertools
import os
import sys
import time
import subprocess
import threading
from xml.etree import ElementTree

import numpy as np

from ..utils.utils_pssm import check_and_save


def sep_file_psfm(parent_file, sub_file):
    dir_name = (parent_file + '/' + os.path.basename(parent_file))
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError:
            pass
    seq_name = []

    for seq in check_and_save(sub_file):
        seq_name.append(seq.name)
        seq_file = dir_name + '/' + str(seq.no) + '.txt'
        with open(seq_file, 'w') as f:
            f.write('>')
            f.write(str(seq.name))
            f.write('\n')
            f.write(str(seq.seq))
    return os.path.abspath(dir_name), seq_name


def km2index(alphabet, km):
    km_tuple = list(itertools.product(alphabet, repeat=km))
    km_string = [''.join(x) for x in km_tuple]
    km_ind = {km_string[i]: i for i in range(0, len(km_string))}
    return km_ind


def run_group_search(index_list, profile_home, sw_dir, process_num):
    threads = []
    sem = threading.Semaphore(process_num)
    for i in range(0, len(index_list)):
        seq = str(index_list[i]) + '.txt'
        threads.append(threading.Thread(target=run_simple_search,
                                        args=(seq, profile_home, sw_dir, sem)))
    for t in threads:
        t.start()

    for t in threads:
        t.join()


def run_simple_search(fd, profile_home, sw_dir, sem):
    sem.acquire()

    protein_name = fd.split('.')[0]

    complet_n = 0
    complet_n += 1
    outfmt_type = 5
    num_iter = 10
    evalue_threshold = 0.001
    fasta_file = profile_home + '/' + protein_name + '.txt'
    xml_file = profile_home + '/xml/' + protein_name + '.xml'
    pssm_file = profile_home + '/pssm/' + protein_name + '.pssm'
    msa_file = profile_home + '/msa/' + protein_name + '.msa'

    BLAST_DB = sw_dir + 'psiblast/nrdb90/nrdb90'
    if sys.platform.startswith('win'):
        psiblast_cmd = sw_dir + 'psiblast/psiblast.exe'

    else:
        psiblast_cmd = sw_dir + 'psiblast/psiblast'
        os.chmod(psiblast_cmd, 0o777)
    cmd = ' '.join([psiblast_cmd,
                    '-query ' + fasta_file,
                    '-db ' + BLAST_DB,
                    '-out ' + xml_file,
                    '-evalue ' + str(evalue_threshold),
                    '-num_iterations ' + str(num_iter),
                    '-outfmt ' + str(outfmt_type),
                    '-out_ascii_pssm ' + pssm_file,  # Write the pssm file
                    '-num_threads ' + '40']
                   )
    subprocess.call(cmd, shell=True)

    msa = []
    # parser the xml format
    tree = ElementTree.ElementTree(file=xml_file)

    # get query info
    # query_def = tree.find('BlastOutput_query-def').text
    # print query_def
    query_len = tree.find('BlastOutput_query-len').text
    # print query_len

    iteration = tree.findall('BlastOutput_iterations/Iteration')[-1]  # get the last iteration

    iteration_hits = iteration.find('Iteration_hits')

    for Hit in list(iteration_hits):
        hsp_evalue = Hit.find('Hit_hsps/Hsp/Hsp_evalue').text

        # only parser the hits that e-value < threshold
        if float(hsp_evalue) > evalue_threshold:
            continue
        # print Hsp_evalue

        # Hit_num = Hit.find('Hit_num').text
        # Hit_id = Hit.find('Hit_id').text
        # Hit_def = Hit.find('Hit_def').text

        Hsp_query_from = Hit.find('Hit_hsps/Hsp/Hsp_query-from').text
        Hsp_query_to = Hit.find('Hit_hsps/Hsp/Hsp_query-to').text
        # Hsp_hit_from = Hit.find('Hit_hsps/Hsp/Hsp_hit-from').text
        # Hsp_hit_to = Hit.find('Hit_hsps/Hsp/Hsp_hit-to').text
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
        # print hit_sequence

        # append in MSA
        msa.append(hit_sequence)

    if not msa:
        # append the protein-self
        ff = open(fasta_file, 'r')
        ff.readline()  # skip the id
        fasta_seq = ff.readline().strip().upper()
        ff.close()

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

        msa.append(fasta_seq)

    # write file
    output = open(msa_file, 'w')
    output.write('\n'.join(msa))
    output.close()
    time.sleep(2)
    sem.release()


def read_msa(msa_file):
    msa = []
    with open(msa_file) as f:
        for line in f:
            msa.append(line.strip())

    return msa


def new_print(query, pfm, protein_name, profile_home, headers):

    output_file = profile_home + '/psfm/' + protein_name + '.psfm'
    pfm = pfm.transpose()

    with open(output_file, 'w') as f:

        f.write(str(' ') + '\t')
        for item in headers:
            f.write(str(item[0]) + '\t')
        f.write('\n')

        for i in range(0, pfm.shape[0]):
            f.write(str(query[i]) + '\t')
            for j in range(0, pfm.shape[1]):
                f.write('%.6f' % float(pfm[i, j]))
                f.write('\t')

            f.write('\n')
    f.close()

    return output_file


def create_matrix(row_size, column_size):
    Matrix = np.zeros((row_size, column_size))
    return Matrix


def single_frequency_matrix(msa, km_index, kmer):
    """Count the frequency with extension methods.
    """
    # MATRIX SHAPE
    # Matrix shape is {#20+400+8000+..., #length of sequence}
    row_size = len(km_index)
    column_size = len(msa[0])
    PFM = create_matrix(row_size, column_size - kmer + 1)

    # FREQUENCY MATRIX
    for col in range(column_size):
        position_specific_composition = []
        for row in msa:
            km_slide = row[col:col + kmer]
            if '-' not in km_slide:
                position_specific_composition.append(km_slide)

        # count frequency
        position_specific_frequency = collections.Counter(position_specific_composition)
        for composition in position_specific_frequency:
            # print composition
            if composition.strip() != '':
                pssm_row = km_index[composition]
                PFM[pssm_row, col] = position_specific_frequency[composition]

    normal_pfm = create_matrix(row_size, column_size - kmer + 1)
    n = np.sum(PFM, axis=0)
    for i in range(0, PFM.shape[0]):
        for j in range(0, PFM.shape[1]):
            if n[j] == 0.0:
                normal_pfm[i, j] = 0.0
            else:
                normal_pfm[i, j] = PFM[i, j] / n[j]

    return normal_pfm


def profile_worker(fd, alphabet, k, profile_home, headers):
    protein_name = fd.split('.')[0]

    query_seq_file = profile_home + '/' + protein_name + '.txt'
    msa_file = profile_home + '/msa/' + protein_name + '.msa'
    msa_ret = read_msa(msa_file)

    temp_len = len(msa_ret[0])

    f = open(query_seq_file, 'r')
    next(f)

    query = []
    for line in f:
        query = list(line)

    # generate single frequency  ------------------------------------
    km_index = km2index(alphabet, k)
    row_size = len(km_index)
    column_size = temp_len
    pfm = create_matrix(row_size, column_size)
    if msa_ret:
        pfm = single_frequency_matrix(msa_ret, km_index, k)
        assert pfm.shape[0] == len(km_index)
    return new_print(query, pfm, protein_name, profile_home, headers)
    # generate single frequency  ------------------------------------
