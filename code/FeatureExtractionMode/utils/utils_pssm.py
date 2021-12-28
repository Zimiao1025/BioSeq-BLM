#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import subprocess
import threading
from numpy import array
from .utils_const import PROTEIN
from .utils_fasta import Seq


def generate_pssm(pssm):
    """transform matrix to vector."""
    pssm_vc = []
    for i in range(len(pssm)):
        result = [float(item) for item in pssm[i][1:]]
        pssm_vc.append(result)
    return array(pssm_vc)


def is_fasta_and_protein(seq):
    """Judge if the seq is in fasta format and protein sequences.
    :param seq: Seq object
    Return True or False.
    """
    if not seq.name:
        error_info = 'Error, sequence ' + str(seq.no) + ' has no sequence name.'
        print(seq)
        sys.stderr.write(error_info)
        return False

    if 0 == seq.length:
        error_info = 'Error, sequence ' + str(seq.no) + ' is null.'
        sys.stderr.write(error_info)
        return False
    for elem in seq.seq:
        if elem not in PROTEIN and elem != 'x':
            error_info = 'Sorry, sequence ' + str(seq.no) \
                         + ' has character ' + str(elem) + '.(The character must be ' + PROTEIN + ').'
            sys.stderr.write(error_info)
            return False
    return True


def check_and_save(file_name):
    """Read the input file and store as Seq objects.
    :param file_name: the input protein sequence file.
    return  an iterator.
    """
    name, seq = '', ''
    count = 0
    with open(file_name) as f:
        for line in f:
            if not line:
                break

            if '>' == line[0]:
                if 0 != count or (0 == count and seq != ''):
                    if is_fasta_and_protein(Seq(name, seq, count)):
                        yield Seq(name, seq, count)
                    else:
                        sys.exit(0)

                seq = ''
                name = line[1:].strip()
                count += 1
            else:
                if 'x' in line or 'X' in line:
                    line = line.replace('x', '')
                    line = line.replace('X', '')
                seq += line.strip()

        # count += 1
        if is_fasta_and_protein(Seq(name, seq, count)):
            yield Seq(name, seq, count)
        else:
            sys.exit(0)


def sep_file(file_name):
    """separate the input file. One sequence in one file.
    :param file_name: the input file.
    """
    dir_name, suffix = os.path.splitext(file_name)
    # print(dir_name)  # D:\Leon\bionlp\BioSeq-NLP/data/results/Protein/sequence/OHE/SVM/PSSM/all_seq
    # print(suffix)  # .txt
    # exit()
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError:
            pass

    # 注意这里的变化，在batch模式下，节约了每次生成pssm文件的时间
    # else:
    #     rand_str = str(random.randint(0, 99999))
    #     dir_name = dir_name + '_' + rand_str
    #     os.mkdir(dir_name)
    seq_name = []
    for seq in check_and_save(file_name):
        seq_name.append(seq.name)
        seq_file = dir_name + '/' + str(seq.no) + '.txt'
        with open(seq_file, 'w') as f:
            f.write('>')
            f.write(str(seq.name))
            f.write('\n')
            f.write(str(seq.seq))
    return os.path.abspath(dir_name), seq_name


def produce_one_frequency(fasta_file, xml_file, pssm_file, sw_dir, sem):
    """Produce fequency profile for one sequence using psiblast.
    :param fasta_file: the file storing one sequence.
    :param xml_file: the generated xml file by psiblast.
    :param pssm_file: the generated pssm file by psiblast.
    :param sw_dir: the main dir of software.
    :param sem: the semaphore used for multiprocessing.
    """

    sem.acquire()
    if sys.platform.startswith('win'):
        psiblast_cmd = sw_dir + 'psiblast/psiblast.exe'
    else:
        psiblast_cmd = sw_dir + 'psiblast/psiblast'
        os.chmod(psiblast_cmd, 0o777)

    evalue_threshold = 0.001
    num_iter = 3
    outfmt_type = 5
    BLAST_DB = sw_dir + 'psiblast/nrdb90/nrdb90'
    # print('BLAST_DB:', BLAST_DB)
    cmd = ' '.join([psiblast_cmd,
                    '-query ' + fasta_file,
                    '-db ' + BLAST_DB,
                    '-out ' + xml_file,
                    '-evalue ' + str(evalue_threshold),
                    '-num_iterations ' + str(num_iter),
                    '-num_threads ' + '5',
                    '-out_ascii_pssm ' + pssm_file,
                    '-outfmt ' + str(outfmt_type)
                    ]
                   )

    subprocess.call(cmd, shell=True)
    time.sleep(2)
    sem.release()


def produce_all_frequency(pssm_path, sw_dir, process_num):
    """Produce frequency profile for all the sequences.
    :param pssm_path: the directory used to store the generated files.
    :param sw_dir: the main dir of software.
    :param process_num: the number of processes used for multiprocessing.
    """
    sequence_files = []
    for i in os.listdir(pssm_path):
        seq_full_path = ''.join([pssm_path, '/', i])
        if os.path.isfile(seq_full_path):
            sequence_files.append(seq_full_path)

    threads = []
    sem = threading.Semaphore(process_num)

    xml_dir = ''.join([pssm_path, '/xml'])
    if not os.path.isdir(xml_dir):
        os.mkdir(xml_dir)
    pssm_dir = ''.join([pssm_path, '/pssm'])
    if not os.path.isdir(pssm_dir):
        os.mkdir(pssm_dir)
    for seq_file in sequence_files:
        name = os.path.splitext(os.path.split(seq_file)[1])[0]
        xml_file = ''.join([xml_dir, '/', name, '.xml'])
        pssm_file = ''.join([pssm_dir, '/', name, '.pssm'])
        # process_list.append(mul.Process(target=produce_one_frequency,
        #                                 args=(seq_file, xml_file, pssm_file, sw_dir, semph)))
        threads.append(threading.Thread(target=produce_one_frequency,
                                        args=(seq_file, xml_file, pssm_file, sw_dir, sem)))
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    return pssm_dir
