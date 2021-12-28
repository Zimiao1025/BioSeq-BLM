#!/bin/bash
ml=$1
seq_file=$2
label_file=$3
cpu=$4

cd ..
cd code/
python BioSeq-BLM_Res.py -category Protein -method One-hot -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method Binary-5bit -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method One-hot-6bit -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method Position-specific-2 -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method AESNN3 -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method PP -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method PSSM -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method PSFM -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method PAM250 -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method SS -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method SASA -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category Protein -method CS -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1

python extract_data.py Res/Protein
python extract.py Res/Protein