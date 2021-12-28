#!/bin/bash
ml=$1
seq_file=$2
label_file=$3
cpu=$4

cd ..
cd code/
python BioSeq-BLM_Res.py -category DNA -method One-hot -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category DNA -method Binary-5bit -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category DNA -method One-hot-6bit -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category DNA -method Position-specific-2 -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category DNA -method Position-specific-3 -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category DNA -method Position-specific-4 -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category DNA -method DBE -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category DNA -method DPC -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category DNA -method TPC -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1
python BioSeq-BLM_Res.py -category DNA -method BLAST-matrix -ml ${ml} -seq_file ${seq_file} -label_file ${label_file} -cpu ${cpu} -bp 1

python extract_data.py Res/DNA
python extract.py Res/DNA