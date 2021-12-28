#!/bin/bash
score=$1
ml=$2
seq_files=$3
labels=$4
cpu=$5

cd ..
cd code/
python BioSeq-BLM_Seq.py -category DNA -mode OHE -method One-hot -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category DNA -mode SR -method DAC -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category DNA -mode BOW -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category DNA -mode TF-IDF -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category DNA -mode TR -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category DNA -mode TM -method LSA -in_tm BOW -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category DNA -mode WE -method word2vec -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category DNA -mode AF -method word2vec -in_af One-hot -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1

python extract_data.py Seq/DNA
python extract.py Seq/DNA