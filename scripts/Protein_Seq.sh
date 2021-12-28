#!/bin/bash
score=$1
ml=$2
seq_files=$3
labels=$4
cpu=$5

cd ..
cd code/
python BioSeq-BLM_Seq.py -category Protein -mode OHE -method One-hot -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category Protein -mode SR -method AC -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category Protein -mode BOW -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category Protein -mode TF-IDF -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category Protein -mode TR -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category Protein -mode TM -method LSA -in_tm BOW -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category Protein -mode WE -method word2vec -words Kmer -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1
python BioSeq-BLM_Seq.py -category Protein -mode AF -method word2vec -in_af One-hot -score ${score} -auto_opt 2 -ml ${ml} -grid 1 -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1

python extract_data.py Seq/Protein
python extract.py Seq/Protein