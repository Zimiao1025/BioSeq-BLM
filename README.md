# BioSeq-BLM: a platform for analyzing DNA, RNA, and protein sequences based on biological language models

System Requirements
----

**Software Requirements:**

* [Python3](https://docs.python-guide.org/starting/install3/linux/)
* [virtualenv](https://virtualenv.pypa.io/en/latest/installation/) or [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) (Optional If using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional If using GPU)

BioSeq-BLM has been tested on Windows, Ubuntu 16.04, and 18.04 operating systems.

## Installation

### virtualenv

```shell
virtualenv -p python3.7 venv

source ./venv/bin/activate

pip install -r requirements.txt
```

### Anaconda

```shell
conda create -n venv python=3.7

conda activate venv

pip install -r requirements.txt
```

#### Not Necessary Softwares

- [BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download)
- [PSIPRED](http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/)
- [SPIDER2](https://sparks-lab.org/downloads/)
- [ViennaRNA](https://www.tbi.univie.ac.at/RNA/ )
- [rate4site](https://www.tau.ac.il/~itaymay/cp/rate4site.html)

## Usage and examples

### Directory Structure Description

```tex
BioSeq-BLM
├───code // python source code for stand-alone package.
│
├───data // Used to place example datasets.
│
├───docs // Manual about stand-alone package.
│
├───results // After running the code, the output results can be found here.
|
├───scripts // Used to place the scripts selecting the best algorithms automatically.
|
├───software // Used to place the "Not Necessary Softwares" in installation.
|
|───LICENSE // the license.
│
|───README.md // repository description.
|
└───requirements.txt // Necessary file for installation.
```

### Examples

> 1. Download the datasets from the [BioSeq-BLM (Download)](http://bliulab.net/BioSeq-BLM/download/#dataset), unzip them and put them in the '/data' folder.
>
> 2. Enter the '/code' directory and run the following command lines.

#### 1 Identification DNase I hypersensitive sites

```shell
python BioSeq-BLM_Seq.py -category DNA -mode TF-IDF -words Mismatch -word_size 4 -cl Kmeans -nc 5 -dr PCA -np 64 -fs F-value -nf 128 -rdb fs -ml SVM -cost 4 -gamma -1 -sp combine -seq_file ../data/1-DHSs/dna_pos.txt ../data/1-DHSs/dna_neg.txt -label +1 -1
```

#### 2 Identification of real microRNA precursors

```shell
python BioSeq-BLM_Seq.py -category RNA -mode OHE -method RSS -cl Kmeans -nc 5 -fs MIC -nf 128 -dr TSVD -np 128 -rdb dr -ml SVM -cost 1 -gamma -4 -seq_file ../data/2-miRNA/rna_pos.txt ../data/2-miRNA/rna_neg.txt -rss_file ../data/2-miRNA/rna_with_2rd_structure.txt -label +1 -1
```

#### 3 Identification of DNA binding proteins

```shell
python BioSeq-BLM_Seq.py -category Protein -mode TM -method LSA -in_tm BOW -words Top-N-Gram -top_n 2 -com_prop 0.7 -sn L1-normalize -cl Kmeans -nc 5 -fs Tree -nf 128 -dr KernelPCA -np 128 -rdb dr -ml RF -seq_file ../data/3-DBPs/Protein_pos.txt ../data/3-DBPs/Protein_neg.txt -label +1 -1
```

#### 4 Identification of intrinsically disordered regions in proteins

```shell
python BioSeq-BLM_Res.py -category Protein -method BLOSUM62 -ml LSTM -epoch 10 -lr 0.01 -dropout 0.5 -batch_size 20 -fixed_len 300 -n_layer 2 -hidden_dim 64 -seq_file ../data/4-IDRs/protein_seq.txt -label_file ../data/4-IDRs/protein_label.txt
```

#### 5 RNA-binding protein identification

```shell
python BioSeq-BLM_Seq.py -category Protein -mode TM -method LSA -in_tm BOW -words Top-N-Gram -top_n 2 -cl Kmeans -nc 5 -fs Tree -nf 128 -dr TSVD -np 128 -rdb no -ml SVM -seq_file ../data/5-RBPs/RBP_590.txt ../data/5-RBPs/NRBP_590.txt -label +1 -1
```

#### 6 RNA secondary structure prediction

```shell
BioSeq-BLM_Seq.py -category DNA -mode OHE -method One-hot -ml RF -seq_file ../data/6-RSS/ph/ph_seq_pos.txt ../data/6-RSS/ph/ph_seq_neg.txt -label +1 -1 -bp 1 -metric AUC
```

```shell
BioSeq-BLM_Seq.py -category DNA -mode OHE -method One-hot -ml RF -seq_file ../data/6-RSS/py/py_seq_pos.txt ../data/6-RSS/py/py_seq_neg.txt -label +1 -1 -bp 1 -metric AUC
```

```shell
BioSeq-BLM_Seq.py -category DNA -mode OHE -method One-hot -ml RF -seq_file ../data/6-RSS/pdb/pdb_seq_pos.txt ../data/6-RSS/pdb/pdb_seq_neg.txt -label +1 -1 -bp 1 -metric AUC
```



## datasets

The datasets used in manuscript can be found here [BioSeq-BLM (Download)](http://bliulab.net/BioSeq-BLM/download/#dataset).

License
----

Mozilla Public License 2.0


Contact
----

Prof. Dr. Bin Liu, email: bliu@bliulab.net