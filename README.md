# CancerSTFormer

CancerSTFormer consists of a pair of spatially aware transcriptomic foundation models to accommodate niche modeling at different length scales. These models, at the 50um-Local and 250um-Extended scales, possess unique capabilities to recover ligand-target gene relationships, niche-specific differentially expressed genes, and organ-specific metastasis associated genes in diverse cancer applications. CancerSTFormer can also reveal the responses of immune-checkpoint blockade therapies, and other targeted therapies, on patients’ tumors given their ST profiles through gene perturbation analysis.

## Installation

We require Pytorch/2.1.2+cu121, python/3.10.9, gcc/11.3.0.
<br>
Our NVIDIA graphics card driver is as follows:

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
```

<br>
We also require ray/2.6.1, tdigest/0.5.2.2, datasets/2.14.5, tokenizers/0.13.2, transformers/4.26.1.
<br>
We also provide package version for all pre-requisite packages here. See [file.version.txt](https://github.com/bernard2012/CancerST/blob/main/file.version.txt)
<br>
CancerSTFormer can be installed by simply copying the codes to user's working directory. There is no pip package installation required.


## Usage

There are two models to choose from. 

### 50um-Local Model

Copy the codes from `src/local` directory to user's folder.

```
cp src/local/*.py .
```

Setup `env.py` (see below). An example is provided in `example/local` folder. This defines the gene dictionary, gene median pickle files, etc. Again, these need to be copied to user's folder and referenced in `env.py`. These pickle files are located in `example/local.perturb`.

```
import sys
import os
import re

def get_token_dictionary_file():
    return "/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/jan21_qian_new_token_dictionary.pickle"
def get_ensembl_dictionary_file():
    return "/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/jan21_qian_gene_name_id_dictionary.pickle"
def get_gene_median_file():
    return "/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/jan21_qian_detected_gene_median_dict.pickle"
```

```
cp example/local.perturb/*.pickle .
```
Run perturbation tutorial. See `example/local.perturb/README.md`.


### 250um-Extended Model

Copy the codes from `src/extended` directory to user's folder.

```
cp src/extended/*.py .
```

Setup `env.py` (see below). An example is provided in `example/local` folder. This defines the gene dictionary, gene median pickle files, etc. Again, these need to be copied to user's folder and referenced in `env.py`. These pickle files are located in `example/extended.perturb`.

```
import sys
import os
import re

def get_token_dictionary_file():
    return "/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/Extended.model/new_token_dictionary.pickle"
def get_ensembl_dictionary_file():
    return "/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/Extended.model/gene_id_dictionary.pickle"
def get_gene_median_file():
    return "/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/Extended.model/detected_gene_median_dict.pickle"
```

```
cp example/extended.perturb/*.pickle .
```
Run perturbation tutorial. See `example/extended.perturb/README.md`. 
