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
We also provide package version for all pre-requisite packages here. See [file.version.txt](./file.version.txt)

 
