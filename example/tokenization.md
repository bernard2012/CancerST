# Tokenization tutorial (250um-Extended Model)

<b>This is a tokenization tutorial for 250um-Extended model.</b> The tokenization procedure for the 50um-Local model is different and is in another page.

The required format of input file is gene counts matrix.
<br>
We can support numpy matrices.
<br>

If users give numpy matrices, we require 3 files per ST sample:
- `new_genes.npy` (1D string numpy array, stores gene names, should be human gene symbols, case-sensitive) 
- `new_spots.npy` (1D string numpy array, stores spots ID)
- `new_matrix.npy` (2D matrix, first dimension is genes, second dimension is spots, float32 or integer)

We also require `tissue_positions_list.csv` for each ST sample. An example is provided below:
```
ACGCCTGACACGCGCT-1,0,0,0,558,543
TACCGATCCAACACTT-1,0,1,1,601,568
ATTAAAGCGGACGAGC-1,0,0,2,558,593
GATAAGGGACGATTAG-1,0,1,3,601,618
GTGCAAATCACCAATA-1,0,0,4,558,642
TGTTGGCTGGCGGAAG-1,0,1,5,601,667
GCATCCTCTCCTATTA-1,0,0,6,558,692
GCGAGGGACTGCTAGA-1,0,1,7,601,717
TGGTACCGGCACAGCC-1,0,0,8,557,742
GCGCGTTTAAATCGTA-1,0,1,9,601,767
```

Name the 4 files above as the following:
`<base>_new_genes.npy`, `<base>_new_matrix.npy`, `<base>_new_spots.npy`, `<base>_tissue_positions_list.csv`. For example:
<br>

```
GSM5808054_new_genes.npy
GSM5808054_new_matrix.npy
GSM5808054_new_spots.npy
GSM5808054_tissue_positions_list.csv
```

## Step 1: Prepare a sample list file.

This sample should be list of samples' "_new_matrix.npy" files. For example, write a file sample.list:
```
GSM5808054_new_matrix.npy
GSM5808055_new_matrix.npy
GSM5808056_new_matrix.npy
GSM5808057_new_matrix.npy
```

## Step 2: Convert samples into pickle files

```
python3 readfiles.feb7.py sample.list
```

This step will generates a set of pickle files, each encapsulates genes, spots, matrix, and tissue position information. For example:
```
GSM5808054_data.pkl
GSM5808055_data.pkl
GSM5808056_data.pkl
GSM5808057_data.pkl
```

## Step 3: Create KNN neighborhood graph per ST sample
```
python3 delauney.py GSM5808054_data.pkl
```

This will generate spatial graph out of tissue coordinates within a ST sample, and save the outputs into `GSM5808054_with_nei.pkl`.

Users can run this step in a loop for multiple samples:
```
for i in `ls -1 *_data.pkl`; do python3 delauney.py $i; done
```

## Step 4: Annotate each ST sample (store metadata)

Users can also supply sample information such as study, tissue, preparation, and sample ID. We require these fields to be completed. Additionally, we also need `n_counts`, which is total log(1+normalized counts) per sample. This is automatically calculated by the program in this step. Users can customize the script for this step.
For example, for our samples we have the following metadata:

```
    header = ["", "n_counts", "study", "Tissue", "Preparation", "Sample ID"]

    study = "GSE193460"
    tissue = "lung"
    prep = "fresh frozen"
    sample_id = gsm_id
```

The inputs are given in the following lines, which need customized in output.colmeta.py:
```
    spots_path  = "GSM5808057_new_spots.npy"
    matrix_path = "GSM5808057_new_matrix.npy"
    out_path = "col.metadata/GSM5808057_with_nei.pkl"
```
Change the above 3 lines to names of each ST sample.  
```
python3 output.colmeta.py 
```

The output file is stored in out_path.
This step needs to be done for each sample.


## Step 5: Tokenization

This is the last step. Organize the `<base>_with_nei.pkl`, `col.metadata/<base>_with_nei.pkl`, for all ST samples in a folder in this way.

```
mkdir lung_samples
cp GSM5808054_with_nei.pkl GSM5808055_with_nei.pkl GSM5808056_with_nei.pkl GSM5808057_with_nei.pkl lung_samples/.
```

```
ls -ltr lung_samples
-rw-rw-r-- 1 qian qian 258343482 Feb  8 11:15 GSM5808054_with_nei.pkl
-rw-rw-r-- 1 qian qian 261716442 Feb  8 11:15 GSM5808055_with_nei.pkl
-rw-rw-r-- 1 qian qian 168940623 Feb  8 11:15 GSM5808056_with_nei.pkl
-rw-rw-r-- 1 qian qian 344775574 Feb  8 11:15 GSM5808057_with_nei.pkl
```

Copy the `col.metadata` from Step 4 for the above samples as follows. This folder should be at the same level as `lung_samples`.

```
ls -ltr col.metadata/
-rw-rw-r-- 1 qian qian 123527 Feb  8 11:28 GSM5808055_with_nei.pkl
-rw-rw-r-- 1 qian qian 122017 Feb  8 11:28 GSM5808054_with_nei.pkl
-rw-rw-r-- 1 qian qian 162107 Feb  8 11:28 GSM5808057_with_nei.pkl
-rw-rw-r-- 1 qian qian  79598 Feb  8 11:28 GSM5808056_with_nei.pkl
```

Modify the tokenizer script, `qian_tokenizer_lung.py`, specifically the main function with these settings:

```
if __name__ == "__main__":
    tk = ST_TranscriptomeTokenizer({"study": "study", "Tissue": "Tissue", "Sample ID": "Sample ID", "Preparation": "Preparation", "barcode":"barcode", "coord_x":"coord_x", "coord_y":"coord_y"},nproc=32)
    tk.tokenize_data("/home/qian/perturbmap/lung_sample",
                     "/home/qian/perturbmap",
                     "CancerSTformer.Extended.Lung.dataset")
```

Next copy the gene dictionary and gene median files to the current directory so it looks like:
```
-rwxrwxr-x 1 qian qian 876139 Feb  8 11:17 detected_gene_median_dict.pickle
-rwxrwxr-x 1 qian qian 393285 Feb  8 11:17 new_token_dictionary.pickle
```

Finally, run the tokenizer.

```
python3 qian_tokenizer_lung.py
```

This should generate the dataset in the current folder:
```
qian@qian-Precision-7875-Tower:~/perturbmap$ ls -ltr Cancer.Extended.Lung.dataset/
total 122800
-rw-rw-r-- 1 qian qian 125737872 Mar 14 17:12 data-00000-of-00001.arrow
-rw-rw-r-- 1 qian qian       386 Mar 14 17:12 state.json
-rw-rw-r-- 1 qian qian       806 Mar 14 17:12 dataset_info.json
```
