# CancerSTFormer-50um Local Model Usage

We provide a tutorial below for in silico gene perturbation using this model. 

## Contents

- [Fine-tuning the model to enable better prediction](#fine-tuning-the-model-to-enable-better-prediction)
  - [Step 1: Define Training Genes](#step-1-define-training-genes)
  - [Step 2: Modify Finetuning Code](#step-2-modify-finetuning-code)
  - [Step 3: Run Finetuning Code](#step-3-run-finetuning-code)
- [In Silico Gene Perturbation using the Fine-tuned CancerSTFormer Model](#in-silico-gene-perturbation-using-the-fine-tuned-cancerstformer-model)
  - [Step 1: Define gene to perturb](#step-1-define-gene-to-perturb)
  - [Step 2: Modify perturbation code](#step-2-modify-perturbation-code)
  - [Step 3: View the perturbation results](#step-3-view-the-perturbation-results)
  - [Step 4: Evaluate the Perturbation Results](#step-4-evaluate-the-perturbation-results)

<br><br>

## Fine-tuning the model to enable better prediction 

([Back to main &uarr;](#contents))

We always recommend first fine tune the CancerSTFormer model before doing in silico gene perturbation. We recommend a Gene Classifier to fine-tune the model. Training genes can be treatment resistance or sensitive genes that come from bulk RNAseq studies or clinical trial studies. For example, we illustrate with ganitumab sensitive genes. Ganitumab is a IGF1R inhibitor. Thus finetuning will allow us better predict ST response to IGF1R deletion.

### Step 1: Define Training Genes 

([Back to main &uarr;](#contents))

We have prepared a ganitumab sensitive gene-set 'ganitumab.upregulated.top300` to train the model to recognize it.

```
CARD14
NRM
PEX6
NCAPH
C1orf233
CASP2
NT5M
GAS2L1P2
RDH11
FBXO24
WDR4
KTI12
HIST1H4I
RAD54L
LPAR3
HERC6
HSBP1L1
CCHCR1
CDK5R1
GINS3
...
```

Next define random genes (randomly selected genes from genome of matched size as above positive gene-set). This is also provided `gene.shuffled.upregulated`.

### Step 2: Modify Finetuning Code

([Back to main &uarr;](#contents))

Modify the code `run_finetune_2f_ganitumab.py`. This program shown below, contains finetuning settings and instructions.
```python
import torch
import numpy as np
import pickle
import os
import sys
from new_emb_extractor import EmbExtractor
from new_in_silico_perturber import InSilicoPerturber
from new_in_silico_perturber_stats import InSilicoPerturberStats
from new_classifier import Classifier
import datetime
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()

direction = sys.argv[1]

file1, file2 = None, None
if direction=="up":
	file1 = "ganitumab.upregulated.top300"
	file2 = "gene.shuffled.upregulated"
elif direction=="down":
	file1 = "ganitumab.downregulated.top300"
	file2 = "gene.shuffled.downregulated"
	
genes_group1 = list(np.loadtxt(file1,dtype=str))
genes_group2 = list(np.loadtxt(file2, dtype=str))
training_args = {"num_train_epochs": 30.0, "weight_decay": 0.35, "learning_rate": 1e-5, "warmup_steps":500, "lr_scheduler_type": "polynomial"}
ray_config = {"num_train_epochs": [1.0,],
"learning_rate": (1e-3, 1e-2),
"weight_decay": (0.01, 0.05),
"lr_scheduler_type": ["linear", "cosine", "polynomial"],
"warmup_steps": (5, 50),
"seed": (0, 100),
"per_device_train_batch_size": [10,],
}

ensemble_dictionary = {}
with open("jan21_qian_gene_name_id_dictionary.pickle", 'rb') as file:
	ensemble_dictionary = pickle.load(file)
good_group1 = [ensemble_dictionary[gene] for gene in genes_group1 if gene in ensemble_dictionary]
good_group2 = [ensemble_dictionary[gene] for gene in genes_group2 if gene in ensemble_dictionary]
print(len(good_group1))
print(len(good_group2))

label1, label2 = None, None
if direction=="up":
	label1 = "Responder"
	label2 = "Random.genes"
elif direction=="down":
	label1 = "Nonresponder"
	label2 = "Random.genes"
gene_dict = {label1: good_group1, label2: good_group2}
id_class_dict = {1: label1, 0: label2}
filter_data_dict={"Disease":["TNBC"]}
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

output_prefix = "responder_test"
output_dir = f"/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/{datestamp}"
os.makedirs(output_dir,exist_ok=True)
num_trials = 4
input_dataset = "/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/STGeneformer_TNBC_Normal_Perturbset_filtered.dataset"

fw = open(output_dir + "/readme", "w")
json_string = json.dumps(ray_config, indent=4)
fw.write(json_string + "\n")
fw.write(file1 + "\n")
fw.write(file2 + "\n")
json_string = json.dumps(id_class_dict, indent=4)
fw.write(json_string +"\n")
fw.write(input_dataset + "\n")
fw.write(str(num_trials) + "\n")
fw.close()

cc = Classifier(classifier="gene", gene_class_dict = gene_dict, max_num_spots = 10_000, freeze_layers = 4, num_crossval_splits = 1,
forward_batch_size=200, nproc=16, training_args = training_args, cust_id_class_dict = id_class_dict, ray_config=ray_config, filter_data=filter_data_dict)

cc.prepare_data(input_data_file=input_dataset, o_directory=output_dir, o_prefix=output_prefix)

all_metrics = cc.validate(model_directory="/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/models",
prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
o_directory=output_dir,  o_prefix=output_prefix, 
n_hyperopt_trials=num_trials)
```

The important settings are:
- **file1**: Positive gene-set (ganitumab sensitive genes)
- **file2**: Negative gene-set (randomly selected genes)
- **ray_config**: The fine-tuning settings, including the settings to iterate through: epochs, learning_rate, weight_decay, warmup_steps, and batch_size. Adjust batch_size according to your GPU memory.
- **num_trials**: Number of Ray Tuning trials (recommend around 50-60).
- **input_dataset**: Input ST dataset to be used for training purpose (in our case TNBC ST samples).
- **model_directory**: Location of the pretrained model, which fine-tuning will begin from
- **Classifier** settings: max_num_spots (the maximum number of spots from input_dataset to take for training purpose), classifier (the type of classifier, in this case, "gene"), num_crossval_splits (1 for 1-split, i.e. 2-fold cross validation, use one fold for training, the other fold for evaluation/model selection. Here split refers to training gene-set split.), freeze_layers (top 4 layers will be frozen. Leaving 2 trainable layers).
<br>

### Step 3: Run Finetuning Code

([Back to main &uarr;](#contents))

Run the codes.

```
python3 run_finetune_2f_ganitumab.py up
```

Upon finishing you will see a training summary table:

```
Trial status: 60 TERMINATED
Current time: 2025-09-29 12:25:16. Total running time: 1hr 58min 59s
Logical resource usage: 16.0/192 CPUs, 1.0/2 GPUs (0.0/1.0 accelerator_type:RTX)
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name            status         num_train_epochs     learning_rate     weight_decay   lr_scheduler_type       warmup_steps         seed     ..._train_batch_size     iter     total time (s)     eval_loss     eval_accuracy     eval_macro_f1     eval_runtime │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ _objective_9d4a16da   TERMINATED                    1        0.0066878         0.014467    linear                      29.39      12.4275                          10        1            234.389      4.93306           0.51217           0.491421          67.2482 │
│ _objective_e275a3f4   TERMINATED                    1        0.00486725        0.0406779   polynomial                  22.4058    43.5417                          10        1            233.552      6.97117           0.311758          0.307622          67.0888 │
│ _objective_cb4f98d2   TERMINATED                    1        0.00164195        0.0196544   cosine                      16.145     96.5905                          10        1            235.717      4.2194            0.362871          0.362009          67.0409 │
│ _objective_9490ea56   TERMINATED                    1        0.00927173        0.024868    polynomial                   6.33049   33.513                           10        1            233.596      0.709057          0.400208          0.28582           66.4502 │
│ _objective_7dca3e41   TERMINATED                    1        0.00108445        0.0179881   cosine                       6.75741    5.95588                         10        1            235.857      4.36708           0.408256          0.404862          67.0773 │
│ _objective_feb5f434   TERMINATED                    1        0.00531807        0.0335563   polynomial                  12.1576     6.53787                         10        1            233.935      2.56062           0.572694          0.562587          66.4148 │
│ _objective_5f881369   TERMINATED                    1        0.00991941        0.0141506   cosine                      46.3167    44.632                           10        1            234.576      0.707876          0.400208          0.28582           66.5832 │
│ _objective_40ac5089   TERMINATED                    1        0.00394853        0.0312211   polynomial                  40.9281    23.2805                          10        1            234.27       4.95956           0.390439          0.364426          66.768  │
│ _objective_d709dc51   TERMINATED                    1        0.00224565        0.0439686   linear                      11.9161    78.6267                          10        1            235.979      5.21251           0.349208          0.348649          67.257  │
│ _objective_fee865c3   TERMINATED                    1        0.00194186        0.0424276   cosine                      28.323     34.3504                          10        1            233.692      5.08754           0.332966          0.330876          66.6259 │
│ _objective_42acf2b3   TERMINATED                    1        0.0010561         0.0297774   cosine                      11.5937    11.9076                          10        1            236.099      4.42339           0.433131          0.422257          67.2418 │
│ _objective_73029647   TERMINATED                    1        0.00924503        0.0335129   linear                      39.8321    33.9634                          10        1            233.489      0.709838          0.400208          0.28582           66.2592 │
│ _objective_c62dfd3e   TERMINATED                    1        0.00358382        0.0290961   polynomial                  11.9306    37.9381                          10        1            235.891      5.7337            0.435614          0.419512          67.2638 │
│ _objective_5913b8ab   TERMINATED                    1        0.00154752        0.0163537   polynomial                  34.4894     4.19612                         10        1            233.863      5.47274           0.338466          0.326617          66.6394 │
│ _objective_6a7dc61c   TERMINATED                    1        0.00115952        0.0186752   linear                      13.4686    19.5992                          10        1            235.911      4.80008           0.363731          0.363706          67.0635 │
│ _objective_0e271915   TERMINATED                    1        0.00572576        0.023532    polynomial                  32.8449    60.2183                          10        1            234.095      4.10923           0.53857           0.497203          66.5217 │
│ _objective_90912e3c   TERMINATED                    1        0.00254584        0.0238335   linear                      41.2621    91.1409                          10        1            237.486      5.58195           0.305657          0.287321          68.0976 │
│ _objective_ca3ba36d   TERMINATED                    1        0.00917971        0.0382281   polynomial                  27.0048    95.6741                          10        1            232.943      0.70401           0.400208          0.28582           66.2129 │
│ _objective_8904b9eb   TERMINATED                    1        0.00109521        0.0167292   linear                      32.3203    81.3991                          10        1            234.581      4.82832           0.412069          0.392878          66.9216 │
│ _objective_0cb03f6f   TERMINATED                    1        0.00659136        0.0182859   polynomial                  16.012     24.5139                          10        1            235.92       2.71989           0.484504          0.467683          67.3096 │
│ _objective_97f6fc5e   TERMINATED                    1        0.00549115        0.0476003   polynomial                  22.936     59.7207                          10        1            234.138      4.23995           0.390731          0.352365          66.5366 │
│ _objective_334d5fc2   TERMINATED                    1        0.00703879        0.0104185   polynomial                  48.2115    58.4315                          10        1            235.837      4.86487           0.334767          0.331562          67.2494 │
│ _objective_884d4ccc   TERMINATED                    1        0.00453988        0.0358146   polynomial                  22.5866    62.1867                          10        1            233.355      4.17962           0.535049          0.507873          66.5397 │
│ _objective_2e9de57d   TERMINATED                    1        0.00287905        0.0250806   polynomial                  32.848     73.9912                          10        1            235.528      5.15002           0.356867          0.356589          66.8877 │
│ _objective_b7a6e519   TERMINATED                    1        0.0074887         0.0274724   polynomial                  38.1163    70.1909                          10        1            234.177      2.82802           0.474963          0.464002          66.6763 │
│ _objective_35197b1c   TERMINATED                    1        0.00437581        0.035735    polynomial                  19.9164    66.2639                          10        1            235.921      4.43787           0.510563          0.474852          66.9957 │
│ _objective_f83d5a67   TERMINATED                    1        0.00333401        0.0487128   polynomial                  19.132     51.7947                          10        1            233.797      5.5929            0.360713          0.359575          66.6815 │
│ _objective_a3732761   TERMINATED                    1        0.00481952        0.0371094   polynomial                   8.47384   83.064                           10        1            236.097      4.99579           0.426657          0.397758          67.2873 │
│ _objective_a3a5a02a   TERMINATED                    1        0.00589731        0.0456297   polynomial                  24.6132    48.9414                          10        1            234.023      6.16974           0.369637          0.366393          66.5221 │
│ _objective_e1ffa1da   TERMINATED                    1        0.00797973        0.0341143   polynomial                  17.7342    88.6334                          10        1            234.978      0.706353          0.400208          0.28582           66.6863 │
│ _objective_3567ebde   TERMINATED                    1        0.00389704        0.0411192   polynomial                   9.09183    0.506184                        10        1            234.432      5.10006           0.417181          0.417164          66.6594 │
│ _objective_9b075d84   TERMINATED                    1        0.00283916        0.0324777   cosine                      22.7647    53.2849                          10        1            235.652      4.10639           0.437301          0.435115          66.9086 │
│ _objective_1c2b7b97   TERMINATED                    1        0.00474311        0.0376592   polynomial                  25.5476    66.9444                          10        1            234.37       5.56424           0.44251           0.440791          66.7991 │
│ _objective_a8caf0ae   TERMINATED                    1        0.00431977        0.0386571   linear                      15.5177    14.198                           10        1            234.979      0.710001          0.400208          0.28582           66.6759 │
│ _objective_8af3a1d6   TERMINATED                    1        0.00639724        0.0217353   cosine                       5.60345   42.0674                          10        1            232.951      0.707788          0.400208          0.28582           66.2716 │
│ _objective_a068e972   TERMINATED                    1        0.00828303        0.0405758   polynomial                  21.1872    46.5163                          10        1            234.925      0.709327          0.400208          0.28582           66.7043 │
│ _objective_7c339307   TERMINATED                    1        0.00521712        0.0352514   cosine                      16.5262    75.1527                          10        1            234.082      5.55139           0.320147          0.316765          66.6194 │
│ _objective_af9bedab   TERMINATED                    1        0.00204092        0.0276969   polynomial                  29.688     29.5511                          10        1            235.872      5.29806           0.368242          0.360179          67.0715 │
│ _objective_4d8f70fe   TERMINATED                    1        0.00331097        0.0312422   polynomial                   9.32709   39.9042                          10        1            234.166      4.69296           0.477218          0.445232          66.5549 │
│ _objective_3e30c5ec   TERMINATED                    1        0.00389816        0.0450234   linear                      13.8285    99.8809                          10        1            236.072      4.21114           0.507042          0.487474          67.1802 │
│ _objective_5a6ad916   TERMINATED                    1        0.00149775        0.0426311   cosine                      18.8622    10.781                           10        1            234.169      5.01176           0.319887          0.318994          66.7929 │
│ _objective_c1a3d566   TERMINATED                    1        0.00254616        0.0263769   polynomial                  25.3083    55.1754                          10        1            236.326      4.78937           0.383608          0.372765          67.3296 │
│ _objective_78183b26   TERMINATED                    1        0.00622828        0.0312043   cosine                      35.7801    28.3368                          10        1            233.533      3.22815           0.416093          0.40371           66.4711 │
│ _objective_1adc2f5e   TERMINATED                    1        0.00426798        0.0205984   linear                      44.2387    63.7077                          10        1            235.889      5.90712           0.409392          0.40882           67.1321 │
│ _objective_f50de48f   TERMINATED                    1        0.00512449        0.0404399   polynomial                  29.461      0.0172735                       10        1            233.918      6.66509           0.285114          0.253964          66.6932 │
│ _objective_a3e06c2d   TERMINATED                    1        0.00174374        0.0102228   polynomial                   5.21713   86.4063                          10        1            235.94       4.85126           0.346076          0.345476          67.068  │
│ _objective_aa5a7053   TERMINATED                    1        0.00297575        0.0496302   linear                      11.4248    17.7562                          10        1            233.902      4.97332           0.406796          0.399537          66.5033 │
│ _objective_3e8570ce   TERMINATED                    1        0.00129274        0.0292593   polynomial                   7.72727   35.1352                          10        1            235.932      4.50747           0.388655          0.380848          67.1788 │
│ _objective_a81dbd31   TERMINATED                    1        0.00991637        0.0132692   cosine                      14.188     76.7835                          10        1            233.102      1.81446           0.481064          0.476483          66.2595 │
│ _objective_c3de0015   TERMINATED                    1        0.00255702        0.046734    polynomial                  20.7474     6.27823                         10        1            236.14       4.8185            0.467937          0.461914          67.3071 │
│ _objective_cad44e52   TERMINATED                    1        0.00354064        0.0433429   polynomial                  31.3819    71.4105                          10        1            234.295      6.25649           0.403859          0.396705          66.717  │
│ _objective_ba930ec7   TERMINATED                    1        0.00225311        0.0357742   linear                      10.4621    47.348                           10        1            236.035      4.3237            0.402106          0.392925          67.1949 │
│ _objective_9ad95cf2   TERMINATED                    1        0.00863617        0.0328525   polynomial                  26.5289    22.9455                          10        1            233.778      3.14087           0.522311          0.521029          66.4142 │
│ _objective_77b2316d   TERMINATED                    1        0.00706819        0.0392085   polynomial                  28.2625    56.7923                          10        1            236.28       2.494             0.454745          0.403166          67.4629 │
│ _objective_29648922   TERMINATED                    1        0.00550523        0.0231803   cosine                      23.4112    62.7176                          10        1            234.365      4.00633           0.582235          0.498279          66.8074 │
│ _objective_04a7d81b   TERMINATED                    1        0.00875542        0.0332147   polynomial                  44.8207    20.9515                          10        1            236.54       3.87576           0.440676          0.438146          67.5383 │
│ _objective_11be06a5   TERMINATED                    1        0.0098234         0.0254752   linear                      37.1565     7.45592                         10        1            232.609      0.705291          0.400208          0.28582           65.9781 │
│ _objective_941e7a47   TERMINATED                    1        0.00801855        0.0307257   polynomial                  40.5679    15.105                           10        1            235.798      4.24307           0.375414          0.372769          67.0529 │
│ _objective_0daa4996   TERMINATED                    1        0.00725638        0.0158677   polynomial                  42.1282    23.0332                          10        1            233.159      3.48375           0.42789           0.35613           66.2857 │
│ _objective_e3a20bc9   TERMINATED                    1        0.0061936         0.0272673   polynomial                  48.4675     2.6445                          10        1            235.813      5.24046           0.344616          0.343367          67.0972 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

This table contains the following useful information:
- eval_loss
- eval_macro_f1
- eval_accuracy

Each line is 1 of 60 tuning trials. Note down the trial ID column (first column, e.g., _objective_29648922. The trial ID is 29648922.) If you want to select a certain trial for downstream analysis, the training model for each trial is saved for later usage.


### How to select trials?

**We generally select the top 5 trials (with lowest eval_loss) and the top 5 trials (with highest eval_macro_f1) for downstream analysis and model evaluation. This helps to get a sense of spread of trained model performances. We do not rely on the so-called best single trial, as it may sometimes mis-predict. Thus our model evaluation is the average of performances of the top 5 trials.**

### Location of the 60 trials' trained model

They are located in, for example: `251118071123/251118_cancerstformer_geneClassifier_responder_test/ksplit1/run-c14060f2/checkpoint-1000/`

Change the trial ID `run-c14060f2` with the trial ID from the above table. You should see the following files in the `checkpoint-1000` folder:
```
-rw-rw-r-- 1 qian qian      683 Nov 18 07:14 config.json
-rw-rw-r-- 1 qian qian     5240 Nov 18 07:14 training_args.bin
-rw-rw-r-- 1 qian qian 35208744 Nov 18 07:14 model.safetensors
-rw-rw-r-- 1 qian qian     1064 Nov 18 07:14 scheduler.pt
-rw-rw-r-- 1 qian qian 53559546 Nov 18 07:14 optimizer.pt
-rw-rw-r-- 1 qian qian     2672 Nov 18 07:14 trainer_state.json
-rw-rw-r-- 1 qian qian    14180 Nov 18 07:14 rng_state.pth
```

<br><br>

## In Silico Gene Perturbation using the Fine-tuned CancerSTFormer Model

([Back to main &uarr;](#contents))

### Step 1: Define gene to perturb

([Back to main &uarr;](#contents))

Define the gene to be perturbed. See `immune.gene.set`:

```
PDCD1
CD274
CTLA4
IDO
```
One query per line. Users can add multiple queries. In this case, the program will perturb each gene individually, one at a time, and saves the results in the gene's own folder.

### Step 2: Modify perturbation code

([Back to main &uarr;](#contents))

See example file: `run_perturb_finetuned.py`.

```import torch
import numpy as np
import pickle
import os
import sys
from new_emb_extractor import EmbExtractor
from new_in_silico_perturber import InSilicoPerturber
from new_in_silico_perturber_stats import InSilicoPerturberStats
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()
def run_perturb(model_path,dataset_path,out_dir,num_embs=None):
    genes_perturb = list(np.loadtxt("immune.gene.set", dtype=str))
    file_path = "jan21_qian_gene_name_id_dictionary.pickle"
    with open(file_path, 'rb') as file:
        ensemble_dictionary = pickle.load(file)
    # Translate gene names to Ensembl IDs using the token_dictionary
    good_genes = [ensemble_dictionary[gene] for gene in genes_perturb if gene in ensemble_dictionary]
    file_path = "jan21_qian_new_token_dictionary.pickle"
    # Open the file in read-binary mode and load it with pickle
    with open(file_path, 'rb') as file:
        token_dictionary = pickle.load(file)
    # Translate gene names to Ensembl IDs using the token_dictionary
    good_genes_final = [gene for gene in good_genes if gene in list(token_dictionary.keys())]
    filter_data_dict={"Disease":["TNBC"]}
    for i, gene in enumerate(good_genes_final):
        try:
            print(f"Perturbing {gene}")
            out_dir_final = out_dir + f"/{gene}"
            os.makedirs(out_dir_final,exist_ok=True)
            isp = InSilicoPerturber(perturb_type="delete", perturb_rank_shift=None,
				genes_to_perturb = [gene], combos=0, anchor_gene=None,
				#model_type="Pretrained",
				model_type="GeneClassifier",
				num_classes=2, emb_mode="spot_and_gene", spot_emb_style="mean_pool", filter_data=filter_data_dict,
				max_num_spots=1000, emb_layer=-1, forward_batch_size=50, nproc=1)
            isp.perturb_data(model_path, dataset_path, out_dir_final, os.path.basename(dataset_path).replace(".dataset","_emb"))
            ispstats = InSilicoPerturberStats(mode="aggregate_gene_shifts", genes_perturbed = [gene], combos=0, anchor_gene=None)
            ispstats.get_stats(out_dir_final, None, out_dir_final, os.path.basename(dataset_path).replace(".dataset","_emb"))
        except Exception as e:
            print(f"Error perturbing gene {gene}: {e}")
#datestamp = sys.argv[1]
#run_id = sys.argv[2]
run_id = "run-d64b2a10"
out_dir = "TNBC_Gene_Shift_%s" % (run_id)
os.makedirs(out_dir,exist_ok=True)
'''
res_dir = None #250919_geneformer_geneClassifier_responder_test
for root, dirs, files in os.walk("finetuned"):
	for d in dirs:
		if d.endswith("responder_test"):
			res_dir = d
			break
'''
cpt_dir = None
for root, dirs, files in os.walk("finetuned/ksplit1/%s" % (run_id)):
	for d in dirs:
		if d.startswith("checkpoint-"):
			cpt_dir = d
			break
model_path = "finetuned/ksplit1/%s/%s" % (run_id, cpt_dir)
dataset_path = "STGeneformer_TNBC_Normal_Perturbset_filtered.dataset"
run_perturb(model_path,dataset_path,out_dir)
```

We should next run the perturbation codes:
```
python3 run_perturb_finetuned.py
```

<br>

### Step 3: View the perturbation results

([Back to main &uarr;](#contents))

The results are stored in the `out_dir` directory that is defined in the previous step. So let us see the directory.

```
cd TNBC_Gene_Shift_run-d64b2a10
ls -ltr
total 12
drwxrwxr-x 2 qian qian 4096 Nov 18 07:27 ENSG00000188389
drwxrwxr-x 2 qian qian 4096 Nov 18 07:28 ENSG00000120217
drwxrwxr-x 2 qian qian 4096 Nov 18 07:28 ENSG00000163599
``` 

We can go into a gene of interest, say ENSG00000120217. View the content using a text editor:

```
,Perturbed,Gene_name,Ensembl_ID,Affected,Affected_gene_name,Affected_Ensembl_ID,Cosine_sim_mean,Cosine_sim_stdev,N_Detections
13794,4242,CD274,ENSG00000120217,spot_emb,,,0.992040998339653,0.003973730458465491,1000
4964,4242,CD274,ENSG00000120217,5588,GIMAP6,ENSG00000133561,0.8500149784719243,0.1078250586677382,136
7243,4242,CD274,ENSG00000120217,12596,SYNGAP1,ENSG00000197283,0.8787107097136008,0.1249411248576963,37
7244,4242,CD274,ENSG00000120217,12570,PCNX3,ENSG00000197136,0.8795121718703964,0.10387161398726272,162
2279,4242,CD274,ENSG00000120217,10679,UBXN2A,ENSG00000173960,0.9160515743143418,0.1154890447283091,170
2813,4242,CD274,ENSG00000120217,12913,TSEN15,ENSG00000198860,0.9163218256252915,0.08835800405125742,268
7804,4242,CD274,ENSG00000120217,5397,PODNL1,ENSG00000132000,0.9364907214011269,0.07595966037060575,87
12009,4242,CD274,ENSG00000120217,9404,CACNB2,ENSG00000165995,0.9367158189415932,0.05329607125514493,8
10158,4242,CD274,ENSG00000120217,7316,FBXO25,ENSG00000147364,0.9385383279342961,0.07751704630792318,123
4965,4242,CD274,ENSG00000120217,7310,DOCK11,ENSG00000147251,0.9385549405759032,0.04462969592033357,88
1834,4242,CD274,ENSG00000120217,11085,ACAD9,ENSG00000177646,0.940849008737132,0.07471179175751033,256
2812,4242,CD274,ENSG00000120217,11527,YBEY,ENSG00000182362,0.9436754660899412,0.08354821016785943,179
1833,4242,CD274,ENSG00000120217,10369,RGS19,ENSG00000171700,0.9455347908726821,0.07308472174144554,282
128,4242,CD274,ENSG00000120217,10390,PWWP2B,ENSG00000171813,0.9458475092361713,0.0668332558393144,203
11952,4242,CD274,ENSG00000120217,11949,MOSMO,ENSG00000185716,0.9482384244007851,0.0495276132513713,67
8914,4242,CD274,ENSG00000120217,9484,PPFIBP2,ENSG00000166387,0.9511858949690689,0.06389755199307791,162
13408,4242,CD274,ENSG00000120217,13395,FXYD7,ENSG00000221946,0.9513469934463501,3.063678741455078e-05,2
9063,4242,CD274,ENSG00000120217,10326,ZNF524,ENSG00000171443,0.9544026698006524,0.08879216088724255,225
11359,4242,CD274,ENSG00000120217,12996,RUFY2,ENSG00000204130,0.9546462617703338,0.044570116054495464,67
...
```

Column 3 and column 4 are the gene perturbed (CD274 and its ENSEMBL gene ID). The column `Affected_gene_name` shows the most affected gene sorted from lowest to highest `Cosine_sim_mean`. The last column `N_Detections` is equally important - it shows number of spots in which the affected gene is expressed. The number here is presented out of a total of 1000 spots. For genes like CACNB2 and FXYD7, with N_detections of 8 and 2 respectively, they should be probably ignored due to low detections. **We therefore recommend filtering genes based on `N_Detections`, or reranking the genes based on a combination of `Cosine_sim_mean` and `N_Detections`.**

<br>

We have a function to rerank genes `read_pert` (located in the file `evaluate.pr.tnbc.immunotherapy.basal.custom.py`). see below:

```
def read_pert(n, checkpoint, detect_min=-1):
    f = open(checkpoint + "/" + n + "/STGeneformer_TNBC_Normal_Perturbset_filtered_emb.csv")
    h = f.readline().rstrip("\n").split(",")
    gene_list = []
    filtered = []
    for l in f:
        l = l.rstrip("\n")
        ll = l.split(",")
        ndetect = int(ll[-1])
        if detect_min!=-1:
            if ndetect<detect_min:
                filtered.append(ll[5])
                continue
        gene = ll[5]
        gene_list.append(gene)
    f.close()
    gene_list = gene_list + filtered
    return gene_list
```

Here we can run read_pert() with detect_min set to a cutoff like 100 - this means N_detections must be greater than or equal to 100 for the gene to be counted, else it is pushed to the bottom of the ranking.

<br>

### Step 4: Evaluate the Perturbation Results

([Back to main &uarr;](#contents))

There is a python script `evaluate.pr.tnbc.immunotherapy.basal.custom.py`. Look at the code:
```import sys
import os
import re
from scipy.stats import hypergeom
import numpy as np

def read_lr_targets(n):
	f = open(n)
	h = f.readline().rstrip("\n").split("\t")[1:]
	by_ligand = {}
	for l in f:
		l = l.rstrip("\n")
		ll = l.split("\t")
		lx = zip(h, ll[1:])
		for i1, i2 in lx:
			by_ligand.setdefault(i1, [])
			by_ligand[i1].append(i2)
	f.close()
	return by_ligand

def read_target_list(n):
	f = open(n)
	tt = []
	for l in f:
		l = l.rstrip("\n")
		tt.append(l.split("\t")[1])
	f.close()
	return tt

def pr_from_ranklist(ranklist, gold):
	tp = 0
	precisions, recalls = [], []
	G = len(gold)                       # number of true positives in gold standard
	for i, gene in enumerate(ranklist, 1):   # 1‑based rank
		if gene in gold:
			tp += 1
		precisions.append(tp / i)
		recalls.append(tp / G)
	return np.asarray(recalls), np.asarray(precisions)

def read_pert(n, checkpoint, detect_min=-1):
	f = open(checkpoint + "/" + n + "/STGeneformer_TNBC_Normal_Perturbset_filtered_emb.csv")
	h = f.readline().rstrip("\n").split(",")
	gene_list = []
	filtered = []
	for l in f:
		l = l.rstrip("\n")
		ll = l.split(",")
		ndetect = int(ll[-1])
		if detect_min!=-1:
			if ndetect<detect_min:
				filtered.append(ll[5])
				continue
		gene = ll[5]
		gene_list.append(gene)
	f.close()
	gene_list = gene_list + filtered
	return gene_list

def read_conversion(n):
	f = open(n)
	m = []
	for l in f:
		l = l.rstrip("\n")
		ll = l.split("\t")
		m.append((ll[0], ll[1]))
	f.close()
	return m

if __name__=="__main__":
	#arg1: checkpoint
	#arg2: detect_min
	#arg3: basal/shuf1
	goldstd = sys.argv[3] #basal or shuf1
	by_ligand = {}
	if goldstd=="basal":
		by_ligand = read_lr_targets("/media/scandisk/Perturbation/pdcd1.ispy2.basal.targets.txt")
	elif goldstd=="shuf1":
		by_ligand = read_lr_targets("../../tnbc.responder.spatialmodel/pembro/shuf1/test/TNBC.PDCD1.targets.200.txt") #good
	else:
		print("Error, not exist")
		sys.exit(0)
	target_list = read_target_list("../profiles.targets.txt")
	#pert_genes = read_pert("ENSG00000120217")
	checkpoint = sys.argv[1]
	#print("Target list", len(target_list))
	#print("Pert list", len(pert_genes))
	#print("Overlap list", len(set(target_list) & set(pert_genes)))
	fw1 = open("%s/TNBC_immunotherapy_PDCD1_fold_pr_over_random.txt" % checkpoint, "w")
	fw2 = open("%s/TNBC_immunotherapy_PDCD1_pr.txt" % checkpoint, "w")
	fw3 = open("%s/TNBC_immunotherapy_PDCD1_recall.txt" % checkpoint, "w")

	sym_ensembl = read_conversion("conversion")
	for sym, ens in sym_ensembl:
		#for sym2 in ["CD274", "PDCD1", "CTLA4"]:
		for direction in ["up", "down"]:
			pert_genes = read_pert(ens, checkpoint, detect_min=int(sys.argv[2]))
			print("Pert gene", sym)
			eval_genes = list(set(target_list) & set(pert_genes))
			l_target = [e for e in by_ligand["PDCD1_%s" % (direction)] if e in eval_genes]
			#predicted ligand targets
			pred = [e for e in pert_genes[:500] if e in eval_genes]
			ov = set(l_target) & set(pred)
			N = len(eval_genes)
			K = len(l_target)
			n = len(pred)
			k = len(ov)
			p = hypergeom.sf(k-1, N, K, n)
			logp = -1.0 * np.log10(p)
			exp_k = hypergeom.isf(0.50, N, K, n) + 1
			fold_over_random = len(ov) / exp_k
			recall, precision = pr_from_ranklist(pert_genes, set(l_target))
			grid = np.arange(0.00, 1.01, 0.01)             # 0.00, 0.01, …, 1.00
			interp_prec = [precision[recall >= r].max() if np.any(recall >= r) else 0.0 for r in grid]
			baseline_pr = len(l_target) / len(pert_genes)
			fold_pr = [ip/baseline_pr for ip in interp_prec]
			fw1.write(" ".join(["%f" % fr for fr in fold_pr]) + "\n")
			fw2.write(" ".join(["%f" % pr for pr in interp_prec]) + "\n")
			fw3.write(" ".join(["%f" % re for re in grid]) + "\n")
		#print(len(ov), len(l_target), len(pred), len(eval_genes), logp, exp_k, fold_over_random)
	fw1.close()
	fw2.close()
	fw3.close()
```

To use the evaluation script, you must define the gold-standard genes, i.e., PD-1 targets. See line:
`pdcd1.ispy2.basal.targets.txt`:

#### Defining gold standard genes

Here is an example of gold-standard genes, which is top 200 PDCD1-upregulated and PDCD1-downregulated genes of a held-out cohort:

```
column	PDCD1_up	PDCD1_down
1	PSME4	ZFP62
2	ATP6V1E2	TMEM187
3	DDX58	SLC25A5
4	BCS1L	ZNF595
5	SLC45A4	DDX41
6	APP	ZNF581
7	RIOK1	TBCEL
8	TOR3A	ARRDC3
9	TRIM25	ERO1LB
10	ND1	FAM188A
11	QSOX2	JUN
12	DPH2	DHRS12
...
```

#### Running the evaluation

The evaluation will use the gold-standard genes to calculate a **Precision over Recall** curve and **Precision-over-random Over Recall** curve. For example run:

```
python3 evaluate.pr.tnbc.immunotherapy.basal.custom.py
```

After running you will see 3 files generated:
```
-rw-rw-r-- 1 qian qian 5454 Sep 21 11:44 TNBC_immunotherapy_PDCD1_recall.txt
-rw-rw-r-- 1 qian qian 5454 Sep 21 11:44 TNBC_immunotherapy_PDCD1_pr.txt
-rw-rw-r-- 1 qian qian 5455 Sep 21 11:44 TNBC_immunotherapy_PDCD1_fold_pr_over_random.txt
```

- The content of `TNBC_immunotherapy_PDCD1_pr.txt` is the **Precision** values, over the recall values (`TNBC_immunotherapy_PDCD1_recall.txt`).
- The content of TNBC_immunotherapy_PDCD1_fold_pr_over_random.txt shows the **Precision-Over-Radom values**, over the same recalls as above. For example, the values below are the fold-precision-over-random values.
```
3.470265 3.470265 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 ...
```

You can easily use ggplot2 or matplotlib (Python) to plot the PR curve.

