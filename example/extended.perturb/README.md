# CancerSTFormer-250um Extended Model Usage

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

We always recommend first fine tune the CancerSTFormer 250um Extended model before doing in silico gene perturbation. We recommend a Gene Classifier to fine-tune the model. Training genes can be treatment resistance or sensitive genes that come from bulk RNAseq studies or clinical trial studies. For example, we illustrate with ganitumab sensitive genes. Ganitumab is a IGF1R inhibitor. Thus finetuning will allow us better predict ST response to IGF1R deletion.

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
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch

import env
from new_classifier import Classifier

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()

# choice = sys.argv[1]  # shuf1, shuf2, ... shuf3, shuf4

file1 = "upregulated.top300"							   # [A]
file2 = "gene.shuffled.upregulated"						# [B]

genes_responder = list(np.loadtxt(file1, dtype=str))
genes_random = list(np.loadtxt(file2, dtype=str))

training_args = {
	"num_train_epochs": 30.0,
	"weight_decay": 0.25,
	"learning_rate": 3e-6,
	"warmup_steps": 1500,
	"lr_scheduler_type": "polynomial",
}

ray_config = {											 # [C]
	"num_train_epochs": [1.0],
	"learning_rate": (1e-3, 1e-2),
	"weight_decay": (0.01, 0.05),
	"lr_scheduler_type": ["linear", "cosine", "polynomial"],
	"warmup_steps": (5, 50),
	"seed": (100, 1000),
	"per_device_train_batch_size": [10],
}

ensemble_dictionary = {}

with open(env.get_ensembl_dictionary_file(), "rb") as file:
	ensemble_dictionary = pickle.load(file)

good_genes = [
	ensemble_dictionary[gene]
	for gene in genes_responder
	if gene in ensemble_dictionary
]

good_random = [
	ensemble_dictionary[gene]
	for gene in genes_random
	if gene in ensemble_dictionary
]

gene_dict = {"Responder": good_genes, "Random.genes": good_random}
id_class_dict = {1: "Responder", 0: "Random.genes"}

current_date = datetime.now()
datestamp = current_date.strftime("%y%m%d%H%M%S")
datestamp_min = current_date.strftime("%y%m%d")

output_prefix = "responder_test"
output_dir = (
	"/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/"
	f"Extended.model/{datestamp}"
)
os.makedirs(output_dir, exist_ok=True)

input_dataset = (									   # [E]
	"/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/Extended.model/"
	"STFormer_TNBC_neighbor.dataset"
)
num_trials = 2										  # [D]

readme_path = os.path.join(output_dir, "readme")
with open(readme_path, "w") as fw:
	json_string = json.dumps(ray_config, indent=4)
	fw.write(json_string + "\n")
	fw.write(file1 + "\n")
	fw.write(file2 + "\n")
	json_string = json.dumps(id_class_dict, indent=4)
	fw.write(json_string + "\n")
	fw.write(input_dataset + "\n")
	fw.write(str(num_trials) + "\n")

cc = Classifier(										# [G]
	classifier="gene",
	gene_class_dict=gene_dict,
	max_num_spots=10_000,
	freeze_layers=4,
	num_crossval_splits=1,
	forward_batch_size=50,
	nproc=16,
	training_args=training_args,
	cust_id_class_dict=id_class_dict,
	ray_config=ray_config,
)

# /home/qian/STFormer_neighbor.dataset
cc.prepare_data(
	input_data_file=input_dataset,
	o_directory=output_dir,
	o_prefix=output_prefix,
)

all_metrics = cc.validate(
	model_directory=(									 # [F]
		"/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/"
		"Extended.model/spatial_models"
	),
	prepared_input_data_file=(
		f"{output_dir}/{output_prefix}_labeled.dataset"
	),
	id_class_dict_file=(
		f"{output_dir}/{output_prefix}_id_class_dict.pkl"
	),
	o_directory=output_dir,
	o_prefix=output_prefix,
	n_hyperopt_trials=num_trials,
)
# output_directory=output_dir,  output_prefix=output_prefix
```

The important settings are:
- **file1** (see `line [A]`): Positive gene-set (ganitumab sensitive genes)
- **file2** (see `line [B]`: Negative gene-set (randomly selected genes)
- **ray_config** (see `line [C]`): The fine-tuning settings, including the settings to iterate through: epochs, learning_rate, weight_decay, warmup_steps, and batch_size. Adjust batch_size according to your GPU memory.
- **num_trials** (see `line [D]`): Number of Ray Tuning trials (recommend around 50-60).
- **input_dataset** (see `line [E]`): Input ST dataset to be used for training purpose (in our case TNBC ST samples).
- **model_directory** (see `line [F]`): Location of the pretrained model, which fine-tuning will begin from
- **Classifier** settings (see `line [G]`): 
  - **max_num_spots** (the maximum number of spots from input_dataset to take for training purpose)
  - **classifier** (the type of classifier, in this case, "gene")
  - **num_crossval_splits** (1 for 1-split, i.e. 2-fold cross validation, use one fold for training, the other fold for evaluation/model selection. Here split refers to training gene-set split.)
  - **freeze_layers** (top 4 layers will be frozen. Leaving 2 trainable layers).
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
Current time: 2025-09-07 08:59:45. Total running time: 7hr 8min 5s
Logical resource usage: 16.0/192 CPUs, 1.0/2 GPUs (0.0/1.0 accelerator_type:RTX)
╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name			status		 num_train_epochs	 learning_rate	 weight_decay   lr_scheduler_type	   warmup_steps	  seed	 ..._train_batch_size	 iter	 total time (s)	 eval_loss	 eval_accuracy	 eval_macro_f1	 eval_runtime │
├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ _objective_1dc48785   TERMINATED					1		0.00232452		0.0437572   linear					  30.0566	517.358					   10		1			850.668	  3.13898		   0.566923		  0.566538		  232.827 │
│ _objective_a9b80c27   TERMINATED					1		0.00277271		0.0388539   cosine					   9.8573	152.935					   10		1			837.629	  3.31957		   0.557544		  0.555755		  228.765 │
│ _objective_610f942f   TERMINATED					1		0.00973166		0.0144145   polynomial				   9.94429   698.435					   10		1			837.618	  1.51457		   0.58768		   0.556975		  228.655 │
│ _objective_ebaa4529   TERMINATED					1		0.00319652		0.0397937   cosine					   9.53568   104.821					   10		1			855.184	  3.29944		   0.568579		  0.563312		  233.989 │
│ _objective_0296504d   TERMINATED					1		0.00109645		0.0443567   linear					  37.3027	157.404					   10		1			843.677	  2.10092		   0.687575		  0.68485		   230.807 │
│ _objective_b922373d   TERMINATED					1		0.00419791		0.016334	linear					  35.8125	688.271					   10		1			855.148	  3.3336			0.636482		  0.628098		  233.312 │
│ _objective_1a4cc208   TERMINATED					1		0.0014652		 0.0155965   linear					  46.1446	551.116					   10		1			838.652	  2.51904		   0.641277		  0.636301		  227.827 │
│ _objective_f39dd638   TERMINATED					1		0.00113723		0.0398002   linear					  46.9489	114.498					   10		1			855.974	  2.33628		   0.645927		  0.644991		  233.999 │
│ _objective_478c0182   TERMINATED					1		0.00429594		0.0260805   polynomial				  35.6479	215.88						10		1			843.911	  3.83718		   0.553345		  0.553329		  229.727 │
│ _objective_61540bf4   TERMINATED					1		0.00132082		0.0201103   polynomial				  16.3034	499.33						10		1			854.916	  3.75565		   0.544483		  0.543482		  233.07  │
│ _objective_65a612e9   TERMINATED					1		0.00347314		0.0275853   polynomial				  23.3297	770.11						10		1			839.982	  3.06961		   0.664181		  0.663833		  227.751 │
│ _objective_ebd8d468   TERMINATED					1		0.00167293		0.0281167   linear					  30.87	  688.736					   10		1			855.332	  2.13172		   0.715154		  0.710487		  233.252 │
│ _objective_c637877a   TERMINATED					1		0.00461591		0.0231051   polynomial				  23.1717	486.292					   10		1			840.292	  3.4724			0.621857		  0.621842		  228.292 │
│ _objective_f0295c06   TERMINATED					1		0.00755757		0.0105823   cosine					  19.1122	730.947					   10		1			856.581	  2.67375		   0.461889		  0.459491		  234.677 │
│ _objective_83aff48c   TERMINATED					1		0.00102474		0.0109389   cosine					   8.51627   132.125					   10		1			840.324	  3.241			 0.552179		  0.549481		  228.623 │
│ _objective_fbeb5515   TERMINATED					1		0.0018195		 0.0185128   linear					  33.6209	795.222					   10		1			856.646	  1.73352		   0.714677		  0.71346		   233.821 │
│ _objective_be328951   TERMINATED					1		0.0076423		 0.0312532   linear					  14.6508	533.617					   10		1			836.478	  0.689475		  0.565532		  0.361239		  226.517 │
│ _objective_39f5105c   TERMINATED					1		0.00234742		0.0283835   linear					  16.8451	314.27						10		1			856.104	  2.44849		   0.639197		  0.637024		  233.513 │
│ _objective_e3b93126   TERMINATED					1		0.00153816		0.0262491   linear					  49.7342	729.775					   10		1			844.831	  2.40262		   0.68372		   0.682295		  231.314 │
│ _objective_c0c7e661   TERMINATED					1		0.00529355		0.0415942   cosine					  46.3137	548.551					   10		1			856.828	  2.29004		   0.578858		  0.56346		   233.984 │
│ _objective_f113fbbc   TERMINATED					1		0.0018648		 0.0364315   linear					  29.5542	999.956					   10		1			840.819	  1.96297		   0.698543		  0.690721		  228.959 │
│ _objective_d14e4a6c   TERMINATED					1		0.00190205		0.0331797   linear					  39.8788	958.548					   10		1			855.59	   3.50832		   0.556299		  0.548967		  233.31  │
│ _objective_92e114db   TERMINATED					1		0.00186717		0.0206276   linear					  41.1123	875.348					   10		1			841.357	  2.62654		   0.610174		  0.607872		  229.363 │
│ _objective_484c4788   TERMINATED					1		0.00235879		0.0477765   linear					  32.0186	869.49						10		1			855.999	  3.32662		   0.588038		  0.586852		  233.764 │
│ _objective_9312a8d8   TERMINATED					1		0.00128733		0.0220134   linear					  25.1188	835.031					   10		1			840.983	  2.76094		   0.605723		  0.602959		  229.051 │
│ _objective_31081fa9   TERMINATED					1		0.00166244		0.0340794   linear					  33.066	 613.265					   10		1			856.764	  2.69681		   0.637276		  0.632353		  233.835 │
│ _objective_fc0845d0   TERMINATED					1		0.00269142		0.0185441   linear					  26.7221	625.997					   10		1			840.736	  2.77315		   0.60608		   0.605194		  228.488 │
│ _objective_317b4b62   TERMINATED					1		0.00217144		0.0239854   linear					  42.2732	387.395					   10		1			856.907	  2.10095		   0.678991		  0.678959		  234.215 │
│ _objective_6f7f18df   TERMINATED					1		0.00122541		0.0311965   linear					  29.2232	949.569					   10		1			840.873	  2.52806		   0.627275		  0.626562		  228.869 │
│ _objective_7203d267   TERMINATED					1		0.00280148		0.0136298   linear					  20.8423	801.197					   10		1			857.715	  2.74863		   0.600874		  0.596919		  234.118 │
│ _objective_3a57a1b0   TERMINATED					1		0.00214135		0.0179926   cosine					  32.6409	427.919					   10		1			840.963	  3.0177			0.588462		  0.57897		   228.608 │
│ _objective_761d0ebe   TERMINATED					1		0.00167508		0.0356378   polynomial				  27.384	 634.1						 10		1			857.656	  2.5072			0.669373		  0.669315		  234.811 │
│ _objective_f4eac0a6   TERMINATED					1		0.00340342		0.0120684   linear					  38.619	 899.978					   10		1			841.36	   2.81845		   0.625924		  0.616459		  229.067 │
│ _objective_47e7bd84   TERMINATED					1		0.00100799		0.0246088   cosine					  42.98	  800.638					   10		1			857.054	  2.07497		   0.658471		  0.658201		  234.384 │
│ _objective_2d109069   TERMINATED					1		0.00263425		0.0289178   linear					  13.0372	666.836					   10		1			839.767	  1.93764		   0.667347		  0.662931		  227.557 │
│ _objective_6be333ec   TERMINATED					1		0.0014557		 0.0174496   linear					  34.678	 741.76						10		1			853.826	  2.45238		   0.655146		  0.655107		  232.157 │
│ _objective_424c2aa6   TERMINATED					1		0.00370022		0.0145242   polynomial				  37.726	 586.734					   10		1			837.172	  3.41914		   0.606703		  0.599073		  227.148 │
│ _objective_af8cac02   TERMINATED					1		0.00297052		0.0454518   cosine					   6.10549   994.863					   10		1			854.324	  3.74299		   0.547781		  0.541005		  234.081 │
│ _objective_5e387866   TERMINATED					1		0.00115936		0.0219469   linear					  30.9653	679.402					   10		1			836.387	  2.85013		   0.618651		  0.616687		  227.03  │
│ _objective_810d65ec   TERMINATED					1		0.00204092		0.0376742   polynomial				  48.8749	263.421					   10		1			851.743	  2.68117		   0.638376		  0.630793		  232.678 │
│ _objective_de1c920d   TERMINATED					1		0.00137638		0.0259981   linear					  35.5675	457.1						 10		1			835.582	  2.06346		   0.692343		  0.686582		  226.544 │
│ _objective_77fc4f28   TERMINATED					1		0.00173549		0.0306283   linear					  43.4734	922.679					   10		1			851.45	   2.97482		   0.594383		  0.594374		  232.904 │
│ _objective_4c7ab030   TERMINATED					1		0.00242761		0.01962	 polynomial				  22.7803	832.543					   10		1			835.964	  2.39486		   0.631832		  0.631801		  226.958 │
│ _objective_db146aac   TERMINATED					1		0.00498453		0.0157725   cosine					  20.5471	334.01						10		1			851.137	  3.00558		   0.57397		   0.568391		  232.608 │
│ _objective_bd6b437f   TERMINATED					1		0.0010867		 0.0330386   linear					  24.9651	590.245					   10		1			836.448	  2.26042		   0.654881		  0.654676		  226.739 │
│ _objective_e6c4b29b   TERMINATED					1		0.00158575		0.0404635   linear					  33.8641	783.162					   10		1			851.151	  1.93337		   0.698649		  0.696031		  232.861 │
│ _objective_fdd1023e   TERMINATED					1		0.00388729		0.0126525   polynomial				  29.0741	719.556					   10		1			835.132	  4.26464		   0.521447		  0.515092		  227.151 │
│ _objective_4157544a   TERMINATED					1		0.00656644		0.0101063   cosine					  44.1427	653.672					   10		1			849.714	  2.37053		   0.643264		  0.64268		   232.883 │
│ _objective_54d9c00d   TERMINATED					1		0.00303485		0.0216367   linear					  36.5522	841.677					   10		1			835.798	  3.09551		   0.643807		  0.643431		  227.508 │
│ _objective_04cdbc1f   TERMINATED					1		0.00142795		0.0271777   linear					  39.6602	761.828					   10		1			851.557	  2.34888		   0.658167		  0.65812		   233.272 │
│ _objective_7ba01182   TERMINATED					1		0.00936795		0.0420161   polynomial				  18.3313	698.403					   10		1			831.188	  0.689851		  0.565532		  0.361239		  224.869 │
│ _objective_5218e7b8   TERMINATED					1		0.00571075		0.0298028   linear					  47.8108	472.428					   10		1			850.759	  2.90571		   0.559226		  0.557357		  233.18  │
│ _objective_ca7d9a32   TERMINATED					1		0.00197705		0.0165173   cosine					  11.6517	520.531					   10		1			836.427	  1.74282		   0.709816		  0.701522		  228.064 │
│ _objective_901a8582   TERMINATED					1		0.00125394		0.024955	linear					  45.1468	961.157					   10		1			849.744	  2.89528		   0.614863		  0.614302		  232.176 │
│ _objective_d0e36fd4   TERMINATED					1		0.00252048		0.0499872   linear					  25.665	 180.739					   10		1			835.563	  2.485			 0.61379		   0.609615		  227.66  │
│ _objective_b05f3481   TERMINATED					1		0.00220876		0.0231997   linear					  41.3017	571.662					   10		1			850.508	  1.99122		   0.717181		  0.710225		  232.763 │
│ _objective_5d1f5e76   TERMINATED					1		0.00326528		0.0194998   polynomial				  30.595	 379.391					   10		1			835.903	  3.47857		   0.59845		   0.591868		  227.436 │
│ _objective_c5ff6d76   TERMINATED					1		0.00179556		0.0326214   linear					  22.252	 873.498					   10		1			849.919	  2.69753		   0.634084		  0.629904		  232.303 │
│ _objective_93223c8f   TERMINATED					1		0.00438925		0.0356896   cosine					  15.6572	913.064					   10		1			834.983	  3.09538		   0.600265		  0.597005		  227.173 │
│ _objective_24b8d655   TERMINATED					1		0.00153992		0.0280158   linear					  31.9625	501.111					   10		1			848.507	  2.30641		   0.677507		  0.672612		  231.282 │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
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
-rw-rw-r-- 1 qian qian	  683 Nov 18 07:14 config.json
-rw-rw-r-- 1 qian qian	 5240 Nov 18 07:14 training_args.bin
-rw-rw-r-- 1 qian qian 35208744 Nov 18 07:14 model.safetensors
-rw-rw-r-- 1 qian qian	 1064 Nov 18 07:14 scheduler.pt
-rw-rw-r-- 1 qian qian 53559546 Nov 18 07:14 optimizer.pt
-rw-rw-r-- 1 qian qian	 2672 Nov 18 07:14 trainer_state.json
-rw-rw-r-- 1 qian qian	14180 Nov 18 07:14 rng_state.pth
```

<br><br>

## In Silico Gene Perturbation using the Fine-tuned CancerSTFormer Model

([Back to main &uarr;](#contents))

### Step 1: Define gene to perturb

([Back to main &uarr;](#contents))

Define the gene to be perturbed. See `immune.gene.set` ([file](immune.gene.set)):

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

```python
import os
import pickle
import sys  # may be used later
from typing import Optional

import numpy as np
import torch

from new_emb_extractor import EmbExtractor  # kept in case of side effects
from new_in_silico_perturber import InSilicoPerturber
from new_in_silico_perturber_stats import InSilicoPerturberStats

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()

CUR = "/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/Extended.model"


def run_perturb(
	model_path: str,
	dataset_path: str,
	out_dir: str,
	num_embs: Optional[int] = None,  # kept for API compatibility
) -> None:
	"""Run in silico perturbations for a list of genes."""

	# Import list of genes to perturb
	genes_perturb = list(
		np.loadtxt("immune.gene.set", dtype=str, ndmin=1),
	)
	# genes_perturb = list(np.loadtxt("tnbc.ligands", dtype=str))  # good
	# genes_perturb = list(np.loadtxt("checkpoint.combo", dtype=str))  # good
	# genes_perturb = list(np.loadtxt("CD40.query", dtype=str))

	file_path = f"{CUR}/SpatialModel/new_token_dictionary.pickle"

	# Open the file in read-binary mode and load it with pickle
	with open(file_path, "rb") as file:
		token_dictionary = pickle.load(file)

	# Translate gene names to those present in the token dictionary
	good_genes_final = [
		gene for gene in genes_perturb if gene in token_dictionary
	]

	for gene in good_genes_final:
		try:
			print(f"Perturbing {gene}")
			out_dir_final = os.path.join(out_dir, gene)
			os.makedirs(out_dir_final, exist_ok=True)

			isp = InSilicoPerturber(
				perturb_type="delete",
				perturb_rank_shift=None,
				genes_to_perturb=[gene],
				combos=0,
				anchor_gene=None,
				# model_type="Pretrained",
				model_type="GeneClassifier",
				# num_classes=0,
				num_classes=2,
				emb_mode="spot_and_gene",
				spot_emb_style="mean_pool",
				# filter_data=filter_data_dict,
				max_num_spots=1000,
				emb_layer=-1,
				forward_batch_size=80,
				nproc=1,
			)

			isp.perturb_data(
				model_path,
				dataset_path,
				out_dir_final,
				os.path.basename(dataset_path).replace(
					".dataset",
					"_emb",
				),
			)

			ispstats = InSilicoPerturberStats(
				mode="aggregate_gene_shifts",
				genes_perturbed=[gene],
				combos=0,
				anchor_gene=None,
			)

			ispstats.get_stats(
				out_dir_final,
				None,
				out_dir_final,
				os.path.basename(dataset_path).replace(
					".dataset",
					"_emb",
				),
			)
		except Exception as exc:
			print(f"Error perturbing gene {gene}: {exc}")


run_id = "run-8eb93bdf"
out_dir = f"TNBC_Gene_Shift_Single_Gene_1000/{run_id}"
os.makedirs(out_dir, exist_ok=True)

model_path = f"{run_id}/checkpoint-1000"
dataset_path = "STFormer_TNBC_neighbor.dataset"

run_perturb(model_path, dataset_path, out_dir)
```

We should next run the perturbation codes:
```
python3 run_perturb_finetuned.py
```

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
0,8060,CD274,CD274,cell_emb,,,0.9912642566400154,0.0015837076613916814,214
34,8060,CD274,CD274,13960,IGHG4,IGHG4,0.866122852072461,0.17187523821186365,262
3462,8060,CD274,CD274,7937,CCL18,CCL18,0.8920158496717128,0.17698367946749904,164
2959,8060,CD274,CD274,17998,MTRNR2L12,MTRNR2L12,0.951243766480022,0.13733519345916714,90
57,8060,CD274,CD274,24288,RPS6,RPS6,0.9530474790435396,0.11935718020258795,367
2,8060,CD274,CD274,24459,S100A9,S100A9,0.9533976870424608,0.11326944617889367,255
14575,8060,CD274,CD274,18032,MUC5AC,MUC5AC,0.961601734161377,0.0,1
6356,8060,CD274,CD274,17394,MGP,MGP,0.9616265628072951,0.10053487156061684,216
116,8060,CD274,CD274,28931,WARS,WARS,0.9644228109745396,0.09781177083814269,319
9304,8060,CD274,CD274,10771,EGFR,EGFR,0.9646104677863743,0.1077886854429446,23
9583,8060,CD274,CD274,14027,IGLC2,IGLC2,0.9655158666584304,0.10560978125850723,343
3439,8060,CD274,CD274,17428,MIEN1,MIEN1,0.966790682418756,0.1016808775793012,227
7137,8060,CD274,CD274,11793,FDCSP,FDCSP,0.9680494996262532,0.08658773529773071,204
13808,8060,CD274,CD274,10702,EDN2,EDN2,0.969096840552564,0.10401673378681789,53
1137,8060,CD274,CD274,10699,EDF1,EDF1,0.9706715152813838,0.054396667704871525,195
14579,8060,CD274,CD274,3913,AKR7A3,AKR7A3,0.9713879823684692,0.0,1
11734,8060,CD274,CD274,28950,WASIR1,WASIR1,0.9730733931064606,0.000132828950881958,2
15905,8060,CD274,CD274,899,AC009682.1,AC009682.1,0.9739029407501221,0.0,1
260,8060,CD274,CD274,14026,IGLC1,IGLC1,0.9744389535727838,0.08378957325756267,241
1050,8060,CD274,CD274,29206,YBX1,YBX1,0.9748185297382195,0.07123541373098129,263
2159,8060,CD274,CD274,20811,PPP1CB,PPP1CB,0.974952310768526,0.04107911878938597,67
782,8060,CD274,CD274,13959,IGHG3,IGHG3,0.9754053573859366,0.08176528725344107,209
15357,8060,CD274,CD274,8735,CLIP3,CLIP3,0.9758907556533813,0.0,1
1493,8060,CD274,CD274,3684,ADIRF,ADIRF,0.9759589632352194,0.03142590221519379,9
1499,8060,CD274,CD274,24226,RPL6,RPL6,0.9761075350493867,0.0437755618032936,221
2511,8060,CD274,CD274,6784,BTF3,BTF3,0.9776018007545714,0.03858765281918246,157
13282,8060,CD274,CD274,21126,PRR4,PRR4,0.9778974850972494,0.015079043266699284,3
...
```

- Column 3 and column 4 are the gene perturbed (CD274 and its ENSEMBL gene ID). 
- The column `Affected_gene_name` shows the most affected gene sorted from lowest to highest `Cosine_sim_mean`. 
- The last column `N_Detections` is equally important - it shows number of spots in which the affected gene is expressed. The number here is presented out of a total of 1000 spots. For genes like CACNB2 and FXYD7, with `N_detections` of 8 and 2 respectively, they should be probably ignored due to low detections. **We therefore recommend filtering genes based on `N_Detections`, or reranking the genes based on a combination of `Cosine_sim_mean` and `N_Detections`.**

<br>

We have a function to rerank genes `read_pert` (located in the file `evaluate.pr.tnbc.immunotherapy.basal.custom.py`, [file](eval/evaluate.pr.tnbc.immunotherapy.basal.custom.py)). see below:

```python
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

Here we can run `read_pert()` with `detect_min` set to a cutoff like 100 - this means `N_detections` must be greater than or equal to 100 for the gene to be counted, else it is pushed to the bottom of the ranking.

<br>

### Step 4: Evaluate the Perturbation Results

([Back to main &uarr;](#contents))

Evaluation should be done in TNBC_Gene_Shift_run-XX folder.
```bash
pwd
#should display TNBC_Gene_Shift_run-d64b2a10
#if not, cd TNBC_Gene_Shift_run-d64b2a10
```

Copy evaluation script from repository `eval` folder.
```bash
cp ../eval/evaluate.pr.tnbc.immunotherapy.basal.custom.py .
```

This is the python script `evaluate.pr.tnbc.immunotherapy.basal.custom.py` ([file](eval/evaluate.pr.tnbc.immunotherapy.basal.custom.py)). Look at the code:
```python
import sys

import numpy as np
from scipy.stats import hypergeom


def read_lr_targets(filename):
    """Read ligand–receptor targets into a dict keyed by ligand."""
    by_ligand = {}
    with open(filename) as f:
        header = f.readline().rstrip("\n").split("\t")[1:]
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t")
            pairs = zip(header, parts[1:])
            for ligand, value in pairs:
                by_ligand.setdefault(ligand, [])
                by_ligand[ligand].append(value)
    return by_ligand


def read_target_list(filename):
    """Read target list from file (second column per line)."""
    targets = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip("\n")
            targets.append(line.split("\t")[1])
    return targets


def pr_from_ranklist(ranklist, gold):
    """Compute precision–recall arrays from a ranked list and gold set."""
    tp = 0
    precisions, recalls = [], []
    g_size = len(gold)  # number of true positives in gold standard

    for i, gene in enumerate(ranklist, 1):  # 1-based rank
        if gene in gold:
            tp += 1
        precisions.append(tp / i)
        recalls.append(tp / g_size)

    return np.asarray(recalls), np.asarray(precisions)


def read_pert(gene_id, checkpoint, detect_min=-1):
    """
    Read perturbed genes for a given gene/checkpoint.

    CSV columns:
        ,Perturbed,Gene_name,Ensembl_ID,Affected,Affected_gene_name,
        Affected_Ensembl_ID,Cosine_sim_mean,Cosine_sim_stdev,N_Detections

    Example row:
        0,4242,CD274,ENSG00000120217,cell_emb,,,0.99,0.0043,1000
    """
    filepath = (
        f"{checkpoint}/{gene_id}/STFormer_TNBC_neighbor_emb.csv"
    )

    gene_list = []
    filtered = []

    with open(filepath) as f:
        header = f.readline().rstrip("\n").split(",")
        for line in f:
            line = line.rstrip("\n")
            parts = line.split(",")
            ndetect = int(parts[-1])

            if detect_min != -1 and ndetect < detect_min:
                filtered.append(parts[5])
                continue

            gene = parts[5]
            gene_list.append(gene)

    gene_list = gene_list + filtered
    return gene_list


def read_conversion(filename):
    """Read symbol–Ensembl conversion pairs."""
    mapping = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t")
            mapping.append((parts[0], parts[1]))
    return mapping


if __name__ == "__main__":
    by_ligand = read_lr_targets(
        "/media/scandisk/Perturbation/"
        "pdcd1.ispy2.basal.targets.txt",
    )

    target_list = read_target_list("../profiles.targets.txt")

    checkpoint = sys.argv[1]

    out_fold_pr = (
        f"{checkpoint}/TNBC_immunotherapy_basal_"
        "fold_pr_over_random.txt"
    )
    out_pr = f"{checkpoint}/TNBC_immunotherapy_basal_pr.txt"
    out_recall = (
        f"{checkpoint}/TNBC_immunotherapy_basal_recall.txt"
    )

    with open(out_fold_pr, "w") as fw1, \
            open(out_pr, "w") as fw2, \
            open(out_recall, "w") as fw3:

        for sym in ["CD274", "PDCD1", "CTLA4"]:
            for direction in ["up", "down"]:
                pert_genes = read_pert(
                    sym,
                    checkpoint,
                    detect_min=int(sys.argv[2]),
                )
                print("Pert gene", sym)

                eval_genes = list(
                    set(target_list) & set(pert_genes),
                )

                l_target = [
                    g
                    for g in by_ligand[f"PDCD1_{direction}"]
                    if g in eval_genes
                ]

                # predicted ligand targets
                pred = [
                    g for g in pert_genes[:500] if g in eval_genes
                ]
                ov = set(l_target) & set(pred)

                n_total = len(eval_genes)
                k_targets = len(l_target)
                n_pred = len(pred)
                k_overlap = len(ov)

                p_val = hypergeom.sf(
                    k_overlap - 1,
                    n_total,
                    k_targets,
                    n_pred,
                )
                logp = -1.0 * np.log10(p_val)

                exp_k = hypergeom.isf(
                    0.50,
                    n_total,
                    k_targets,
                    n_pred,
                ) + 1

                fold_over_random = len(ov) / exp_k

                recall, precision = pr_from_ranklist(
                    pert_genes,
                    set(l_target),
                )

                # 0.00, 0.01, …, 1.00
                grid = np.arange(0.00, 1.01, 0.01)

                interp_prec = [
                    precision[recall >= r].max()
                    if np.any(recall >= r)
                    else 0.0
                    for r in grid
                ]

                baseline_pr = len(l_target) / len(pert_genes)
                fold_pr = [
                    ip / baseline_pr for ip in interp_prec
                ]

                fw1.write(
                    " ".join(f"{fr:f}" for fr in fold_pr) + "\n",
                )
                fw2.write(
                    " ".join(f"{pr:f}" for pr in interp_prec) + "\n",
                )
                fw3.write(
                    " ".join(f"{re:f}" for re in grid) + "\n",
                )
```

To use the evaluation script, you must define the gold-standard genes, i.e., PD-1 targets. See line:
`pdcd1.ispy2.basal.targets.txt`:

#### Defining gold standard genes

Before running the evaluation script, we need to define gold standard genes. Here is an example of gold-standard genes, which is top 200 PDCD1-upregulated and PDCD1-downregulated genes of a held-out cohort. See `pdcd1.ispy2.basal.targets.txt` ([file](eval/pdcd1.ispy2.basal.targets.txt)).

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
- The content of `TNBC_immunotherapy_PDCD1_fold_pr_over_random.txt` shows the **Precision-Over-Radom values**, over the same recalls as above. For example, the values below are the fold-precision-over-random values.
```
3.470265 3.470265 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 1.006523 ...
```

You can easily use ggplot2 or matplotlib (Python) to plot the PR curve.

