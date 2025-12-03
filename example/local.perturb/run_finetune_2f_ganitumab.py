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

choice = sys.argv[1]
direction = sys.argv[2]

file1, file2 = None, None

if direction=="up":
	file1 = "ganitumab.upregulated.top300"
	file2 = "gene.shuffled.upregulated"
elif direction=="down":
	file1 = "ganitumab.downregulated.top300"
	file2 = "gene.shuffled.downregulated"
	
#file2 = "../tnbc.responder.spatialmodel/ganitumab/%s/downregulated.top300" % choice

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

'''
ray_config = {"num_train_epochs": [20,],
"learning_rate": (1e-6, 1e-5),
"weight_decay": (0.16, 0.35),
"lr_scheduler_type": ["linear", "cosine", "polynomial"],
"warmup_steps": (100, 2000),
"seed": (0, 100),
"per_device_train_batch_size": [12,],
}
'''

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

cc = Classifier(classifier="gene", gene_class_dict = gene_dict, 
max_num_spots = 10_000, freeze_layers = 4, num_crossval_splits = 1,
forward_batch_size=200, nproc=16, training_args = training_args, cust_id_class_dict = id_class_dict, ray_config=ray_config, filter_data=filter_data_dict)

cc.prepare_data(input_data_file=input_dataset, o_directory=output_dir, o_prefix=output_prefix)

all_metrics = cc.validate(model_directory="/media/stu.backup2/Qian/ivy.codes/cancerstformer/data/models",
prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
o_directory=output_dir,  o_prefix=output_prefix, 
#n_hyperopt_trials=10)
n_hyperopt_trials=num_trials)

cc.plot_conf_mat(conf_mat_dict={"Geneformer": all_metrics["conf_matrix"]},
o_directory=output_dir,o_prefix=output_prefix,)

cc.plot_roc(roc_metric_dict={"Geneformer": all_metrics["all_roc_metrics"]},
model_style_dict={"Geneformer": {"color": "red", "linestyle": "-"}},
title="Responder vs NonResponder selected genes", o_directory=output_dir,
o_prefix=output_prefix,)

'''
output_prefix = "responder_metastasis_alldata"
output_dir = f"{datestamp}"
os.makedirs(output_dir,exist_ok=True)
cc = Classifier(classifier="gene", gene_class_dict = gene_dict,
max_ncells = 10_000, freeze_layers = 4, num_crossval_splits = 0,
forward_batch_size=200, nproc=16, training_args = training_args)

cc.prepare_data(input_data_file="../STGeneformer/STGeneformer.dataset",
output_directory=output_dir, output_prefix=output_prefix)

trainer_test = cc.train_all_data(model_directory="models",
prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
output_directory=output_dir, output_prefix=output_prefix)
'''
