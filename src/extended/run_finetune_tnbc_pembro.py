import torch
import numpy as np
import pickle
import os
import sys
from new_classifier import Classifier
import datetime
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()

choice = sys.argv[1] #shuf1, shuf2, ... shuf3, shuf4

file1 = "upregulated.top300"
file2 = "gene.shuffled.upregulated"
genes_responder = list(np.loadtxt(file1,dtype=str))
genes_random = list(np.loadtxt(file2, dtype=str))

training_args = {"num_train_epochs": 30.0, "weight_decay": 0.25, "learning_rate": 3e-6, "warmup_steps":1500, "lr_scheduler_type": "polynomial"}

ray_config = {"num_train_epochs": [1.0,],
"learning_rate": (1e-3, 1e-2),
"weight_decay": (0.01, 0.05),
"lr_scheduler_type": ["linear", "cosine", "polynomial"],
"warmup_steps": (5, 50),
"seed": (100, 1000),
"per_device_train_batch_size": [10,],
}

ensemble_dictionary = {}
with open("gene_id_dictionary.pickle", 'rb') as file:
	ensemble_dictionary = pickle.load(file)


good_genes = [ensemble_dictionary[gene] for gene in genes_responder if gene in ensemble_dictionary]
good_random = [ensemble_dictionary[gene] for gene in genes_random if gene in ensemble_dictionary]

gene_dict = {"Responder": good_genes, "Random.genes": good_random}
id_class_dict = {1: "Responder", 0: "Random.genes"}

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

output_prefix = "responder_test"
output_dir  = f"{datestamp}"
os.makedirs(output_dir,exist_ok=True)

input_dataset = "/media/qian/vol2/Qian/models.to.test/Extended.model/STFormer_TNBC_neighbor.dataset"
num_trials = 60

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
forward_batch_size=50, nproc=16, training_args = training_args, cust_id_class_dict = id_class_dict, ray_config = ray_config)

#/home/qian/STFormer_neighbor.dataset
cc.prepare_data(input_data_file=input_dataset, output_directory=output_dir, output_prefix=output_prefix)

all_metrics = cc.validate(model_directory="/media/qian/vol2/Qian/models.to.test/Extended.model/spatial_models",
prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
output_directory=output_dir,  output_prefix=output_prefix, n_hyperopt_trials=num_trials)
#output_directory=output_dir,  output_prefix=output_prefix)
